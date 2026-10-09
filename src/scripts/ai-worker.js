import {
  AutoProcessor,
  AutoConfig,
  Lfm2ForCausalLM,
  RawImage,
  TextStreamer,
  Tensor,
  env
} from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/webgpu';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';
import { getKernel } from '@huggingface/kernels';
import { initMegaKernel } from './megakernel-client.js';
import { createLfmEngine, requestLfmDevice, MAX_NEW_TOKENS } from './lfm-engine/engine.js';
import { blobSource } from './lfm-engine/weights.js';

// Guard against Emscripten pthread sub-worker re-initialization
const isPthread = typeof self !== 'undefined' && self.name?.startsWith('em-pthread');

if (!isPthread) {
  if (typeof self !== 'undefined' && self.location && self.location.origin) {
    const isLocal = self.location.hostname === 'localhost' || self.location.hostname === '127.0.0.1';
    if (!isLocal) {
      env.remoteHost = self.location.origin + '/hf';
    }
  }
  env.allowLocalModels = false;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;

  const MODEL_ID = 'onnx-community/LFM2.5-VL-450M-ONNX';
  const DTYPE = 'q4f16';

  // Staged load: vision stage (vision_encoder + embed_tokens) runs first and is released,
  // then the text stage (embed_tokens + decoder) runs. Both never sit on the GPU together.
  const VISION_FILES = [`vision_encoder_${DTYPE}.onnx`, `vision_encoder_${DTYPE}.onnx_data`];
  const EMBED_FILES = [`embed_tokens_${DTYPE}.onnx`, `embed_tokens_${DTYPE}.onnx_data`];
  const DECODER_FILES = [`decoder_model_merged_${DTYPE}.onnx`, `decoder_model_merged_${DTYPE}.onnx_data`];

  let processor = null;
  let imageTokenId = null;
  let cachedEngine = null;

  function modelFileUrl(file) {
    const host = env.remoteHost.endsWith('/') ? env.remoteHost : env.remoteHost + '/';
    return `${host}${MODEL_ID}/resolve/main/onnx/${file}`;
  }

  // Download a model file into the same Cache API cache transformers.js uses,
  // so the text stage (loaded by transformers.js) finds it without a second download.
  async function fetchModelFile(file, onProgress) {
    const url = modelFileUrl(file);
    const cache = typeof caches !== 'undefined' ? await caches.open(env.cacheKey || 'transformers-cache') : null;
    let res = cache ? await cache.match(url) : null;
    if (res) {
      const bytes = new Uint8Array(await res.arrayBuffer());
      onProgress?.(file, bytes.length, bytes.length);
      return bytes;
    }
    const net = await fetch(url);
    if (!net.ok) throw new Error(`HTTP ${net.status} fetching ${file}`);
    const total = Number(net.headers.get('content-length')) || 0;
    const reader = net.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.(file, loaded, total || loaded);
    }
    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.length; }
    if (cache) {
      try {
        await cache.put(url, new Response(bytes, { headers: { 'content-length': String(loaded) } }));
      } catch (e) {
        console.warn('[ai-worker] Cache put failed for', file, e.message);
      }
    }
    return bytes;
  }

  // Same cache as fetchModelFile, but returns a disk-backed Blob (no full copy in RAM).
  // The custom decoder engine slices tensors out of it one by one.
  async function getModelBlob(file) {
    const url = modelFileUrl(file);
    const cache = typeof caches !== 'undefined' ? await caches.open(env.cacheKey || 'transformers-cache') : null;
    let res = cache ? await cache.match(url) : null;
    if (!res) {
      const bytes = await fetchModelFile(file);
      res = cache ? await cache.match(url) : null;
      if (!res) return new Blob([bytes]);
    }
    return res.blob();
  }

  async function createStageSession(files, onProgress) {
    const [modelBytes, dataBytes] = await Promise.all(files.map((f) => fetchModelFile(f, onProgress)));
    return ort.InferenceSession.create(modelBytes, {
      executionProviders: ['webgpu'],
      externalData: [{ path: files[1], data: dataBytes }]
    });
  }

  // fp16 <-> fp32 helpers (only used when vision features and token embeddings differ in dtype)
  const _f32 = new Float32Array(1);
  const _u32 = new Uint32Array(_f32.buffer);
  function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  function floatToHalf(x) {
    _f32[0] = x;
    const b = _u32[0];
    const sign = (b >>> 16) & 0x8000;
    const exp = ((b >>> 23) & 0xff) - 127 + 15;
    const man = b & 0x7fffff;
    if (((b >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
    if (exp >= 31) return sign | 0x7bff;
    if (exp <= 0) {
      if (exp < -10) return sign;
      const m2 = man | 0x800000;
      const shift = 14 - exp;
      let h = m2 >> shift;
      if ((m2 >> (shift - 1)) & 1) h++;
      return sign | h;
    }
    let h = sign | (exp << 10) | (man >> 13);
    if (man & 0x1000) h++;
    return h;
  }
  function tensorToFloat32(t) {
    const d = t.data;
    if (d instanceof Float32Array) return d;
    if (d instanceof Uint16Array) {
      const out = new Float32Array(d.length);
      for (let i = 0; i < d.length; i++) out[i] = halfToFloat(d[i]);
      return out;
    }
    return Float32Array.from(d); // Float16Array
  }
  function writeFromFloat32(target, offset, src) {
    if (target instanceof Uint16Array) {
      for (let i = 0; i < src.length; i++) target[offset + i] = floatToHalf(src[i]);
    } else {
      target.set(src, offset);
    }
  }

  async function discoverAvailableHfKernels() {
    try {
      const res = await fetch('https://huggingface.co/api/kernels?author=webgpu-kernels&limit=500');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        return new Set(data.map(k => k.id).filter(Boolean));
      }
    } catch (e) {
      console.warn('[ai-worker] Dynamic kernel registry query notice:', e.message);
    }
    return new Set();
  }

  function dataUrlToBlob(dataUrl) {
    if (dataUrl instanceof Blob) return dataUrl;
    if (typeof dataUrl === 'string' && dataUrl.startsWith('data:')) {
      const parts = dataUrl.split(',');
      const mimeMatch = parts[0].match(/:(.*?);/);
      const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
      const bstr = atob(parts[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new Blob([u8arr], { type: mime });
    }
    return dataUrl;
  }

  const DIAGNOSIS_SCHEMA = {
    type: 'object',
    properties: {
      reasoning: {
        type: 'string',
        description: 'Step-by-step visual reasoning in detail in ASD STE100 format before reaching diagnosis. Even think about fixes and remedies too along with preventions if any disesase is present. ALso, try to identify the disease but do not try to force identify it as sometimes plant can be healthy if present while thinking. Think hard.'
      },
      crop: {
        type: 'string',
        description: 'Identified crop or plant species. Try to identify in reasoning time already yell name '
      },
      status: {
        type: 'string',
        description: 'Plant health condition: "healthy" if no disease or pest found, or "diseased" if visual symptoms or pest damage exist',
        enum: ['healthy', 'diseased']
      },
      disease_name: {
        type: 'string',
        description: 'Specific common or scientific name of disease identified. Identify it during reasoning time try to be more accurate based on the conditions. or "Healthy Leaf" if none. Sir, disease must be think carefully about.'
      },
      confidence: {
        type: 'string',
        description: 'Confidence level based on visual clarity and distinctness of symptoms: high, medium, or low. DOnt be lways highly fake confident. SHow real confidence.',
        enum: ['high', 'medium', 'low']
      },
      key_symptoms: {
        type: 'array',
        description: 'List of 2 to 4 observable visual symptoms seen on the leaf (e.g. concentric brown rings, water-soaked lesions, yellow margins)',
        items: {
          type: 'string',
          description: 'Observable visual symptom'
        }
      },
      organic_treatment: {
        type: 'array',
        description: 'List of actionable eco-friendly or organic control treatments. Think while reasoning',
        items: {
          type: 'string',
          description: 'Organic treatment measure'
        }
      },
      chemical_treatment: {
        type: 'array',
        description: 'List of standard synthetic chemicals or fungicides with active ingredients. Think while reasoning',
        items: {
          type: 'string',
          description: 'Chemical treatment measure'
        }
      },
      prevention_measures: {
        type: 'array',
        description: 'List of cultural practices to prevent spread and recurrence. Think while reasoning',
        items: {
          type: 'string',
          description: 'Preventative cultural measure'
        }
      }
    },
    required: [
      'reasoning',
      'crop',
      'status',
      'disease_name',
      'confidence',
      'key_symptoms',
      'organic_treatment',
      'chemical_treatment',
      'prevention_measures'
    ],
    additionalProperties: false
  };

  async function loadModel() {
    if (processor) {
      return { processor };
    }

    // Check WebGPU hardware availability first
    if (typeof navigator !== 'undefined' && !navigator.gpu) {
      const errMsg = 'WebGPU is not supported by your browser or graphics hardware.';
      self.postMessage({ type: 'error', message: errMsg });
      throw new Error(errMsg);
    }

    // -----------------------------------------------------------------
    // Tier 1: Initialize local MegaKernel bundle and discover available kernels
    // -----------------------------------------------------------------
    self.postMessage({
      type: 'status',
      status: 'loading',
      message: 'Tier 1: Loading bundled MegaKernel (200+ WebGPU kernels in memory)...'
    });

    await initMegaKernel();
    const registry = await discoverAvailableHfKernels();
    console.log(`[ai-worker] Registry contains ${registry.size} WebGPU kernels.`);

    // Candidate operator categories needed by LFM (Vision + Transformer Decoder)
    const CANDIDATE_PATTERNS = [
      'RotaryEmbedding', // RoPE (ai.onnx.RotaryEmbedding / com.microsoft.RotaryEmbedding)
      'Softmax',         // Softmax (ai.onnx.Softmax / com.microsoft.BiasSoftmax)
      'Attention',       // Attention / GQA (ai.onnx.Attention / LinearAttention / GroupQueryAttention)
      'Normalization',   // LayerNorm / SkipSimplifiedLayerNormalization
      'MatMul',          // Matrix Multiplication
      'Conv'             // Patch Convolutions for Vision Encoder
    ];

    const targetKernels = [];
    if (registry.size > 0) {
      for (const kernelId of registry) {
        if (CANDIDATE_PATTERNS.some(pat => kernelId.includes(pat))) {
          targetKernels.push(kernelId);
        }
      }
    } else {
      // Direct fallback if registry list request is restricted
      targetKernels.push(
        'webgpu-kernels/ai.onnx.RotaryEmbedding',
        'webgpu-kernels/ai.onnx.Softmax',
        'webgpu-kernels/ai.onnx.Attention',
        'webgpu-kernels/ai.onnx.Conv',
        'webgpu-kernels/ai.onnx.LayerNormalization',
        'webgpu-kernels/ai.onnx.MatMul'
      );
    }

    const loadedCustomKernels = [];
    for (const repoId of targetKernels) {
      try {
        await getKernel(repoId, { version: 1 });
        const cleanName = repoId.replace('webgpu-kernels/', '');
        loadedCustomKernels.push(cleanName);
        console.log(`[ai-worker] Tier 1: Auto-fetched custom kernel: ${cleanName}`);
      } catch (e) {
        // Kernels that are not yet compiled or optional can safely be skipped
      }
    }

    const tierInfo = loadedCustomKernels.length > 0
      ? `WebGPU + ${loadedCustomKernels.length} Custom Kernels (${loadedCustomKernels.slice(0, 4).join(', ')}${loadedCustomKernels.length > 4 ? '...' : ''})`
      : 'WebGPU Standard';

    self.postMessage({
      type: 'status',
      status: 'loading',
      message: `Tier 2: Loading LFM2.5 Vision Model (${tierInfo})...`
    });

    const MODEL_FILE_COUNT = 3;
    const progressMap = new Map();
    const reportProgress = (file, loaded, total) => {
      if (!file?.endsWith('.onnx_data') || !total) {
        return;
      }
      progressMap.set(file, loaded / total);
      const totalProgress = (Array.from(progressMap.values()).reduce((sum, v) => sum + v, 0) / MODEL_FILE_COUNT) * 100;
      const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
      const totalMB = (total / (1024 * 1024)).toFixed(1);
      self.postMessage({
        type: 'progress',
        progress: {
          file,
          progress: Math.min(100, Math.round(totalProgress)),
          loadedMB,
          totalMB
        }
      });
    };
    const progressCallback = (info) => {
      if (info.status === 'progress') reportProgress(info.file, info.loaded, info.total);
    };

    try {
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback
      });

      const config = await AutoConfig.from_pretrained(MODEL_ID);
      imageTokenId = BigInt(config.image_token_id ?? config.image_token_index);

      // Download weights into the cache only. Nothing is uploaded to the GPU until a stage runs.
      for (const file of [...VISION_FILES, ...EMBED_FILES, ...DECODER_FILES]) {
        await fetchModelFile(file, reportProgress);
      }

      self.postMessage({
        type: 'status',
        status: 'ready',
        message: `LFM2.5 WebGPU Ready (staged load)! [${tierInfo}]`
      });

      return { processor };
    } catch (err) {
      console.error('Failed to load LFM2.5 WebGPU model:', err);
      self.postMessage({
        type: 'error',
        message: `${err.name || 'Error'}: ${err.message || 'Error initializing WebGPU model.'}`
      });
      throw err;
    }
  }

  function extractJson(text) {
    if (!text || typeof text !== 'string') {
      throw new Error('No text received to parse JSON');
    }
    const clean = text.trim();
    // 1. Direct JSON parse
    try {
      return JSON.parse(clean);
    } catch (_) {}

    // 2. Strip markdown fences ```json ... ```
    const fenceMatch = clean.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fenceMatch) {
      try {
        return JSON.parse(fenceMatch[1].trim());
      } catch (_) {}
    }

    // 3. Extract substring between first '{' and last '}'
    const firstBrace = clean.indexOf('{');
    const lastBrace = clean.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      const jsonSub = clean.substring(firstBrace, lastBrace + 1);
      try {
        return JSON.parse(jsonSub);
      } catch (_) {}
    }

    throw new Error('Failed to parse structured diagnosis JSON.');
  }

  // Stage 1: vision_encoder + embed_tokens on WebGPU, merge image features into the
  // token embeddings, copy the result to CPU, then release both sessions.
  async function runVisionStage(inputs) {
    if (ort.env?.wasm && !ort.env.wasm.wasmPaths) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
    }
    const toOrt = (t) => new ort.Tensor(t.type, t.data, t.dims);
    const copyOut = async (tensor) => {
      const out = { type: tensor.type, data: (await tensor.getData()).slice(), dims: [...tensor.dims] };
      tensor.dispose?.();
      return out;
    };

    let feats;
    let vision = null;
    try {
      vision = await createStageSession(VISION_FILES);
      const feeds = {};
      for (const name of vision.inputNames) feeds[name] = toOrt(inputs[name]);
      const result = await vision.run(feeds);
      feats = await copyOut(result[vision.outputNames[0]]);
    } finally {
      if (vision) await vision.release();
    }

    let emb;
    let embed = null;
    try {
      embed = await createStageSession(EMBED_FILES);
      const result = await embed.run({ input_ids: toOrt(inputs.input_ids) });
      emb = await copyOut(result[embed.outputNames[0]]);
    } finally {
      if (embed) await embed.release();
    }

    // Replace each image-token position with its image feature vector
    const hidden = emb.dims.at(-1);
    const nFeatures = feats.data.length / hidden;
    const ids = inputs.input_ids.data;
    const sameKind = feats.type === emb.type && feats.data.constructor === emb.data.constructor;
    const featsF32 = sameKind ? null : tensorToFloat32(feats);
    let k = 0;
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] !== imageTokenId) continue;
      if (k >= nFeatures) throw new Error('More image tokens than image features.');
      const start = k * hidden;
      if (sameKind) {
        emb.data.set(feats.data.subarray(start, start + hidden), i * hidden);
      } else {
        writeFromFloat32(emb.data, i * hidden, featsF32.subarray(start, start + hidden));
      }
      k++;
    }
    if (k !== nFeatures) {
      throw new Error(`Image tokens (${k}) and image features (${nFeatures}) do not match.`);
    }
    return new Tensor(emb.type, emb.data, emb.dims);
  }

  // Lfm2ForCausalLM drops `inputs_embeds` from generate() kwargs and has no vision methods.
  // Inject the prepared embeds into the first forward pass only; later steps embed new tokens.
  function injectInputsEmbeds(textModel, inputsEmbeds, inputs) {
    const unavailable = () => { throw new Error('Vision stage is not loaded in the text model.'); };
    textModel.encode_image = unavailable;
    textModel._merge_input_ids_with_image_features = unavailable;

    let first = true;
    const originalForward = textModel.forward.bind(textModel);
    textModel.forward = (modelInputs) => {
      if (first) {
        modelInputs.inputs_embeds = inputsEmbeds;
      } else {
        modelInputs.inputs_embeds = null;
        modelInputs.pixel_values = inputs.pixel_values;
      }
      first = false;
      return originalForward(modelInputs);
    };
  }

  // Stage 2 (custom engine): full-WebGPU f16 decoder. The JSON-schema logits processor needs the
  // logits on the CPU for every token, so generation runs in constrained (pickToken) mode.
  async function runEngineStage(inputsEmbeds, inputs, structuredProcessor, streamer, data = {}) {
    const d = inputsEmbeds.data;
    const embeds = d instanceof Float32Array || d instanceof Uint16Array ? d : Float32Array.from(d);
    const M = inputsEmbeds.dims.at(-2);
    const promptIds = Array.from(inputs.input_ids.data, Number);
    const stageT0 = performance.now();

    if (!cachedEngine) {
      const [decGraph, decBlob, embGraph, embBlob] = await Promise.all([
        getModelBlob(DECODER_FILES[0]).then((b) => b.arrayBuffer()),
        getModelBlob(DECODER_FILES[1]),
        getModelBlob(EMBED_FILES[0]).then((b) => b.arrayBuffer()),
        getModelBlob(EMBED_FILES[1])
      ]);
      const device = await requestLfmDevice();
      cachedEngine = await createLfmEngine({
        device,
        decoder: { graphBytes: new Uint8Array(decGraph), source: blobSource(decBlob) },
        embed: { graphBytes: new Uint8Array(embGraph), source: blobSource(embBlob) }
      });
    }
    const engine = cachedEngine;
    try {
      streamer.put([promptIds.map(BigInt)]); // first put = prompt, skipped by the streamer
      const result = await engine.generate({
        embeds,
        M,
        maxNew: MAX_NEW_TOKENS,
        eos: [7],
        pickToken: (logits, generated) => {
          structuredProcessor([promptIds.concat(generated)], { data: logits, dims: [1, logits.length] });
          let best = 0;
          for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
          return best;
        },
        onToken: (id) => streamer.put([[BigInt(id)]])
      });
      streamer.end();
      console.log('[ai-worker] engine', result.reason, `${result.tokens.length} tokens`, `total ${(performance.now() - stageT0).toFixed(0)} ms (incl. engine init)`, `ttft ${result.ttftMs.toFixed(0)} ms`, `${result.tokensPerSec.toFixed(1)} tok/s`, `gpu+readback ${result.waitMs.toFixed(0)} ms`, `cpu pick ${result.pickMs.toFixed(0)} ms`, `from click ${(performance.now() - (data.clickTimestamp || stageT0)).toFixed(0)} ms`, engine.stats);
      return processor.tokenizer.decode(result.tokens, { skip_special_tokens: true });
    } finally {
      // Do NOT destroy engine or device; keep resident across requests
    }
  }

  self.addEventListener('message', async (event) => {
    if (!event.data || typeof event.data !== 'object') return;
    const { type, data } = event.data;
    if (!type || (type !== 'load' && type !== 'analyze')) return;

    if (type === 'load') {
      try {
        await loadModel();
      } catch {
        // reported in loadModel
      }
      return;
    }

    if (type === 'analyze') {
      try {
        if (!processor) {
          await loadModel();
        }

        const { image, userNote } = data;

        self.postMessage({
          type: 'status',
          status: 'analyzing',
          message: 'Gemini Google Gemma4 E2B reasoning over crop symptoms...'
        });

        const structuredProcessor = new StructuredOutputProcessor(processor.tokenizer, {
          type: 'json_schema',
          json_schema: DIAGNOSIS_SCHEMA
        });

        // 1. Process image
        const imageBlob = dataUrlToBlob(image);
        const rawImage = await RawImage.fromBlob(imageBlob);

        // 2. Assemble prompt
        const systemPrompt = `You are an expert plant pathologist and agronomist.
        Respond strictly in JSON matching this schema:
${JSON.stringify(DIAGNOSIS_SCHEMA, null, 2)}.
REMINDER: EVERYTHING DOES NOT HAVE Disease. SOme thing can be HEALTHLY TOO :)
Examine this crops carefully. Reason first good length. Try to answer all points plz and Think long to identify real disease correctly. DO not fake the disease.
${userNote ? `Farmer Note: "${userNote}"` : ''}

Reasoning must be the first property.`;

        const messages = [
          {
            role: 'user',
            content: [
              { type: 'image' },
              { type: 'text', text: systemPrompt }
            ]
          }
        ];

        const promptText = processor.apply_chat_template(messages, { add_generation_prompt: true });

        // 3. Prepare inputs with correct argument order
        const inputs = await processor(rawImage, promptText, { add_special_tokens: false });

        let rawTokens = '';
        let firstTokenAt = 0;
        const makeStreamer = () => new TextStreamer(processor.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (token) => {
            if (!firstTokenAt) firstTokenAt = performance.now();
            rawTokens += token;
            self.postMessage({
              type: 'token',
              token: token,
              fullText: rawTokens
            });
          }
        });
        const streamer = makeStreamer();

        self.postMessage({
          type: 'status',
          status: 'analyzing',
          message: 'Stage 1/2: encoding image (vision weights on GPU, released afterwards)...'
        });
        const inputsEmbeds = await runVisionStage(inputs);

        self.postMessage({
          type: 'status',
          status: 'analyzing',
          message: 'Stage 2/2: loading text decoder (custom WebGPU engine) and generating...'
        });

        let decodedText = '';
        let engineOk = false;
        const forceTransformers = data.decoder === 'transformers';
        try {
          if (forceTransformers) throw new Error('decoder=transformers requested (A/B test)');
          decodedText = (await runEngineStage(inputsEmbeds, inputs, structuredProcessor, streamer, data) || '').trim();
          engineOk = true;
        } catch (engineErr) {
          console.warn('[ai-worker] Custom engine failed, falling back to transformers.js decoder:', engineErr);
          rawTokens = '';
          self.postMessage({
            type: 'status',
            status: 'analyzing',
            message: 'Custom engine unavailable, using fallback decoder...'
          });
        }

        if (!engineOk) {
          // Fresh constraint state + streamer: the failed attempt may have advanced both.
          const fbProcessor = new StructuredOutputProcessor(processor.tokenizer, {
            type: 'json_schema',
            json_schema: DIAGNOSIS_SCHEMA
          });
          const fbStreamer = makeStreamer();
          const fbT0 = performance.now();
          let genT0 = fbT0;
          let output;
          let textModel = null;
          try {
            textModel = await Lfm2ForCausalLM.from_pretrained(MODEL_ID, {
              dtype: { embed_tokens: DTYPE, decoder_model_merged: DTYPE },
              device: 'webgpu'
            });
            injectInputsEmbeds(textModel, inputsEmbeds, inputs);
            genT0 = performance.now();
            firstTokenAt = 0;

            output = await textModel.generate({
              input_ids: inputs.input_ids,
              attention_mask: inputs.attention_mask,
              max_new_tokens: MAX_NEW_TOKENS,
              do_sample: false,
              streamer: fbStreamer,
              logits_processor: [fbProcessor]
            });
          } finally {
            // Free the decoder GPU memory before the next analysis
            if (textModel) await textModel.dispose();
          }

          // Slicing and JSON Parsing: prefer rawTokens from streamer (skip_prompt: true)
          const promptLen = inputs.input_ids?.dims?.at(-1) || 0;
          const genTokens = promptLen > 0 ? output.slice(null, [promptLen, null]) : output;
          const decoded = processor.batch_decode(genTokens, { skip_special_tokens: true });
          decodedText = (decoded[0] || '').trim();
          {
            const n = genTokens.dims?.at(-1) || 0;
            const end = performance.now();
            const ttft = (firstTokenAt || end) - genT0;
            const decMs = Math.max(1, end - (firstTokenAt || genT0));
            console.log('[ai-worker] transformers.js decoder', `${n} tokens`, `total ${(end - fbT0).toFixed(0)} ms`, `load ${(genT0 - fbT0).toFixed(0)} ms`, `ttft ${ttft.toFixed(0)} ms (text-chunk granularity)`, `decode ${(((n - 1) / decMs) * 1000).toFixed(1)} tok/s`);
          }
        }
        const fullText = rawTokens.trim() || decodedText;

        let structuredResult = null;
        try {
          structuredResult = extractJson(fullText);
        } catch (parseErr) {
          console.error('JSON parsing failed:', parseErr, 'Raw output was:', fullText);
          throw parseErr;
        }

        self.postMessage({
          type: 'result',
          data: structuredResult,
          rawText: fullText
        });

        self.postMessage({
          type: 'status',
          status: 'ready',
          message: 'Diagnosis complete.'
        });
      } catch (err) {
        console.error('Error during analysis:', err);
        self.postMessage({
          type: 'error',
          message: `${err.name || 'Error'}: ${err.message || 'Error analyzing crop leaf.'}`
        });
      }
    }
  });
}
