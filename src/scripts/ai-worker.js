import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  TextStreamer,
  env
} from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';
import { getKernel } from '@huggingface/kernels';
import { initMegaKernel } from './megakernel-client.js';

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

  let processor = null;
  let model = null;

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
    if (processor && model) {
      return { processor, model };
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
    const progressCallback = (info) => {
      if (info.status !== 'progress' || !info.file?.endsWith('.onnx_data') || !info.total) {
        return;
      }
      progressMap.set(info.file, info.loaded / info.total);
      const totalProgress = (Array.from(progressMap.values()).reduce((sum, v) => sum + v, 0) / MODEL_FILE_COUNT) * 100;
      const loadedMB = (info.loaded / (1024 * 1024)).toFixed(1);
      const totalMB = (info.total / (1024 * 1024)).toFixed(1);
      self.postMessage({
        type: 'progress',
        progress: {
          file: info.file,
          progress: Math.min(100, Math.round(totalProgress)),
          loadedMB,
          totalMB
        }
      });
    };

    try {
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback
      });

      model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
        dtype: {
          vision_encoder: "q4f16",
          embed_tokens: "q4f16",
          decoder_model_merged: "q4f16",
        },
        device: 'webgpu',
        progress_callback: progressCallback
      });

      self.postMessage({
        type: 'status',
        status: 'ready',
        message: `LFM2.5 WebGPU Ready! [${tierInfo}]`
      });

      return { processor, model };
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
        if (!model || !processor) {
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
        const streamer = new TextStreamer(processor.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (token) => {
            rawTokens += token;
            self.postMessage({
              type: 'token',
              token: token,
              fullText: rawTokens
            });
          }
        });

        const output = await model.generate({
          ...inputs,
          max_new_tokens: 4096,
          do_sample: false,
          streamer: streamer,
          logits_processor: [structuredProcessor]
        });

        // Slicing and JSON Parsing: prefer rawTokens from streamer (skip_prompt: true)
        const promptLen = inputs.input_ids?.dims?.at(-1) || 0;
        const genTokens = promptLen > 0 ? output.slice(null, [promptLen, null]) : output;
        const decoded = processor.batch_decode(genTokens, { skip_special_tokens: true });
        const decodedText = (decoded[0] || '').trim();
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
