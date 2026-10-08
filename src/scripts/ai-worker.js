import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  TextStreamer,
  env
} from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';

// Guard against Emscripten pthread sub-worker re-initialization
const isPthread = typeof self !== 'undefined' && self.name?.startsWith('em-pthread');

if (!isPthread) {
  env.allowLocalModels = false;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;

  const MODEL_ID = 'onnx-community/LFM2.5-VL-450M-ONNX';
  const DTYPE = 'q4f16';

  let processor = null;
  let model = null;

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
        description: 'Step-by-step visual reasoning: inspect leaf discoloration, lesions, concentric rings, chlorosis, necrosis, veins, and evaluate pathogen type before reaching diagnosis. Even think about fixes and remedies too along with preventions if any disesase is present. Think hard.'
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

    self.postMessage({
      type: 'status',
      status: 'loading',
      message: 'Loading LFM2.5-VL-450M (q4f16 WebGPU)...'
    });

    const MODEL_FILE_COUNT = 3;
    const progressMap = new Map();
    const progressCallback = (info) => {
      if (info.status !== 'progress' || !info.file?.endsWith('.onnx_data') || !info.total) {
        return;
      }
      progressMap.set(info.file, info.loaded / info.total);
      const totalProgress = (Array.from(progressMap.values()).reduce((sum, v) => sum + v, 0) / MODEL_FILE_COUNT) * 100;
      self.postMessage({
        type: 'progress',
        progress: {
          file: info.file,
          progress: Math.min(100, Math.round(totalProgress))
        }
      });
    };

    try {
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback
      });

      model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
        dtype: {
          vision_encoder: 'fp16',
          embed_tokens: 'fp16',
          decoder_model_merged: 'q4f16'
        },
        device: 'webgpu',
        progress_callback: progressCallback
      });

      self.postMessage({
        type: 'status',
        status: 'ready',
        message: 'LFM2.5-VL-450M WebGPU loaded and ready!'
      });

      return { processor, model };
    } catch (err) {
      console.error('Failed to load LFM2.5-VL model:', err);
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
          message: 'LFM2.5-VL reasoning over crop symptoms...'
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
Examine this crops carefully. Reason first good length. Try to answer all points plz and Think long to identify real disease correctly. DO not fake the disease.
${userNote ? `Farmer Note: "${userNote}"` : ''}
Respond strictly in JSON matching this schema:
${JSON.stringify(DIAGNOSIS_SCHEMA, null, 2)}
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

        const promptText = processor.apply_chat_template(messages, {
          tokenize: false,
          add_generation_prompt: true
        });

        // 3. Prepare inputs with correct argument order (images, text)
        const inputs = await processor(rawImage, promptText, {
          add_special_tokens: false
        });

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
          max_new_tokens: 1024,
          repetition_penalty: 1.08,
          do_sample: false,
          streamer: streamer,
          logits_processor: [structuredProcessor]
        });

        // Slicing and JSON Parsing: prefer rawTokens from streamer (skip_prompt: true)
        const promptLen = inputs.input_ids?.dims?.at(-1) || 0;
        const genTokens = promptLen > 0 ? output.slice(null, [promptLen, null]) : output;
        const decoded = processor.tokenizer.batch_decode(genTokens, { skip_special_tokens: true });
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
