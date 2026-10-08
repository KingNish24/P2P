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
  let structuredProcessor = null;

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
        description: 'Step-by-step visual reasoning: inspect leaf discoloration, lesions, concentric rings, chlorosis, necrosis, veins, and evaluate pathogen type before reaching diagnosis.'
      },
      crop: {
        type: 'string',
        description: 'Identified crop or plant species, e.g. Tomato, Potato, Corn, Apple, Grape, Rice'
      },
      status: {
        type: 'string',
        description: 'Plant health condition: "healthy" if no disease or pest found, or "diseased" if visual symptoms or pest damage exist',
        enum: ['healthy', 'diseased']
      },
      disease_name: {
        type: 'string',
        description: 'Specific common or scientific name of disease identified (e.g. Late Blight, Powdery Mildew, Leaf Spot) or "Healthy Leaf" if none'
      },
      pathogen_type: {
        type: 'string',
        description: 'Biological category of causal agent: fungal, bacterial, viral, pest, environmental, or none if healthy',
        enum: ['fungal', 'bacterial', 'viral', 'pest', 'environmental', 'none']
      },
      confidence: {
        type: 'string',
        description: 'Confidence level based on visual clarity and distinctness of symptoms: high, medium, or low',
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
        description: 'List of actionable eco-friendly or organic control treatments (e.g. neem oil spray, copper soap, bio-fungicides, removing infected foliage)',
        items: {
          type: 'string',
          description: 'Organic treatment measure'
        }
      },
      chemical_treatment: {
        type: 'array',
        description: 'List of standard synthetic chemicals or fungicides with active ingredients (e.g. Mancozeb 75% WP, Chlorothalonil)',
        items: {
          type: 'string',
          description: 'Chemical treatment measure'
        }
      },
      prevention_measures: {
        type: 'array',
        description: 'List of cultural practices to prevent spread and recurrence (e.g. drip irrigation, crop rotation, resistant cultivars, adequate spacing)',
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
      'pathogen_type',
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

    const progressCallback = (item) => {
      self.postMessage({
        type: 'progress',
        progress: item
      });
    };

    try {
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback
      });

      model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        device: 'webgpu',
        progress_callback: progressCallback
      });

      structuredProcessor = new StructuredOutputProcessor(processor.tokenizer, {
        type: 'json_schema',
        json_schema: DIAGNOSIS_SCHEMA
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

        // 1. Process image
        const imageBlob = dataUrlToBlob(image);
        const rawImage = await RawImage.fromBlob(imageBlob);

        // 2. Assemble prompt
        const systemPrompt = `You are an expert plant pathologist and agronomist.
Examine this crop leaf carefully.
In the "reasoning" field, deduce step-by-step:
- Leaf surface condition, lesions, spot patterns, halos, veins, or healthy color.
- Distinguish between fungal, bacterial, viral, nutrient deficiency, or healthy tissue.
- Conclude diagnosis and actionable evidence-based treatments.
${userNote ? `Farmer Note: "${userNote}"` : ''}
Respond strictly in JSON matching the schema with reasoning as the first property.`;

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
        const inputs = await processor(rawImage, promptText);

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
          do_sample: false,
          streamer: streamer,
          logits_processor: [structuredProcessor]
        });

        const decoded = processor.tokenizer.batch_decode(output, { skip_special_tokens: true });
        const fullText = (decoded[0] || rawTokens).trim();

        let structuredResult = null;
        try {
          structuredResult = JSON.parse(fullText);
        } catch (parseErr) {
          console.warn('Direct parse failed, trying regex match:', parseErr);
          const match = fullText.match(/\{[\s\S]*\}/);
          if (match) {
            structuredResult = JSON.parse(match[0]);
          } else {
            throw new Error('Failed to parse structured diagnosis JSON.');
          }
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
