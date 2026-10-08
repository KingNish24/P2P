import {
  AutoTokenizer,
  AutoModelForImageTextToText,
  Gemma4ImageProcessor,
  RawImage,
  TextStreamer,
  env
} from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';

// Configure transformers.js for edge browser environment
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

const MODEL_ID = 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX';
const DTYPE = 'q2f16';

let tokenizer = null;
let imageProcessor = null;
let model = null;
let structuredProcessor = null;

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
      enum: ['healthy', 'diseased']
    },
    disease_name: {
      type: 'string',
      description: 'Specific disease name or "Healthy Leaf" if healthy'
    },
    pathogen_type: {
      type: 'string',
      enum: ['fungal', 'bacterial', 'viral', 'pest', 'environmental', 'none']
    },
    confidence: {
      type: 'string',
      enum: ['high', 'medium', 'low']
    },
    key_symptoms: {
      type: 'array',
      items: { type: 'string' }
    },
    organic_treatment: {
      type: 'array',
      items: { type: 'string' }
    },
    chemical_treatment: {
      type: 'array',
      items: { type: 'string' }
    },
    prevention_measures: {
      type: 'array',
      items: { type: 'string' }
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
  if (tokenizer && model && imageProcessor) {
    return { tokenizer, model, imageProcessor };
  }

  self.postMessage({
    type: 'status',
    status: 'loading',
    message: 'Loading Gemma 4 E2B (q2f16 WebGPU)...'
  });

  const progressCallback = (item) => {
    self.postMessage({
      type: 'progress',
      progress: item
    });
  };

  try {
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: progressCallback
    });

    imageProcessor = new Gemma4ImageProcessor({
      max_soft_tokens: 280,
      patch_size: 16,
      pooling_kernel_size: 3,
      resample: 3,
      rescale_factor: 0.00392156862745098,
      do_rescale: true,
      do_resize: true,
      do_convert_rgb: true
    });

    model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
      dtype: DTYPE,
      device: 'webgpu',
      progress_callback: progressCallback
    });

    structuredProcessor = new StructuredOutputProcessor(tokenizer, {
      type: 'json_schema',
      json_schema: DIAGNOSIS_SCHEMA
    });

    self.postMessage({
      type: 'status',
      status: 'ready',
      message: 'Gemma 4 E2B WebGPU loaded and ready!'
    });

    return { tokenizer, model, imageProcessor };
  } catch (err) {
    console.error('Failed to load Gemma 4 model:', err);
    self.postMessage({
      type: 'error',
      message: err.message || 'Error initializing WebGPU model.'
    });
    throw err;
  }
}

self.addEventListener('message', async (event) => {
  const { type, data } = event.data;

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
      if (!model || !tokenizer || !imageProcessor) {
        await loadModel();
      }

      const { image, userNote } = data;

      self.postMessage({
        type: 'status',
        status: 'analyzing',
        message: 'Gemma reasoning over crop symptoms...'
      });

      // 1. Process image
      let imageInputs = null;
      let softTokenCount = 280;
      if (image) {
        const rawImage = await RawImage.fromURL(image);
        imageInputs = await imageProcessor(rawImage);
        if (imageInputs.num_soft_tokens_per_image && imageInputs.num_soft_tokens_per_image[0]) {
          softTokenCount = imageInputs.num_soft_tokens_per_image[0];
        }
      }

      // 2. Assemble Gemma 4 multimodal prompt
      const systemPrompt = `You are an expert plant pathologist and agronomist.
Examine this crop leaf carefully.
In the "reasoning" field, deduce step-by-step:
- Leaf surface condition, lesions, spot patterns, halos, veins, or healthy color.
- Distinguish between fungal, bacterial, viral, nutrient deficiency, or healthy tissue.
- Conclude diagnosis and actionable evidence-based treatments.
${userNote ? `Farmer Note: "${userNote}"` : ''}
Respond strictly in JSON matching the schema with reasoning as the first property.`;

      const promptWithImage = `<|turn>user\n<|boi|>${'<|image|>'.repeat(softTokenCount)}<|eoi|>\n${systemPrompt}<turn|>\n<|turn>model\n`;

      const textInputs = tokenizer(promptWithImage);

      let rawTokens = '';
      const streamer = new TextStreamer(tokenizer, {
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

      const generateInputs = {
        ...textInputs,
        max_new_tokens: 1024,
        do_sample: false,
        streamer: streamer,
        logits_processor: [structuredProcessor]
      };

      if (imageInputs?.pixel_values) {
        generateInputs.pixel_values = imageInputs.pixel_values;
      }

      const output = await model.generate(generateInputs);

      const decoded = tokenizer.batch_decode(output, { skip_special_tokens: true });
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
        message: err.message || 'Error analyzing crop leaf.'
      });
    }
  }
});
