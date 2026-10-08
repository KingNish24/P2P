import { pipeline, TextStreamer, env } from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';

// Configure transformers.js for edge browser environment
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

const MODEL_ID = 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX';
const DTYPE = 'q2f16';

let generator = null;
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
  if (generator) return generator;

  self.postMessage({
    type: 'status',
    status: 'loading',
    message: 'Initializing Gemma 4 E2B (q2f16 WebGPU)...'
  });

  try {
    generator = await pipeline('image-text-to-text', MODEL_ID, {
      dtype: DTYPE,
      device: 'webgpu',
      progress_callback: (item) => {
        self.postMessage({
          type: 'progress',
          progress: item
        });
      }
    });

    structuredProcessor = new StructuredOutputProcessor(generator.tokenizer, {
      type: 'json_schema',
      json_schema: DIAGNOSIS_SCHEMA
    });

    self.postMessage({
      type: 'status',
      status: 'ready',
      message: 'Gemma 4 E2B WebGPU loaded and ready!'
    });

    return generator;
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
      // already reported to UI
    }
    return;
  }

  if (type === 'analyze') {
    try {
      if (!generator) {
        await loadModel();
      }

      const { image, userNote } = data;

      self.postMessage({
        type: 'status',
        status: 'analyzing',
        message: 'Gemma reasoning over crop symptoms...'
      });

      const prompt = `You are an expert plant pathologist and agronomist. 
Inspect the provided crop leaf image.
First, fill the "reasoning" field with step-by-step visual deduction:
1. Examine leaf surface, margins, and veins.
2. Note spot patterns, discoloration, halos, or healthy green pigment.
3. Determine if tissue is healthy or infected by fungal, bacterial, viral, or pest pathogen.
4. Conclude with exact diagnosis and actionable, safe treatments.
${userNote ? `Farmer Note: "${userNote}"` : ''}
Respond ONLY in valid JSON matching schema.`;

      const messages = [
        {
          role: 'user',
          content: [
            { type: 'image', image: image },
            { type: 'text', text: prompt }
          ]
        }
      ];

      let rawTokens = '';
      const streamer = new TextStreamer(generator.tokenizer, {
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

      const output = await generator(messages, {
        max_new_tokens: 1024,
        do_sample: false,
        streamer: streamer,
        logits_processor: [structuredProcessor]
      });

      const content = output[0]?.generated_text?.at(-1)?.content || rawTokens;
      let structuredResult = null;

      try {
        structuredResult = JSON.parse(content);
      } catch (parseErr) {
        console.warn('Direct parse failed, extracting JSON block:', parseErr);
        const match = content.match(/\{[\s\S]*\}/);
        if (match) {
          structuredResult = JSON.parse(match[0]);
        } else {
          throw new Error('Could not parse structured diagnosis JSON.');
        }
      }

      self.postMessage({
        type: 'result',
        data: structuredResult,
        rawText: content
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
