// src/scripts/lfm-engine/vision.js
/**
 * Full WebGPU Vision Encoder for LFM2.5-VL-450M (SigLIP2 architecture).
 *
 * Pipeline:
 *  1. Patch Embedding (768 -> 768 4-bit MatMulNBits + FP16 bias)
 *  2. Precomputed 2D Positional Embeddings for [32, 32] patches added to embeddings
 *  3. 12 Transformer Encoder Layers:
 *     - Pre-LN (D=768, eps=1e-6)
 *     - Multi-Head Attention (12 heads, head_dim=64, scale=0.125) with chunked online softmax
 *     - Out-Projection (768 -> 768 4-bit MatMulNBits + bias) + Residual Add
 *     - Pre-LN (D=768, eps=1e-6)
 *     - MLP FC1 (768 -> 3072 4-bit MatMulNBits + bias)
 *     - FastGELU activation (tanh)
 *     - MLP FC2 (3072 -> 768 4-bit MatMulNBits + bias) + Residual Add
 *  4. Final LayerNorm (D=768, eps=1e-6)
 *  5. Space-to-Depth 2x downsampler: [32, 32, 768] -> [16, 16, 3072] (256 tokens)
 *  6. 2-layer Multimodal Projector:
 *     - Linear 1 (3072 -> 2048 4-bit MatMulNBits + bias)
 *     - GELU activation (erf-based)
 *     - Linear 2 (2048 -> 1024 4-bit MatMulNBits + bias)
 *
 * Output: GPUBuffer containing [256, 1024] FP16 visual token features.
 */

import { loadWeights } from "./weights.js";
import { toHalfBits } from "./engine.js";
import * as VK from "./vision-kernels.js";

const NUM_PATCHES_H = 32;
const NUM_PATCHES_W = 32;
const NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W; // 1024
const HIDDEN_DIM = 768;
const NUM_HEADS = 12;
const HEAD_DIM = 64;
const MLP_DIM = 3072;
const NUM_LAYERS = 12;

const DOWN_TOKENS = 256; // 16 * 16
const DOWN_DIM = 3072;   // 768 * 2 * 2
const PROJ_HIDDEN = 2048;
const PROJ_OUT = 1024;

/**
 * Precompute the 2D positional embeddings table for [32, 32] patches from the learned
 * [16, 16, 768] base grid using the model's exact bilinear/area interpolation formula.
 * @param {Float32Array} baseFloats - [16, 16, 768] base grid
 * @param {number} H - height in patches (32)
 * @param {number} W - width in patches (32)
 * @returns {Uint16Array} FP16 half-bits tensor of shape [H * W, 768]
 */
export function computePosEmbed(baseFloats, H = NUM_PATCHES_H, W = NUM_PATCHES_W) {
  const y_scale = 16.0 / H;
  const x_scale = 16.0 / W;
  const y_radius = Math.max(y_scale, 1.0);
  const x_radius = Math.max(x_scale, 1.0);
  const kh = Math.ceil(y_radius * 2.0 + 2.0); // 4
  const kw = Math.ceil(x_radius * 2.0 + 2.0); // 4

  const numPatches = H * W;
  const out = new Uint16Array(numPatches * HIDDEN_DIM);
  const patchVec = new Float32Array(HIDDEN_DIM);

  for (let p = 0; p < numPatches; p++) {
    const y = Math.floor(p / W);
    const x = p % W;
    const y_center = (y + 0.5) * y_scale - 0.5;
    const x_center = (x + 0.5) * x_scale - 0.5;
    const y_min = Math.floor(y_center - y_radius);
    const x_min = Math.floor(x_center - x_radius);

    let totalWeight = 0;
    patchVec.fill(0);

    for (let ky = 0; ky < kh; ky++) {
      const ys = y_min + ky;
      const dist_y = Math.abs(ys - y_center);
      const wy = Math.max(0.0, 1.0 - dist_y / y_radius);
      const valid_y = ys >= 0 && ys < 16;
      const cy = Math.max(0, Math.min(15, ys));

      for (let kx = 0; kx < kw; kx++) {
        const xs = x_min + kx;
        const dist_x = Math.abs(xs - x_center);
        const wx = Math.max(0.0, 1.0 - dist_x / x_radius);
        const valid_x = xs >= 0 && xs < 16;
        const cx = Math.max(0, Math.min(15, xs));

        let weight = wy * wx;
        if (!valid_y || !valid_x) weight = 0.0;
        if (weight > 0) {
          totalWeight += weight;
          const baseOffset = (cy * 16 + cx) * HIDDEN_DIM;
          for (let d = 0; d < HIDDEN_DIM; d++) {
            patchVec[d] += baseFloats[baseOffset + d] * weight;
          }
        }
      }
    }

    const denom = Math.max(totalWeight, 1e-7);
    const outOffset = p * HIDDEN_DIM;
    for (let d = 0; d < HIDDEN_DIM; d++) {
      out[outOffset + d] = toHalfBits(patchVec[d] / denom);
    }
  }
  return out;
}

/**
 * Creates the LFM2.5 SigLIP2 Vision Engine.
 * @param {Object} options
 * @param {GPUDevice} options.device
 * @param {Uint8Array} options.modelBytes - ONNX bytes for vision_encoder_q4f16.onnx
 * @param {Object} options.source - external data reader with .read(offset, length)
 * @param {Function} [options.onProgress]
 */
export async function createVisionEngine({ device, modelBytes, source, onProgress }) {
  if (!device.features.has("shader-f16")) {
    throw new Error("Vision engine requires WebGPU 'shader-f16' feature");
  }

  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;

  // 1. Load weights
  const weights = await loadWeights(device, modelBytes, source, {
    skip: ["/model/constants/FLOAT16/-65504.0"], // unused mask constant
    onProgress: (p) => onProgress && onProgress(0.9 * p)
  });

  const T = (name) => {
    const t = weights.tensors.get(name);
    if (!t) throw new Error("Missing vision initializer: " + name);
    return t.buffer;
  };

  const proj = (baseQuantName, biasName, K_, N_, M_ = 1024) => {
    const base = baseQuantName.replace(/_weight_quant$/, "");
    return {
      q: T(base + "_weight_quant"),
      s: T(base + "_weight_scales"),
      z: T(base + "_weight_zp"),
      b: T(biasName),
      K: K_,
      N: N_,
      M: M_
    };
  };

  // 2. Precompute 2D positional embeddings for [32, 32] patches
  const baseInit = weights.graph.initializers.get("/model/embeddings/pos_embed/base_weight");
  let baseFloats;
  if (baseInit.external) {
    const raw = await source.read(Number(baseInit.external.offset || 0), Number(baseInit.external.length));
    baseFloats = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  } else if (baseInit.raw) {
    baseFloats = new Float32Array(baseInit.raw.buffer, baseInit.raw.byteOffset, baseInit.raw.byteLength / 4);
  } else if (baseInit.floats) {
    baseFloats = baseInit.floats;
  } else {
    throw new Error("Unable to read base_weight initializer for position embeddings");
  }

  const posEmbedHalfBits = computePosEmbed(baseFloats, NUM_PATCHES_H, NUM_PATCHES_W);

  // 3. Allocate activation and workspace buffers
  const allBuffers = [];
  const mk = (size, usage = S | CD | CS, label) => {
    const buf = device.createBuffer({ size: Math.max(4, (size + 3) & ~3), usage, label });
    allBuffers.push(buf);
    return buf;
  };

  const inputF16 = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "input_f16");
  const hBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "h_buf");
  const normBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "norm_buf");
  const qBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "q_buf");
  const kBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "k_buf");
  const vBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "v_buf");
  const attnBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "attn_buf");
  const attnOutBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "attn_out_buf");
  const fc1Buf = mk(NUM_PATCHES * MLP_DIM * 2, undefined, "fc1_buf");
  const fc2Buf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "fc2_buf");

  const posEmbedBuf = mk(NUM_PATCHES * HIDDEN_DIM * 2, undefined, "pos_embed_buf");
  device.queue.writeBuffer(posEmbedBuf, 0, posEmbedHalfBits);

  const s2dBuf = mk(DOWN_TOKENS * DOWN_DIM * 2, undefined, "s2d_buf");
  const proj1Buf = mk(DOWN_TOKENS * PROJ_HIDDEN * 2, undefined, "proj1_buf");
  const outBuf = mk(DOWN_TOKENS * PROJ_OUT * 2, undefined, "image_features_f16");

  // Optional F32 conversion pipeline if an FP32 GPU buffer is fed
  let castPipeline = null;
  let castBindGroup = null;
  let cachedF32Input = null;

  // 4. Compile pipelines & bind groups
  const modules = new Map();
  const pipes = new Map();
  const pipe = (code, constants = {}) => {
    const key = code.length + ":" + code.slice(0, 80) + JSON.stringify(constants);
    let p = pipes.get(key);
    if (!p) {
      let m = modules.get(code);
      if (!m) {
        m = device.createShaderModule({ code });
        modules.set(code, m);
      }
      p = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main", constants } });
      pipes.set(key, p);
    }
    return p;
  };

  const bind = (p, bufs) =>
    device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } }))
    });

  const ops = [];
  const addOp = (pipeline, bufs, grid) => {
    const bindGroup = bind(pipeline, bufs);
    ops.push({ pipeline, bindGroup, grid });
  };

  // Step A: Patch Embedding (768 -> 768)
  const patchPj = proj(
    "model_embeddings_patch_embedding_MatMul_weight_quant",
    "model.embeddings.patch_embedding.Add.bias",
    768,
    768,
    1024
  );
  addOp(
    pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 768, M: 1024 }),
    [inputF16, patchPj.q, patchPj.s, patchPj.z, patchPj.b, hBuf],
    [12, 16, 1] // 768/64, 1024/64
  );

  // Step B: Add 2D Positional Embeddings
  addOp(
    pipe(VK.ADD_RESIDUAL, { TOTAL: NUM_PATCHES * HIDDEN_DIM }),
    [hBuf, posEmbedBuf],
    [Math.ceil((NUM_PATCHES * HIDDEN_DIM) / 256), 1, 1]
  );

  // Step C: 12 Transformer Layers
  for (let l = 0; l < NUM_LAYERS; l++) {
    // 1. Pre-LN 1
    const ln1Scale = T(`model.layers.${l}.layer_norm1_layernorm.weight`);
    const ln1Bias = T(`model.layers.${l}.layer_norm1_layernorm.bias`);
    addOp(
      pipe(VK.LAYER_NORM, { D: HIDDEN_DIM }),
      [hBuf, ln1Scale, ln1Bias, normBuf],
      [NUM_PATCHES, 1, 1]
    );

    // 2. Q, K, V Projections
    const qPj = proj(`model_layers_${l}_attn_q_proj_MatMul_weight_quant`, `model.layers.${l}.attn.q_proj.Add.bias`, 768, 768, 1024);
    const kPj = proj(`model_layers_${l}_attn_k_proj_MatMul_weight_quant`, `model.layers.${l}.attn.k_proj.Add.bias`, 768, 768, 1024);
    const vPj = proj(`model_layers_${l}_attn_v_proj_MatMul_weight_quant`, `model.layers.${l}.attn.v_proj.Add.bias`, 768, 768, 1024);

    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 768, M: 1024 }), [normBuf, qPj.q, qPj.s, qPj.z, qPj.b, qBuf], [12, 16, 1]);
    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 768, M: 1024 }), [normBuf, kPj.q, kPj.s, kPj.z, kPj.b, kBuf], [12, 16, 1]);
    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 768, M: 1024 }), [normBuf, vPj.q, vPj.s, vPj.z, vPj.b, vBuf], [12, 16, 1]);

    // 3. Multi-Head Attention
    addOp(
      pipe(VK.VISION_ATTN, { S: NUM_PATCHES, NUM_HEADS, HEAD_DIM, SCALE: 0.125 }),
      [qBuf, kBuf, vBuf, attnBuf],
      [NUM_HEADS, NUM_PATCHES, 1] // 12, 1024
    );

    // 4. Out-Projection
    const outPj = proj(`model_layers_${l}_attn_out_proj_MatMul_weight_quant`, `model.layers.${l}.attn.out_proj.Add.bias`, 768, 768, 1024);
    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 768, M: 1024 }), [attnBuf, outPj.q, outPj.s, outPj.z, outPj.b, attnOutBuf], [12, 16, 1]);

    // 5. Residual connection (h = h + attnOut)
    addOp(
      pipe(VK.ADD_RESIDUAL, { TOTAL: NUM_PATCHES * HIDDEN_DIM }),
      [hBuf, attnOutBuf],
      [Math.ceil((NUM_PATCHES * HIDDEN_DIM) / 256), 1, 1]
    );

    // 6. Pre-LN 2
    const ln2Scale = T(`model.layers.${l}.layer_norm2_layernorm.weight`);
    const ln2Bias = T(`model.layers.${l}.layer_norm2_layernorm.bias`);
    addOp(
      pipe(VK.LAYER_NORM, { D: HIDDEN_DIM }),
      [hBuf, ln2Scale, ln2Bias, normBuf],
      [NUM_PATCHES, 1, 1]
    );

    // 7. MLP FC1 (768 -> 3072)
    const fc1Pj = proj(`model_layers_${l}_mlp_fc1_MatMul_weight_quant`, `model.layers.${l}.mlp.fc1.Add.bias`, 768, 3072, 1024);
    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 768, N: 3072, M: 1024 }), [normBuf, fc1Pj.q, fc1Pj.s, fc1Pj.z, fc1Pj.b, fc1Buf], [48, 16, 1]);

    // 8. FastGELU activation
    addOp(
      pipe(VK.FAST_GELU, { TOTAL: NUM_PATCHES * MLP_DIM }),
      [fc1Buf],
      [Math.ceil((NUM_PATCHES * MLP_DIM) / 256), 1, 1]
    );

    // 9. MLP FC2 (3072 -> 768)
    const fc2Pj = proj(`model_layers_${l}_mlp_fc2_MatMul_weight_quant`, `model.layers.${l}.mlp.fc2.Add.bias`, 3072, 768, 1024);
    addOp(pipe(VK.MATMUL_ADD_BIAS, { K: 3072, N: 768, M: 1024 }), [fc1Buf, fc2Pj.q, fc2Pj.s, fc2Pj.z, fc2Pj.b, fc2Buf], [12, 16, 1]);

    // 10. Residual connection (h = h + fc2Out)
    addOp(
      pipe(VK.ADD_RESIDUAL, { TOTAL: NUM_PATCHES * HIDDEN_DIM }),
      [hBuf, fc2Buf],
      [Math.ceil((NUM_PATCHES * HIDDEN_DIM) / 256), 1, 1]
    );
  }

  // Step D: Final LayerNorm (768)
  const fnScale = T("model.layers.12.final_norm_layernorm.weight");
  const fnBias = T("model.layers.12.final_norm_layernorm.bias");
  addOp(
    pipe(VK.LAYER_NORM, { D: HIDDEN_DIM }),
    [hBuf, fnScale, fnBias, normBuf],
    [NUM_PATCHES, 1, 1]
  );

  // Step E: SpaceToDepth 2x: [32, 32, 768] -> [16, 16, 3072]
  addOp(
    pipe(VK.SPACE_TO_DEPTH_2X),
    [normBuf, s2dBuf],
    [Math.ceil(196608 / 256), 1, 1] // 196608 vec4s = 768 workgroups
  );

  // Step F: Projector Linear 1 (3072 -> 2048, M = 256)
  const proj1Pj = proj(
    "model_multimodal_projector_linear_1_MatMul_weight_quant",
    "model.multimodal_projector.linear_1.Add.bias",
    3072,
    2048,
    256
  );
  addOp(
    pipe(VK.MATMUL_ADD_BIAS, { K: 3072, N: 2048, M: 256 }),
    [s2dBuf, proj1Pj.q, proj1Pj.s, proj1Pj.z, proj1Pj.b, proj1Buf],
    [32, 4, 1] // 2048/64, 256/64
  );

  // Step G: Projector GELU (Exact erf-based)
  addOp(
    pipe(VK.GELU, { TOTAL: DOWN_TOKENS * PROJ_HIDDEN }),
    [proj1Buf],
    [Math.ceil((DOWN_TOKENS * PROJ_HIDDEN) / 256), 1, 1]
  );

  // Step H: Projector Linear 2 (2048 -> 1024, M = 256)
  const proj2Pj = proj(
    "model_multimodal_projector_linear_2_MatMul_weight_quant",
    "model.multimodal_projector.linear_2.Add.bias",
    2048,
    1024,
    256
  );
  addOp(
    pipe(VK.MATMUL_ADD_BIAS, { K: 2048, N: 1024, M: 256 }),
    [proj1Buf, proj2Pj.q, proj2Pj.s, proj2Pj.z, proj2Pj.b, outBuf],
    [16, 4, 1] // 1024/64, 256/64
  );

  if (onProgress) onProgress(1);

  /**
   * Encodes a 512x512 image into 256 visual token embeddings.
   * @param {GPUBuffer | Float32Array | Uint16Array} pixelValuesBuffer
   *  - GPUBuffer: [1024, 768] in f16 (1.5 MB) or f32 (3 MB)
   *  - Float32Array: [1024 * 768]
   *  - Uint16Array: [1024 * 768] f16 bits
   * @returns {GPUBuffer} output GPU buffer with shape [256, 1024] in f16 format
   */
  function encode(pixelValuesBuffer) {
    const enc = device.createCommandEncoder({ label: "vision_encoder" });

    // Handle input feeding
    if (pixelValuesBuffer instanceof GPUBuffer) {
      if (pixelValuesBuffer.size >= NUM_PATCHES * HIDDEN_DIM * 4) {
        // FP32 GPU buffer -> cast to FP16
        if (!castPipeline) {
          castPipeline = pipe(VK.CAST_F32_TO_F16, { TOTAL: NUM_PATCHES * HIDDEN_DIM });
        }
        if (cachedF32Input !== pixelValuesBuffer) {
          cachedF32Input = pixelValuesBuffer;
          castBindGroup = bind(castPipeline, [pixelValuesBuffer, inputF16]);
        }
        const castPass = enc.beginComputePass({ label: "cast_f32_to_f16" });
        castPass.setPipeline(castPipeline);
        castPass.setBindGroup(0, castBindGroup);
        castPass.dispatchWorkgroups(Math.ceil((NUM_PATCHES * HIDDEN_DIM) / 256));
        castPass.end();
      } else if (pixelValuesBuffer !== inputF16) {
        // FP16 GPU buffer -> copy to inputF16
        enc.copyBufferToBuffer(pixelValuesBuffer, 0, inputF16, 0, NUM_PATCHES * HIDDEN_DIM * 2);
      }
    } else if (pixelValuesBuffer instanceof Float32Array) {
      // Float32Array on CPU -> convert to f16 and upload
      const hBits = new Uint16Array(pixelValuesBuffer.length);
      for (let i = 0; i < pixelValuesBuffer.length; i++) {
        hBits[i] = toHalfBits(pixelValuesBuffer[i]);
      }
      device.queue.writeBuffer(inputF16, 0, hBits);
    } else if (pixelValuesBuffer instanceof Uint16Array) {
      // Already f16 half-bits on CPU -> upload directly
      device.queue.writeBuffer(inputF16, 0, pixelValuesBuffer);
    } else {
      throw new Error("Unsupported pixelValuesBuffer input type: " + typeof pixelValuesBuffer);
    }

    // Execute vision encoder graph
    const pass = enc.beginComputePass({ label: "vision_pass" });
    for (const op of ops) {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      pass.dispatchWorkgroups(op.grid[0], op.grid[1] || 1, op.grid[2] || 1);
    }
    pass.end();

    device.queue.submit([enc.finish()]);
    return outBuf;
  }

  /** Release all GPU buffers and pipeline caches. */
  function destroy() {
    for (const b of allBuffers) {
      try { b.destroy(); } catch (_) {}
    }
    for (const t of weights.tensors.values()) {
      try { t.buffer.destroy(); } catch (_) {}
    }
    pipes.clear();
    modules.clear();
  }

  return {
    encode,
    destroy,
    outBuf,
    weights,
    stats: {
      numLayers: NUM_LAYERS,
      patches: NUM_PATCHES,
      outputTokens: DOWN_TOKENS,
      outputDim: PROJ_OUT,
      weightTensors: weights.tensors.size
    }
  };
}
