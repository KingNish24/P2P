// src/scripts/lfm-engine/vision-kernels.js
/**
 * WGSL compute shaders for the LFM2.5 SigLIP2 Vision Encoder.
 * Everything is stored as f16 and accumulated in f32.
 *
 * Quantised weight layout (ONNX MatMulNBits, bits=4, block_size=32, zero points present):
 *   quant  : u8  [N, K/32, 16]  -> little-endian nibbles
 *   scales : f16 [N, K/32]
 *   zp     : u8  [N, ceil(K/32 / 2)] -> low nibble = even block, high nibble = odd block
 *   w = (q - zp) * scale
 */

const HEAD = "enable f16;\n";

// Shared dequant snippet used by MatMulNBits
const ZP_FN = `
fn zp_of(zp_arr_word: u32, byte_idx: u32, blk: u32) -> f32 {
  let zb = (zp_arr_word >> ((byte_idx & 3u) * 8u)) & 255u;
  return f32((zb >> ((blk & 1u) * 4u)) & 15u);
}
`;

/**
 * 4-bit quant MatMulNBits with bias add for prefill/batch (M = 1024 or M = 256).
 * Y[m, Y_OFF + n] = sum_k X[m, k] * W[n, k] + bias[n]
 * Tiled 64x64, 4x4 outputs per thread, K step 32 = one quant block.
 */
export const MATMUL_ADD_BIAS = `${HEAD}${ZP_FN}
override K: u32;
override N: u32;
override M: u32 = 1024u;
override Y_STRIDE: u32 = 0u;
override Y_OFF: u32 = 0u;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f16>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<f16>;
@group(0) @binding(5) var<storage, read_write> y: array<f16>;
var<workgroup> xs: array<f32, 2048>; // [k][m]
var<workgroup> ws: array<f32, 2048>; // [k][n]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let n0 = wid.x * 64u;
  let m0 = wid.y * 64u;
  let tx = tid & 15u;
  let ty = tid >> 4u;
  let nb = K / 32u;
  let zr = (nb + 1u) / 2u;
  let wpr = K / 8u;
  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);
  for (var kb = 0u; kb < nb; kb++) {
    for (var e = tid; e < 2048u; e += 256u) {
      let mm = e >> 5u;
      let kk = e & 31u;
      let gm = m0 + mm;
      var v = 0.0;
      if (gm < M) { v = f32(x[gm * K + kb * 32u + kk]); }
      xs[kk * 64u + mm] = v;
    }
    {
      let nn = tid >> 2u;
      let wi = tid & 3u;
      let gn = n0 + nn;
      if (gn < N) {
        let word = wq[gn * wpr + kb * 4u + wi];
        let s = f32(sc[gn * nb + kb]);
        let zi = gn * zr + (kb >> 1u);
        let z = zp_of(zp[zi >> 2u], zi, kb);
        for (var i = 0u; i < 8u; i++) {
          ws[(wi * 8u + i) * 64u + nn] = (f32((word >> (i * 4u)) & 15u) - z) * s;
        }
      } else {
        for (var i = 0u; i < 8u; i++) { ws[(wi * 8u + i) * 64u + nn] = 0.0; }
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let r = k * 64u;
      let b = vec4<f32>(ws[r + tx], ws[r + tx + 16u], ws[r + tx + 32u], ws[r + tx + 48u]);
      c0 += xs[r + ty] * b;
      c1 += xs[r + ty + 16u] * b;
      c2 += xs[r + ty + 32u] * b;
      c3 += xs[r + ty + 48u] * b;
    }
    workgroupBarrier();
  }
  let stride = select(Y_STRIDE, N, Y_STRIDE == 0u);
  var cc = array<vec4<f32>, 4>(c0, c1, c2, c3);
  for (var i = 0u; i < 4u; i++) {
    let gm = m0 + ty + 16u * i;
    if (gm < M) {
      for (var j = 0u; j < 4u; j++) {
        let gn = n0 + tx + 16u * j;
        if (gn < N) {
          let b = f32(bias[gn]);
          y[gm * stride + Y_OFF + gn] = f16(cc[i][j] + b);
        }
      }
    }
  }
}
`;

/**
 * Standard FP16 LayerNorm over dimension D=768 (mean, variance, scale, bias with eps=1e-6).
 * One workgroup of 256 threads per row (wid.x = token row).
 */
export const LAYER_NORM = `${HEAD}
override D: u32 = 768u;
override EPS: f32 = 1e-6;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> scale: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let row = wid.x;
  let base = row * D;

  // 1. Mean
  var s = 0.0;
  for (var i = tid; i < D; i += 256u) {
    s += f32(x[base + i]);
  }
  red[tid] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { red[tid] += red[tid + stride]; }
    workgroupBarrier();
  }
  let mean = red[0] / f32(D);
  workgroupBarrier();

  // 2. Variance
  var ss = 0.0;
  for (var i = tid; i < D; i += 256u) {
    let diff = f32(x[base + i]) - mean;
    ss += diff * diff;
  }
  red[tid] = ss;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { red[tid] += red[tid + stride]; }
    workgroupBarrier();
  }
  let inv_std = inverseSqrt(red[0] / f32(D) + EPS);
  workgroupBarrier();

  // 3. Scale and Bias
  for (var i = tid; i < D; i += 256u) {
    let norm = (f32(x[base + i]) - mean) * inv_std;
    y[base + i] = f16(norm * f32(scale[i]) + f32(bias[i]));
  }
}
`;

/** Elementwise f16 tensor addition (x = x + res). */
export const ADD_RESIDUAL = `${HEAD}
override TOTAL: u32 = 786432u;
@group(0) @binding(0) var<storage, read_write> x: array<f16>;
@group(0) @binding(1) var<storage, read> res: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i < TOTAL) {
    x[i] = f16(f32(x[i]) + f32(res[i]));
  }
}
`;

/**
 * Bidirectional Multi-Head Self-Attention with chunked online softmax.
 * S=1024 tokens, 12 heads, head_dim=64, scale=0.125.
 * Workgroup: (head wid.x, token wid.y).
 */
export const VISION_ATTN = `${HEAD}
override S: u32 = 1024u;
override NUM_HEADS: u32 = 12u;
override HEAD_DIM: u32 = 64u;
override SCALE: f32 = 0.125;
@group(0) @binding(0) var<storage, read> q: array<f16>;
@group(0) @binding(1) var<storage, read> k: array<f16>;
@group(0) @binding(2) var<storage, read> v: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let h = wid.x;
  let m = wid.y;
  let d = tid;

  let q_base = m * 768u + h * 64u;
  qs[d] = f32(q[q_base + d]) * SCALE;
  workgroupBarrier();

  var running_max = -1e30;
  var running_sum = 0.0;
  var acc = 0.0;

  for (var j0 = 0u; j0 < S; j0 += 64u) {
    let j = j0 + d;
    var dot = -1e30;
    if (j < S) {
      let k_base = j * 768u + h * 64u;
      var sum_d = 0.0;
      for (var dd = 0u; dd < 64u; dd++) {
        sum_d += qs[dd] * f32(k[k_base + dd]);
      }
      dot = sum_d;
    }
    sc[d] = dot;
    workgroupBarrier();

    var chunk_max = sc[0];
    for (var i = 1u; i < 64u; i++) {
      chunk_max = max(chunk_max, sc[i]);
    }

    if (j0 == 0u) {
      running_max = chunk_max;
      for (var i = 0u; i < 64u; i++) {
        let j2 = j0 + i;
        if (j2 < S) {
          let p = exp(sc[i] - chunk_max);
          running_sum += p;
          let v_base = j2 * 768u + h * 64u;
          acc += p * f32(v[v_base + d]);
        }
      }
    } else {
      let new_max = max(running_max, chunk_max);
      let alpha = exp(running_max - new_max);
      running_sum = running_sum * alpha;
      acc = acc * alpha;

      for (var i = 0u; i < 64u; i++) {
        let j2 = j0 + i;
        if (j2 < S) {
          let p = exp(sc[i] - new_max);
          running_sum += p;
          let v_base = j2 * 768u + h * 64u;
          acc += p * f32(v[v_base + d]);
        }
      }
      running_max = new_max;
    }
    workgroupBarrier();
  }

  outp[q_base + d] = f16(acc / running_sum);
}
`;

/**
 * FastGELU: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
 * Clamps the tanh argument to [-14.0, 14.0] to prevent exp overflow inside GPU tanh.
 * Applied in-place on fc1 intermediate activations [1024, 3072].
 */
export const FAST_GELU = `${HEAD}
override TOTAL: u32 = 3145728u; // 1024 * 3072
@group(0) @binding(0) var<storage, read_write> x: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= TOTAL) { return; }
  let v = f32(x[i]);
  let u = 0.7978845608 * (v + 0.044715 * v * v * v);
  let res = 0.5 * v * (1.0 + tanh(clamp(u, -14.0, 14.0)));
  x[i] = f16(res);
}
`;

/**
 * SpaceToDepth (blocksize=2): [32, 32, 768] -> [16, 16, 3072] (256 tokens of 3072 dims).
 * Vectorized over 4 f16s (vec4<f16>).
 * Total vec4s = 256 * 3072 / 4 = 196,608.
 */
export const SPACE_TO_DEPTH_2X = `${HEAD}
@group(0) @binding(0) var<storage, read> in_buf: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<vec4<f16>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let vec_idx = g.x;
  if (vec_idx >= 196608u) { return; }
  let out_tok = vec_idx / 768u; // 0..255
  let vout = vec_idx % 768u;    // 0..767
  let blk = vout / 192u;        // 0..3
  let vc = vout % 192u;         // 0..191

  let by = blk >> 1u;           // 0..1
  let bx = blk & 1u;            // 0..1

  let h2 = out_tok / 16u;       // 0..15
  let w2 = out_tok % 16u;       // 0..15

  let h = h2 * 2u + by;         // 0..31
  let w = w2 * 2u + bx;         // 0..31

  let in_vec_idx = (h * 32u + w) * 192u + vc;
  out_buf[vec_idx] = in_buf[in_vec_idx];
}
`;

/**
 * Standard / Exact GELU for multimodal projector: 0.5 * x * (1.0 + erf(x / sqrt(2)))
 * Applied in-place on projector linear 1 intermediate activations [256, 2048].
 */
export const GELU = `${HEAD}
override TOTAL: u32 = 524288u; // 256 * 2048

fn erf_approx(x: f32) -> f32 {
  let a1 = 0.254829592;
  let a2 = -0.284496736;
  let a3 = 1.421413741;
  let a4 = -1.453152027;
  let a5 = 1.061405429;
  let p = 0.3275911;
  let sign = select(-1.0, 1.0, x >= 0.0);
  let abs_x = abs(x);
  let t = 1.0 / (1.0 + p * abs_x);
  let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_x * abs_x);
  return sign * y;
}

@group(0) @binding(0) var<storage, read_write> x: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= TOTAL) { return; }
  let v = f32(x[i]);
  let res = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865475));
  x[i] = f16(res);
}
`;

/** Optional utility shader: convert FP32 pixel values to FP16 */
export const CAST_F32_TO_F16 = `${HEAD}
override TOTAL: u32 = 786432u;
@group(0) @binding(0) var<storage, read> in_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_f16: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i < TOTAL) {
    out_f16[i] = f16(in_f32[i]);
  }
}
`;
