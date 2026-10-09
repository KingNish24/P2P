// src/scripts/lfm-engine/kernels.js
/**
 * WGSL kernels for the LFM2.5 decoder. Everything is stored as f16 and accumulated in f32.
 *
 * Quantised weight layout (ONNX MatMulNBits, bits=4, block_size=32, zero points present):
 *   quant  : u8  [N, K/32, 16]  -> little-endian nibbles: nibble i of u32 word w = element (8w + i) of the row
 *   scales : f16 [N, K/32]
 *   zp     : u8  [N, ceil(K/32 / 2)] -> low nibble = even block, high nibble = odd block
 *   w = (q - zp) * scale
 */

const HEAD = "enable f16;\n";

// Shared dequant snippet used by gemv/gemm/embed (needs: sc, zp, nb, zr bindings in scope)
const ZP_FN = `
fn zp_of(zp_arr_word: u32, byte_idx: u32, blk: u32) -> f32 {
  let zb = (zp_arr_word >> ((byte_idx & 3u) * 8u)) & 255u;
  return f32((zb >> ((blk & 1u) * 4u)) & 15u);
}
`;

/** y[Y_OFF + n] = sum_k x[k] * W[n][k]   (M = 1) */
export const GEMV = `${HEAD}${ZP_FN}
override K: u32;
override N: u32;
override Y_OFF: u32 = 0u;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f16>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f16>;
var<workgroup> xs: array<f16, 4608>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  for (var i = tid; i < K; i += 256u) { xs[i] = x[i]; }
  workgroupBarrier();
  let grp = tid >> 5u;
  let lane = tid & 31u;
  let n = wid.x * 8u + grp;
  let nb = K / 32u;
  let zr = (nb + 1u) / 2u;
  let wpr = K / 8u;
  var acc = 0.0;
  if (n < N) {
    for (var w = lane; w < wpr; w += 32u) {
      let word = wq[n * wpr + w];
      let blk = w >> 2u;
      let s = f32(sc[n * nb + blk]);
      let zi = n * zr + (blk >> 1u);
      let z = zp_of(zp[zi >> 2u], zi, blk);
      var a = 0.0;
      for (var i = 0u; i < 8u; i++) {
        let q = f32((word >> (i * 4u)) & 15u);
        a += f32(xs[w * 8u + i]) * (q - z);
      }
      acc += a * s;
    }
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[tid] += red[tid + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && n < N) { y[Y_OFF + n] = f16(red[tid]); }
}
`;

/** Y[m, Y_OFF + n] = sum_k X[m,k] * W[n][k]  (tiled 64x64, 4x4 outputs per thread, K step 32 = one quant block) */
export const GEMM = `${HEAD}${ZP_FN}
override K: u32;
override N: u32;
override Y_STRIDE: u32;
override Y_OFF: u32 = 0u;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f16>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f16>;
@group(0) @binding(5) var<storage, read> mt: array<u32>;
var<workgroup> xs: array<f32, 2048>; // [k][m]
var<workgroup> ws: array<f32, 2048>; // [k][n]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let M = mt[1];
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
  var cc = array<vec4<f32>, 4>(c0, c1, c2, c3);
  for (var i = 0u; i < 4u; i++) {
    let gm = m0 + ty + 16u * i;
    if (gm < M) {
      for (var j = 0u; j < 4u; j++) {
        let gn = n0 + tx + 16u * j;
        if (gn < N) { y[gm * Y_STRIDE + Y_OFF + gn] = f16(cc[i][j]); }
      }
    }
  }
}
`;
/** h += o (optional); xn = rmsnorm(h) * w. DOUBLE: norm of (h + h) (final norm of the ORT graph). One workgroup per row. */
export const ADD_RMS = `${HEAD}
override N: u32 = 1024u;
override HAS_ADD: bool = true;
override DOUBLE: bool = false;
@group(0) @binding(0) var<storage, read_write> h: array<f16>;
@group(0) @binding(1) var<storage, read> o: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;
@group(0) @binding(3) var<storage, read_write> xn: array<f16>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let base = wid.x * N;
  var ss = 0.0;
  for (var i = tid; i < N; i += 256u) {
    var v = f32(h[base + i]);
    if (HAS_ADD) {
      v = f32(f16(v + f32(o[base + i])));
      h[base + i] = f16(v);
    }
    if (DOUBLE) { v = v * 2.0; }
    ss += v * v;
  }
  red[tid] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { red[tid] += red[tid + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(N) + 1e-5);
  for (var i = tid; i < N; i += 256u) {
    var v = f32(h[base + i]);
    if (DOUBLE) { v = v * 2.0; }
    xn[base + i] = f16(v * inv * f32(w[i]));
  }
}
`;

/** Gated short convolution of the LFM2 conv layer. bcx = [B | C | x], depthwise conv (k=3) over a 2-step state. */
export const CONV_STEP = `${HEAD}
@group(0) @binding(0) var<storage, read> bcx: array<f16>;
@group(0) @binding(1) var<storage, read> cw: array<f16>;
@group(0) @binding(2) var<storage, read_write> st: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
@group(0) @binding(4) var<storage, read> mt: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= 1024u) { return; }
  let M = mt[1];
  var s0 = f32(st[c * 2u]);
  var s1 = f32(st[c * 2u + 1u]);
  let w0 = f32(cw[c * 3u]);
  let w1 = f32(cw[c * 3u + 1u]);
  let w2 = f32(cw[c * 3u + 2u]);
  for (var t = 0u; t < M; t++) {
    let b = f32(bcx[t * 3072u + c]);
    let cc = f32(bcx[t * 3072u + 1024u + c]);
    let xx = f32(bcx[t * 3072u + 2048u + c]);
    let u = f32(f16(b * xx));
    let yv = f32(f16(w0 * s0 + w1 * s1 + w2 * u));
    outp[t * 1024u + c] = f16(cc * yv);
    s0 = s1;
    s1 = u;
  }
  st[c * 2u] = f16(s0);
  st[c * 2u + 1u] = f16(s1);
}
`;

/**
 * q/k RMS norm (per head) + RoPE + KV-cache write. Workgroup (unit, token):
 * unit 0..15 = q heads, 16..23 = k heads, 24..31 = v heads (copy only).
 * qkv row layout [q(1024) | k(512) | v(512)]. KV cache layout [pos][kv_head][64] (token-major, so growth = flat copy).
 */
export const QK_PREP = `${HEAD}
@group(0) @binding(0) var<storage, read> qkv: array<f16>;
@group(0) @binding(1) var<storage, read> nwb: array<f16>;
@group(0) @binding(2) var<storage, read_write> qout: array<f16>;
@group(0) @binding(3) var<storage, read_write> kc: array<f16>;
@group(0) @binding(4) var<storage, read_write> vc: array<f16>;
@group(0) @binding(5) var<storage, read> mt: array<u32>;
@group(0) @binding(6) var<storage, read> rope: array<f16>;
var<workgroup> nv: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let unit = wid.x;
  let m = wid.y;
  let pos = mt[0] + m;
  let rowbase = m * 2048u;
  if (unit >= 24u) {
    let kh = unit - 24u;
    vc[(pos * 8u + kh) * 64u + d] = qkv[rowbase + 1536u + kh * 64u + d];
    return;
  }
  let isq = unit < 16u;
  var off = 1024u + (unit - 16u) * 64u;
  if (isq) { off = unit * 64u; }
  let v = f32(qkv[rowbase + off + d]);
  red[d] = v * v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (d < s) { red[d] += red[d + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / 64.0 + 1e-5);
  var wi = 64u + d;
  if (isq) { wi = d; }
  nv[d] = f32(f16(v * inv * f32(nwb[wi])));
  workgroupBarrier();
  let i = d & 31u;
  let cs = f32(rope[pos * 64u + i]);
  let sn = f32(rope[pos * 64u + 32u + i]);
  var r = 0.0;
  if (d < 32u) { r = nv[d] * cs - nv[d + 32u] * sn; }
  else { r = nv[d] * cs + nv[d - 32u] * sn; }
  if (isq) { qout[m * 1024u + unit * 64u + d] = f16(r); }
  else { kc[(pos * 8u + (unit - 16u)) * 64u + d] = f16(r); }
}
`;

/** Causal GQA attention (16 q heads / 8 kv heads, head_dim 64) with chunked online softmax. Workgroup (head, token). */
export const ATTN = `${HEAD}
@group(0) @binding(0) var<storage, read> q: array<f16>;
@group(0) @binding(1) var<storage, read> kc: array<f16>;
@group(0) @binding(2) var<storage, read> vc: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
@group(0) @binding(4) var<storage, read> mt: array<u32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let h = wid.x;
  let m = wid.y;
  let pos = mt[0] + m;
  let kh = h >> 1u;
  if (tid < 64u) { qs[tid] = f32(q[m * 1024u + h * 64u + tid]) * 0.125; }
  workgroupBarrier();
  var rm = -3.0e38;
  var rl = 0.0;
  var acc = 0.0;
  for (var j0 = 0u; j0 <= pos; j0 += 128u) {
    let j = j0 + tid;
    var s = -3.0e38;
    if (j <= pos) {
      var d = 0.0;
      let kb = (j * 8u + kh) * 64u;
      for (var dd = 0u; dd < 64u; dd++) { d += qs[dd] * f32(kc[kb + dd]); }
      s = d;
    }
    sc[tid] = s;
    workgroupBarrier();
    var cm = -3.0e38;
    for (var jj = 0u; jj < 128u; jj++) { cm = max(cm, sc[jj]); }
    let nm = max(rm, cm);
    let f = exp(rm - nm);
    rl = rl * f;
    acc = acc * f;
    if (tid < 64u) {
      for (var jj = 0u; jj < 128u; jj++) {
        let j2 = j0 + jj;
        if (j2 > pos) { break; }
        let p = exp(sc[jj] - nm);
        rl += p;
        acc += p * f32(vc[(j2 * 8u + kh) * 64u + tid]);
      }
    }
    rm = nm;
    workgroupBarrier();
  }
  if (tid < 64u) { outp[m * 1024u + h * 64u + tid] = f16(acc / rl); }
}
`;

/** a = silu(gate) * up, gu row = [gate(4608) | up(4608)] */
export const SILU_MUL = `${HEAD}
@group(0) @binding(0) var<storage, read> gu: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<storage, read> mt: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  let total = mt[1] * 4608u;
  if (idx >= total) { return; }
  let row = idx / 4608u;
  let col = idx - row * 4608u;
  let gt = f32(gu[row * 9216u + col]);
  let up = f32(gu[row * 9216u + 4608u + col]);
  let sg = f32(f16(1.0 / (1.0 + exp(-gt))));
  let a = f32(f16(gt * sg));
  outp[idx] = f16(a * up);
}
`;

/** h[0..1023] = dequant(embed_tokens[tok]) */
export const EMBED = `${HEAD}${ZP_FN}
@group(0) @binding(0) var<storage, read> tok: array<u32>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f16>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> h: array<f16>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) w: u32) {
  let row = tok[0];
  let word = wq[row * 128u + w];
  let blk = w >> 2u;
  let s = f32(sc[row * 32u + blk]);
  let zi = row * 16u + (blk >> 1u);
  let z = zp_of(zp[zi >> 2u], zi, blk);
  for (var i = 0u; i < 8u; i++) {
    h[w * 8u + i] = f16((f32((word >> (i * 4u)) & 15u) - z) * s);
  }
}
`;

/** Greedy argmax over the logits. Also stores the token in outTok[step] and advances position + step in mt. */
export const ARGMAX = `${HEAD}
override V: u32 = 65536u;
@group(0) @binding(0) var<storage, read> logits: array<f16>;
@group(0) @binding(1) var<storage, read_write> tok: array<u32>;
@group(0) @binding(2) var<storage, read_write> mt: array<u32>;
@group(0) @binding(3) var<storage, read_write> outTok: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  var best = -3.0e38;
  var bidx = 0u;
  for (var i = t; i < V; i += 256u) {
    let v = f32(logits[i]);
    if (v > best) { best = v; bidx = i; }
  }
  bv[t] = best;
  bi[t] = bidx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = bv[t + s];
      let oi = bi[t + s];
      if (ov > bv[t] || (ov == bv[t] && oi < bi[t])) { bv[t] = ov; bi[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    tok[0] = bi[0];
    outTok[mt[2]] = bi[0];
    mt[2] = mt[2] + 1u;
    mt[0] = mt[0] + 1u;
  }
}
`;
