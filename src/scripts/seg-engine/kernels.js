// src/scripts/seg-engine/kernels.js
/**
 * WGSL kernels for the custom u2netp engine.
 * Activations live in storage buffers as [C][H][W]; every kernel takes element offsets,
 * so channel slices of a concat buffer can be read/written in place.
 *
 * `f16 = true`  -> activations + weights stored as f16 (needs the `shader-f16` feature), math in f32
 * `f16 = false` -> everything f32
 */

// Conv kernels are generated per (output-channel tile, kernel size) by buildConvKernel() below (tiled GEMM with implicit im2col).

export function buildKernels(f16) {
  const T = f16 ? "f16" : "f32";
  const header = (f16 ? "enable f16;\n" : "") + `alias T = ${T};\n`;

  // ---------------------------------------------------------------- maxpool 2D (generic k/stride, no padding)
  const maxpool = `${header}
struct P {
  C: u32, H: u32, W: u32, OH: u32,
  OW: u32, kH: u32, kW: u32, sH: u32,
  sW: u32, inOff: u32, outOff: u32, n: u32,
};
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read_write> dst: array<T>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  let ox = i % p.OW;
  let oy = (i / p.OW) % p.OH;
  let c = i / (p.OW * p.OH);
  var m = -3.4e38;
  for (var ky = 0u; ky < p.kH; ky = ky + 1u) {
    let iy = oy * p.sH + ky;
    if (iy >= p.H) { continue; }
    for (var kx = 0u; kx < p.kW; kx = kx + 1u) {
      let ix = ox * p.sW + kx;
      if (ix >= p.W) { continue; }
      m = max(m, f32(src[p.inOff + (c * p.H + iy) * p.W + ix]));
    }
  }
  dst[p.outOff + i] = T(m);
}
`;

  // ---------------------------------------------------------------- bilinear resize (pytorch_half_pixel)
  const resize = `${header}
struct P {
  C: u32, H: u32, W: u32, OH: u32,
  OW: u32, inOff: u32, outOff: u32, n: u32,
  act: u32, p0: u32, p1: u32, p2: u32,
};
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read_write> dst: array<T>;
@group(0) @binding(2) var<uniform> p: P;

fn activate(v: f32, a: u32) -> f32 {
  if (a == 1u) { return max(v, 0.0); }
  if (a == 2u) { return 1.0 / (1.0 + exp(-v)); }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  let ox = i % p.OW;
  let oy = (i / p.OW) % p.OH;
  let c = i / (p.OW * p.OH);
  var sy = (f32(oy) + 0.5) * f32(p.H) / f32(p.OH) - 0.5;
  var sx = (f32(ox) + 0.5) * f32(p.W) / f32(p.OW) - 0.5;
  sy = clamp(sy, 0.0, f32(p.H - 1u));
  sx = clamp(sx, 0.0, f32(p.W - 1u));
  let y0 = u32(floor(sy));
  let x0 = u32(floor(sx));
  let y1 = min(y0 + 1u, p.H - 1u);
  let x1 = min(x0 + 1u, p.W - 1u);
  let fy = sy - f32(y0);
  let fx = sx - f32(x0);
  let b = p.inOff + c * p.H * p.W;
  let v00 = f32(src[b + y0 * p.W + x0]);
  let v01 = f32(src[b + y0 * p.W + x1]);
  let v10 = f32(src[b + y1 * p.W + x0]);
  let v11 = f32(src[b + y1 * p.W + x1]);
  let v = (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy;
  dst[p.outOff + i] = T(activate(v, p.act));
}
`;

  // ---------------------------------------------------------------- elementwise add (+act)
  const add = `${header}
struct P { n: u32, aOff: u32, bOff: u32, outOff: u32, act: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<T>;
@group(0) @binding(1) var<storage, read> b: array<T>;
@group(0) @binding(2) var<storage, read_write> dst: array<T>;
@group(0) @binding(3) var<uniform> p: P;

fn activate(v: f32, act: u32) -> f32 {
  if (act == 1u) { return max(v, 0.0); }
  if (act == 2u) { return 1.0 / (1.0 + exp(-v)); }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  dst[p.outOff + i] = T(activate(f32(a[p.aOff + i]) + f32(b[p.bOff + i]), p.act));
}
`;

  // ---------------------------------------------------------------- standalone activation
  const act = `${header}
struct P { n: u32, inOff: u32, outOff: u32, act: u32 };
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read_write> dst: array<T>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  let v = f32(src[p.inOff + i]);
  var r = v;
  if (p.act == 1u) { r = max(v, 0.0); }
  if (p.act == 2u) { r = 1.0 / (1.0 + exp(-v)); }
  dst[p.outOff + i] = T(r);
}
`;

  // ---------------------------------------------------------------- copy (concat inputs that cannot be written in place)
  const copy = `${header}
struct P { n: u32, srcOff: u32, dstOff: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read_write> dst: array<T>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = src[p.srcOff + i];
}
`;

  // ---------------------------------------------------------------- preprocess: RGBA8 (packed u32) -> normalized CHW
  const pre = `${header}
struct P { n: u32, outOff: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<T>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  let px = rgba[i];
  let r = f32(px & 255u) / 255.0;
  let g = f32((px >> 8u) & 255u) / 255.0;
  let b = f32((px >> 16u) & 255u) / 255.0;
  dst[p.outOff + i] = T((r - 0.485) / 0.229);
  dst[p.outOff + p.n + i] = T((g - 0.456) / 0.224);
  dst[p.outOff + 2u * p.n + i] = T((b - 0.406) / 0.225);
}
`;

  // ---------------------------------------------------------------- postprocess: min/max reduction (single workgroup)
  const minmax = `${header}
struct P { n: u32, inOff: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: P;

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var mn = 3.4e38;
  var mx = -3.4e38;
  for (var i = li; i < p.n; i = i + 256u) {
    let v = f32(src[p.inOff + i]);
    mn = min(mn, v);
    mx = max(mx, v);
  }
  smin[li] = mn;
  smax[li] = mx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (li < s) {
      smin[li] = min(smin[li], smin[li + s]);
      smax[li] = max(smax[li], smax[li + s]);
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    stats[0] = smin[0];
    stats[1] = smax[0];
    atomicStore(&counter[0], 0u);
  }
}
`;

  // ---------------------------------------------------------------- postprocess: threshold -> mask + edge (RGBA8 packed)
  const mask = `${header}
struct P { W: u32, H: u32, inOff: u32, n: u32 };
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * 256u + li;
  if (i >= p.n) { return; }
  let mn = stats[0];
  let range = max(stats[1] - mn, 1e-30);
  let thr = mn + 0.45 * range;
  let x = i % p.W;
  let y = i / p.W;
  var m = 0u;
  var e = 0u;
  if (f32(src[p.inOff + i]) > thr) {
    m = 0xFFFFFFFFu;
    atomicAdd(&counter[0], 1u);
    var edge = (x == 0u) || (x == p.W - 1u) || (y == 0u) || (y == p.H - 1u);
    if (!edge) {
      edge = (f32(src[p.inOff + i - 1u]) <= thr) || (f32(src[p.inOff + i + 1u]) <= thr) ||
             (f32(src[p.inOff + i - p.W]) <= thr) || (f32(src[p.inOff + i + p.W]) <= thr);
    }
    if (edge) { e = 0xFF78FF00u; } // RGBA = (0, 255, 120, 255)
  }
  outBuf[i] = m;
  outBuf[p.n + i] = e;
}
`;

  return { maxpool, resize, add, act, copy, pre, minmax, mask };
}

// ---------------------------------------------------------------------------
// Convolution: port of the @huggingface/kernels `conv-1x1-gemm-tiled-reg`
// algorithm with implicit im2col.
//   GEMM:    M = output channels, N = pixels, K = IC*KH*KW
//   tiles:   BM (oc) x 64 (pixels), BK = 16, thread tile 4 x 4 (TM x TN)
//   A tile:  weights row-major [OCP][KPAD] (K contiguous) loaded as vec4, no bounds checks
//   B tile:  implicit im2col gather, 4 consecutive pixels per vec4
//   shared tiles are vec4<T> (f16 when available), accumulation stays f32
// ---------------------------------------------------------------------------
export const CONV_BN = 64;
export const CONV_BK = 16;

/** Output-channel tile for a layer (limits wasted rows on tiny OC). */
export function convTileBM(oc) {
  if (oc > 32) return 64;
  if (oc > 16) return 32;
  return 16;
}

export function buildConvKernel(f16, BM, KH, KW) {
  const T = f16 ? "f16" : "f32";
  const header = (f16 ? "enable f16;\n" : "") + `alias T = ${T};\n`;
  const WGY = BM / 4;
  const NT = 16 * WGY;
  const accDecl = [0, 1, 2, 3].map((r) => `  var acc${r} = vec4<f32>(0.0);`).join("\n");
  const fma = [0, 1, 2, 3]
    .map((sub) => {
      const comp = "xyzw"[sub];
      return (
        `      let bv${sub} = vec4<f32>(tileB[kv * 4u + ${sub}u][lid.x]);\n` +
        [0, 1, 2, 3].map((r) => `      acc${r} = acc${r} + vec4<f32>(av${r}.${comp}) * bv${sub};`).join("\n")
      );
    })
    .join("\n");
  const aLoads = [0, 1, 2, 3].map((r) => `      let av${r} = vec4<f32>(tileA[aRow + ${r}u][kv]);`).join("\n");
  const store = [0, 1, 2, 3]
    .map((r) => {
      const cols = [0, 1, 2, 3]
        .map((c) => `      if (n0 + ${c}u < HW) { dst[p.outOff + oc * HW + n0 + ${c}u] = T(activate(acc${r}.${"xyzw"[c]} + b, p.act)); }`)
        .join("\n");
      return `    {\n      let oc = m0 + ${r}u;\n      if (oc < p.OC) {\n      let b = bias[oc];\n${cols}\n      }\n    }`;
    })
    .join("\n");
  return `${header}
const BM = ${BM}u;
const BN = ${CONV_BN}u;
const BK = ${CONV_BK}u;
const NT = ${NT}u;
const KH = ${KH}u;
const KW = ${KW}u;
const KK = ${KH * KW}u;

struct P {
  W: u32, H: u32, IC: u32, OC: u32,
  KPAD: u32, padY: u32, padX: u32, dilY: u32,
  dilX: u32, inOff: u32, outOff: u32, act: u32,
};
@group(0) @binding(0) var<storage, read> src: array<T>;
@group(0) @binding(1) var<storage, read> wts: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<T>;
@group(0) @binding(4) var<uniform> p: P;

var<workgroup> tileA: array<array<vec4<T>, 4>, BM>;
var<workgroup> tileB: array<array<vec4<T>, 16>, BK>;

fn activate(v: f32, a: u32) -> f32 {
  if (a == 1u) { return max(v, 0.0); }
  if (a == 2u) { return 1.0 / (1.0 + exp(-v)); }
  return v;
}

@compute @workgroup_size(16, ${WGY}, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let HW = p.H * p.W;
  let K = p.IC * KK;
  let mBase = wg.y * BM;
  let nBase = wg.x * BN;
  let li = lid.y * 16u + lid.x;
${accDecl}

  let numTiles = (K + BK - 1u) / BK;
  for (var kt = 0u; kt < numTiles; kt = kt + 1u) {
    let kBase = kt * BK;
    for (var linear = li; linear < BM * 4u; linear = linear + NT) {
      let ar = linear / 4u;
      let ac4 = linear % 4u;
      tileA[ar][ac4] = wts[((mBase + ar) * p.KPAD + kBase + ac4 * 4u) / 4u];
    }
    for (var linear = li; linear < BK * 16u; linear = linear + NT) {
      let br = linear / 16u;
      let bc4 = linear % 16u;
      let k = kBase + br;
      var bv = vec4<T>(T(0.0));
      if (k < K) {
        let ic = k / KK;
        let r = k - ic * KK;
        let ky = r / KW;
        let kx = r - ky * KW;
        for (var t = 0u; t < 4u; t = t + 1u) {
          let n = nBase + bc4 * 4u + t;
          if (n < HW) {
            let oh = n / p.W;
            let ow = n - oh * p.W;
            let iy = i32(oh) - i32(p.padY) + i32(ky * p.dilY);
            let ix = i32(ow) - i32(p.padX) + i32(kx * p.dilX);
            if (iy >= 0 && iy < i32(p.H) && ix >= 0 && ix < i32(p.W)) {
              bv[t] = src[p.inOff + ic * HW + u32(iy) * p.W + u32(ix)];
            }
          }
        }
      }
      tileB[br][bc4] = bv;
    }
    workgroupBarrier();
    let aRow = lid.y * 4u;
    for (var kv = 0u; kv < 4u; kv = kv + 1u) {
${aLoads}
${fma}
    }
    workgroupBarrier();
  }

  let m0 = mBase + lid.y * 4u;
  let n0 = nBase + lid.x * 4u;
${store}
}
`;
}
