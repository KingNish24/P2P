// src/scripts/lfm-engine/engine.js
/**
 * Full-WebGPU decoder for LFM2.5-VL-450M (text model, 16 layers: 10 gated-conv + 6 GQA attention).
 * f16 storage everywhere (f32 accumulation inside kernels). No fallback: needs the `shader-f16` feature.
 *
 * Usage:
 *   const device = await requestLfmDevice();
 *   const eng = await createLfmEngine({ device, decoder:{graphBytes, source}, embed:{graphBytes, source} });
 *   const r = await eng.generate({ embeds /* Float32Array|Uint16Array(f16) [M*1024] *\/, M, maxNew: 2048, eos: 7, onToken });
 */
import { loadWeights } from "./weights.js";
import * as K from "./kernels.js";

export const KV_INIT_BYTES = 20 * 1024 * 1024; // initial KV budget (all layers, K+V)
export const KV_GROW_BYTES = 2 * 1024 * 1024; // growth step when the cache is full
export const MAX_NEW_TOKENS = 2048;
export const MAX_CTX = 4096; // RoPE table size = hard context limit

const HID = 1024;
const KV_HEADS = 8;
const HEAD_DIM = 64;
const LAYER_TYPES = "ccaccaccacacacac"; // c = conv, a = attention
const N_ATTN = 6;
const KV_BYTES_PER_TOKEN = N_ATTN * 2 * KV_HEADS * HEAD_DIM * 2; // 12288

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
export function toHalfBits(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e;
    let h = m >>> shift;
    const rem = m & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h++;
    return sign | h;
  }
  let h = (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
  return sign | h;
}

export async function requestLfmDevice() {
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU is not available");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  if (!adapter.features.has("shader-f16")) throw new Error("LFM engine needs the WebGPU 'shader-f16' feature (not available on this GPU/browser)");
  return adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
}

function ropeTable(maxCtx) {
  // [pos][cos(32) | sin(32)] f16. theta = 1e6, rotary dim 64.
  const out = new Uint16Array(maxCtx * 64);
  for (let p = 0; p < maxCtx; p++) {
    for (let i = 0; i < 32; i++) {
      const ang = p * Math.pow(1e6, -i / 32);
      out[p * 64 + i] = toHalfBits(Math.cos(ang));
      out[p * 64 + 32 + i] = toHalfBits(Math.sin(ang));
    }
  }
  return out;
}

export async function createLfmEngine({
  device,
  decoder,
  embed,
  kvInitBytes = KV_INIT_BYTES,
  kvGrowBytes = KV_GROW_BYTES,
  maxCtx = MAX_CTX,
  chunk = 256,
  onProgress
}) {
  if (!device.features.has("shader-f16")) throw new Error("LFM engine needs the WebGPU 'shader-f16' feature");
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;

  // ---- weights ---------------------------------------------------------
  const wDec = await loadWeights(device, decoder.graphBytes, decoder.source, {
    skip: ["cos_cache", "sin_cache"],
    onProgress: (p) => onProgress && onProgress(0.9 * p)
  });
  const wEmb = await loadWeights(device, embed.graphBytes, embed.source);
  if (onProgress) onProgress(1);
  const T = (n) => {
    const t = wDec.tensors.get(n);
    if (!t) throw new Error("Missing tensor " + n);
    return t.buffer;
  };
  const proj = (base, K_, N_) => ({ q: T(base + "_quant"), s: T(base + "_scales"), z: T(base + "_zp"), K: K_, N: N_ });
  let weightBytes = 0;
  for (const t of wDec.tensors.values()) weightBytes += t.bytes;
  for (const t of wEmb.tensors.values()) weightBytes += t.bytes;

  // ---- pipelines -------------------------------------------------------
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

  // ---- activations -----------------------------------------------------
  const mk = (size, usage = S | CD | CS, label) => device.createBuffer({ size: Math.max(4, (size + 3) & ~3), usage, label });
  const CH = chunk;
  const hBuf = mk(CH * HID * 2, undefined, "h");
  const xnBuf = mk(CH * HID * 2, undefined, "xn");
  const oBuf = mk(CH * HID * 2, undefined, "o");
  const bcxBuf = mk(CH * 3072 * 2, undefined, "bcx");
  const cyBuf = mk(CH * HID * 2, undefined, "cy");
  const qkvBuf = mk(CH * 2048 * 2, undefined, "qkv");
  const qrotBuf = mk(CH * HID * 2, undefined, "qrot");
  const attnBuf = mk(CH * HID * 2, undefined, "attn");
  const guBuf = mk(CH * 9216 * 2, undefined, "gu");
  const actBuf = mk(CH * 4608 * 2, undefined, "act");
  const logitsBuf = mk(65536 * 2, undefined, "logits");
  const tokBuf = mk(16, undefined, "tok");
  const outTokBuf = mk(4 * (MAX_NEW_TOKENS + 64), undefined, "outTok");
  const metaBuf = mk(16, undefined, "meta");
  const ropeBuf = mk(maxCtx * 64 * 2, undefined, "rope");
  device.queue.writeBuffer(ropeBuf, 0, ropeTable(maxCtx));
  const convState = new Map();
  for (let l = 0; l < 16; l++) if (LAYER_TYPES[l] === "c") convState.set(l, mk(HID * 2 * 2, undefined, "conv" + l));
  const stage = device.createBuffer({ size: 4 * 256, usage: GPUBufferUsage.MAP_READ | CD });
  const actBytes = [hBuf, xnBuf, oBuf, bcxBuf, cyBuf, qkvBuf, qrotBuf, attnBuf, guBuf, actBuf, logitsBuf, outTokBuf, ropeBuf].reduce((s, b) => s + b.size, 0);

  // ---- KV cache --------------------------------------------------------
  let cap = Math.floor(kvInitBytes / KV_BYTES_PER_TOKEN);
  let kBufs = [];
  let vBufs = [];
  const allocKV = (n) => {
    const ks = [];
    const vs = [];
    for (let i = 0; i < N_ATTN; i++) {
      ks.push(mk(n * KV_HEADS * HEAD_DIM * 2, undefined, "k" + i));
      vs.push(mk(n * KV_HEADS * HEAD_DIM * 2, undefined, "v" + i));
    }
    return [ks, vs];
  };
  [kBufs, vBufs] = allocKV(cap);

  // ---- op list ---------------------------------------------------------
  const ops = [];
  const attnOps = []; // rebuilt on KV growth
  const addSimple = (code, constants, bufs, grid) => {
    const p = pipe(code, constants);
    const op = { p, bg: bind(p, bufs), grid };
    ops.push(op);
    return op;
  };
  const addMM = (x, y, pj, { yOff = 0, yStride = pj.N, store = ops } = {}) => {
    const pV = pipe(K.GEMV, { K: pj.K, N: pj.N, Y_OFF: yOff });
    const pG = pipe(K.GEMM, { K: pj.K, N: pj.N, Y_STRIDE: yStride, Y_OFF: yOff });
    const op = {
      mm: true,
      pj,
      pV,
      pG,
      bgV: bind(pV, [{ buffer: x, offset: 0, size: pj.K * 2 }, pj.q, pj.s, pj.z, y]),
      bgG: bind(pG, [x, pj.q, pj.s, pj.z, y, metaBuf])
    };
    store.push(op);
    return op;
  };
  const addRms = (wBuf, hasAdd, dbl) => addSimple(K.ADD_RMS, { N: HID, HAS_ADD: hasAdd ? 1 : 0, DOUBLE: dbl ? 1 : 0 }, [hBuf, oBuf, wBuf, xnBuf], (M) => [M, 1]);

  const opNorm = (l) => T(`model.layers.${l}.operator_layernorm.weight`);
  addRms(opNorm(0), false, false);
  let ai = 0;
  for (let l = 0; l < 16; l++) {
    if (LAYER_TYPES[l] === "c") {
      addMM(xnBuf, bcxBuf, proj(`model_layers_${l}_conv_in_proj_MatMul_weight`, 1024, 3072));
      addSimple(K.CONV_STEP, {}, [bcxBuf, T(`model.layers.${l}.conv.conv.weight`), convState.get(l), cyBuf, metaBuf], () => [4, 1]);
      addMM(cyBuf, oBuf, proj(`model_layers_${l}_conv_out_proj_MatMul_weight`, 1024, 1024));
    } else {
      const a = ai++;
      addMM(xnBuf, qkvBuf, proj(`model_layers_${l}_attn_q_proj_MatMul_weight`, 1024, 1024), { yOff: 0, yStride: 2048 });
      addMM(xnBuf, qkvBuf, proj(`model_layers_${l}_attn_k_proj_MatMul_weight`, 1024, 512), { yOff: 1024, yStride: 2048 });
      addMM(xnBuf, qkvBuf, proj(`model_layers_${l}_attn_v_proj_MatMul_weight`, 1024, 512), { yOff: 1536, yStride: 2048 });
      // q/k norm weights packed [q(64) | k(64)]
      const nw = mk(256, S | CD, "qknorm" + l);
      {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(wDec.tensors.get(`model.layers.${l}.attn.q_norm.layernorm.weight`).buffer, 0, nw, 0, 128);
        enc.copyBufferToBuffer(wDec.tensors.get(`model.layers.${l}.attn.k_norm.layernorm.weight`).buffer, 0, nw, 128, 128);
        device.queue.submit([enc.finish()]);
      }
      const prep = { p: pipe(K.QK_PREP), grid: () => [32, 0] };
      const att = { p: pipe(K.ATTN), grid: () => [16, 0] };
      const rebuild = () => {
        prep.bg = bind(prep.p, [qkvBuf, nw, qrotBuf, kBufs[a], vBufs[a], metaBuf, ropeBuf]);
        att.bg = bind(att.p, [qrotBuf, kBufs[a], vBufs[a], attnBuf, metaBuf]);
      };
      rebuild();
      attnOps.push(rebuild);
      prep.grid = (M) => [32, M];
      att.grid = (M) => [16, M];
      ops.push(prep, att);
      addMM(attnBuf, oBuf, proj(`model_layers_${l}_attn_o_proj_MatMul_weight`, 1024, 1024));
    }
    addRms(T(`model.layers.${l}.ffn_layernorm.weight`), true, false);
    addMM(xnBuf, guBuf, proj(`model_layers_${l}_mlp_gate_proj_MatMul_weight`, 1024, 4608), { yOff: 0, yStride: 9216 });
    addMM(xnBuf, guBuf, proj(`model_layers_${l}_mlp_up_proj_MatMul_weight`, 1024, 4608), { yOff: 4608, yStride: 9216 });
    addSimple(K.SILU_MUL, {}, [guBuf, actBuf, metaBuf], (M) => [Math.ceil((M * 4608) / 256), 1]);
    addMM(actBuf, oBuf, proj(`model_layers_${l}_mlp_down_proj_MatMul_weight`, 4608, 1024));
    if (l < 15) addRms(opNorm(l + 1), true, false);
    else addRms(T("model.layers.16.final_norm_layernorm.weight"), true, true);
  }

  // lm head + argmax (+ embed lookup for decode)
  const lmProj = proj("lm_head_MatMul_weight", 1024, 65536);
  const lmOps = [];
  const lm = addMM(xnBuf, logitsBuf, lmProj, { store: lmOps });
  const argP = pipe(K.ARGMAX);
  const argBG = bind(argP, [logitsBuf, tokBuf, metaBuf, outTokBuf]);
  const E = (n) => wEmb.tensors.get(n).buffer;
  const embP = pipe(K.EMBED);
  const embBG = bind(embP, [tokBuf, E("model_embed_tokens_weight_quant"), E("model_embed_tokens_weight_scales"), E("model_embed_tokens_weight_zp"), hBuf]);

  const runOps = (pass, M, decode) => {
    for (const op of ops) {
      if (op.mm) {
        pass.setPipeline(decode ? op.pV : op.pG);
        pass.setBindGroup(0, decode ? op.bgV : op.bgG);
        if (decode) pass.dispatchWorkgroups(Math.ceil(op.pj.N / 8));
        else pass.dispatchWorkgroups(Math.ceil(op.pj.N / 64), Math.ceil(M / 64));
      } else {
        const g = op.grid(M);
        pass.setPipeline(op.p);
        pass.setBindGroup(0, op.bg);
        pass.dispatchWorkgroups(g[0], g[1] || 1);
      }
    }
  };
  const runHead = (pass, bgV, withArgmax = true) => {
    pass.setPipeline(lm.pV);
    pass.setBindGroup(0, bgV);
    pass.dispatchWorkgroups(Math.ceil(lmProj.N / 8));
    if (!withArgmax) return;
    pass.setPipeline(argP);
    pass.setBindGroup(0, argBG);
    pass.dispatchWorkgroups(1);
  };
  const lmBGDecode = lm.bgV;

  // ---- KV growth -------------------------------------------------------
  const growStep = Math.max(1, Math.floor(kvGrowBytes / KV_BYTES_PER_TOKEN));
  let growths = 0;
  async function ensureCapacity(need) {
    if (need <= cap) return true;
    if (need > maxCtx) return false;
    let nc = cap;
    while (nc < need) nc += growStep;
    nc = Math.min(nc, maxCtx);
    device.pushErrorScope("out-of-memory");
    const [nk, nv] = allocKV(nc);
    const err = await device.popErrorScope();
    if (err) {
      for (const b of [...nk, ...nv]) b.destroy();
      return false;
    }
    const enc = device.createCommandEncoder();
    const bytes = cap * KV_HEADS * HEAD_DIM * 2;
    for (let i = 0; i < N_ATTN; i++) {
      enc.copyBufferToBuffer(kBufs[i], 0, nk[i], 0, bytes);
      enc.copyBufferToBuffer(vBufs[i], 0, nv[i], 0, bytes);
    }
    device.queue.submit([enc.finish()]);
    const oldK = kBufs;
    const oldV = vBufs;
    kBufs = nk;
    vBufs = nv;
    cap = nc;
    growths++;
    attnOps.forEach((f) => f());
    await device.queue.onSubmittedWorkDone();
    for (const b of [...oldK, ...oldV]) b.destroy();
    return true;
  }

  async function readTokens(enc, from, count) {
    const tw = performance.now();
    enc.copyBufferToBuffer(outTokBuf, from * 4, stage, 0, count * 4);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ, 0, count * 4);
    timing.waitMs += performance.now() - tw;
    const out = Array.from(new Uint32Array(stage.getMappedRange(0, count * 4).slice(0)));
    stage.unmap();
    return out;
  }

  function toHalfArray(src) {
    if (src instanceof Uint16Array) return src;
    const out = new Uint16Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = toHalfBits(src[i]);
    return out;
  }

  const stats = { weightBytes, activationBytes: actBytes, get kvCapacityTokens() { return cap; }, get kvBytes() { return cap * KV_BYTES_PER_TOKEN; }, get kvGrowths() { return growths; } };

  // f16 -> f32 lookup table for logits readback in constrained mode
  let h2f = null;
  let logitsF32 = null;
  let logitsStage = null;
  async function readPick(enc, tokensSoFar) {
    if (!h2f) {
      h2f = new Float32Array(65536);
      const tmp = new Uint32Array(1);
      const tf = new Float32Array(tmp.buffer);
      for (let h = 0; h < 65536; h++) {
        const s = (h & 0x8000) << 16;
        const e = (h >> 10) & 31;
        const m = h & 1023;
        if (e === 0) h2f[h] = (h & 0x8000 ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
        else if (e === 31) h2f[h] = m ? NaN : h & 0x8000 ? -Infinity : Infinity;
        else {
          tmp[0] = s | ((e + 112) << 23) | (m << 13);
          h2f[h] = tf[0];
        }
      }
      logitsF32 = new Float32Array(65536);
      logitsStage = device.createBuffer({ size: 65536 * 2, usage: GPUBufferUsage.MAP_READ | CD });
    }
    const tw = performance.now();
    enc.copyBufferToBuffer(logitsBuf, 0, logitsStage, 0, 65536 * 2);
    device.queue.submit([enc.finish()]);
    await logitsStage.mapAsync(GPUMapMode.READ);
    const tp = performance.now();
    timing.waitMs += tp - tw;
    const u = new Uint16Array(logitsStage.getMappedRange());
    for (let i = 0; i < 65536; i++) logitsF32[i] = h2f[u[i]];
    logitsStage.unmap();
    const id = pickFn(logitsF32, tokensSoFar);
    timing.pickMs += performance.now() - tp;
    return id;
  }
  let pickFn = null;
  const timing = { waitMs: 0, pickMs: 0 };

  /**
   * @param {{embeds: Float32Array|Uint16Array, M: number, maxNew?: number, eos?: number|number[], onToken?: (id:number)=>void, signal?: AbortSignal, batch?: number,
   *   pickToken?: (logits: Float32Array, generated: number[]) => number}} o
   *   pickToken: constrained mode (e.g. JSON-schema logits processor). The callback may mutate `logits` and must return the chosen token id.
   *   Without it, decoding is greedy fully on the GPU (batched, no per-token readback of logits).
   */
  async function generate({ embeds, M, maxNew = MAX_NEW_TOKENS, eos = 7, onToken, signal, batch = 4, pickToken }) {
    const constrained = typeof pickToken === "function";
    pickFn = pickToken || null;
    timing.waitMs = 0;
    timing.pickMs = 0;
    let first;
    const eosSet = new Set(Array.isArray(eos) ? eos : [eos]);
    maxNew = Math.min(maxNew, MAX_NEW_TOKENS);
    if (M + 1 > maxCtx) throw new Error(`Prompt too long (${M} tokens, max ${maxCtx})`);
    if (!(await ensureCapacity(M + 1))) throw new Error("KV cache allocation failed for prompt");
    for (const st of convState.values()) device.queue.writeBuffer(st, 0, new Uint16Array(HID * 2));
    const t0 = performance.now();
    // ---- prefill (chunked) ----
    for (let s = 0; s < M; s += CH) {
      const mc = Math.min(CH, M - s);
      const last = s + mc >= M;
      device.queue.writeBuffer(hBuf, 0, toHalfArray(embeds.subarray(s * HID, (s + mc) * HID)));
      device.queue.writeBuffer(metaBuf, 0, new Uint32Array([s, mc, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      runOps(pass, mc, false);
      if (last) {
        const bgLast = bind(lm.pV, [{ buffer: xnBuf, offset: (mc - 1) * HID * 2, size: HID * 2 }, lmProj.q, lmProj.s, lmProj.z, logitsBuf]);
        runHead(pass, bgLast, !constrained);
      }
      pass.end();
      if (last) {
        if (constrained) first = await readPick(enc, []);
        else first = (await readTokens(enc, 0, 1))[0];
      } else device.queue.submit([enc.finish()]);
    }
    const tFirst = performance.now();
    const tokens = [first];
    let reason = "length";
    if (onToken) onToken(first);
    if (eosSet.has(first)) reason = "eos";
    else if (tokens.length >= maxNew) reason = "max_new_tokens";
    device.queue.writeBuffer(metaBuf, 0, new Uint32Array([M, 1, 1, 0]));
    if (constrained) device.queue.writeBuffer(tokBuf, 0, new Uint32Array([first]));
    // ---- decode, constrained: one token per submit, CPU applies the logits processor ----
    while (constrained && reason === "length") {
      if (signal?.aborted) {
        reason = "aborted";
        break;
      }
      const n = tokens.length;
      const pos = M + n - 1;
      if (!(await ensureCapacity(pos + 1))) {
        reason = "kv_full";
        break;
      }
      device.queue.writeBuffer(metaBuf, 0, new Uint32Array([pos, 1, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(embP);
      pass.setBindGroup(0, embBG);
      pass.dispatchWorkgroups(1);
      runOps(pass, 1, true);
      runHead(pass, lmBGDecode, false);
      pass.end();
      const id = await readPick(enc, tokens);
      device.queue.writeBuffer(tokBuf, 0, new Uint32Array([id]));
      tokens.push(id);
      if (onToken) onToken(id);
      if (eosSet.has(id)) reason = "eos";
      else if (tokens.length >= maxNew) reason = "max_new_tokens";
    }
    // ---- decode ----
    while (reason === "length") {
      if (signal?.aborted) {
        reason = "aborted";
        break;
      }
      const n = tokens.length;
      let B = Math.min(batch, maxNew - n);
      const pos = M + n - 1;
      if (!(await ensureCapacity(pos + B))) {
        B = Math.min(B, cap - pos);
        if (B <= 0) {
          reason = "kv_full";
          break;
        }
      }
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let b = 0; b < B; b++) {
        pass.setPipeline(embP);
        pass.setBindGroup(0, embBG);
        pass.dispatchWorkgroups(1);
        runOps(pass, 1, true);
        runHead(pass, lmBGDecode);
      }
      pass.end();
      const got = await readTokens(enc, n, B);
      for (const id of got) {
        tokens.push(id);
        if (onToken) onToken(id);
        if (eosSet.has(id)) {
          reason = "eos";
          break;
        }
        if (tokens.length >= maxNew) {
          reason = "max_new_tokens";
          break;
        }
      }
      // run until a stop reason is set
      if (reason === "length" && tokens.length >= maxNew) reason = "max_new_tokens";
    }
    const t1 = performance.now();
    return {
      tokens,
      reason,
      promptTokens: M,
      ttftMs: tFirst - t0,
      decodeMs: t1 - tFirst,
      waitMs: timing.waitMs,
      pickMs: timing.pickMs,
      tokensPerSec: tokens.length > 1 ? ((tokens.length - 1) / (t1 - tFirst)) * 1000 : 0
    };
  }

  /** Debug helper: run one prefill and return the logits of the last row (f32). */
  async function debugLogits() {
    const rd = device.createBuffer({ size: 65536 * 2, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(logitsBuf, 0, rd, 0, 65536 * 2);
    device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const u = new Uint16Array(rd.getMappedRange().slice(0));
    rd.unmap();
    rd.destroy();
    return u;
  }

  function destroy() {
    for (const t of wDec.tensors.values()) t.buffer.destroy();
    for (const t of wEmb.tensors.values()) t.buffer.destroy();
    for (const b of [hBuf, xnBuf, oBuf, bcxBuf, cyBuf, qkvBuf, qrotBuf, attnBuf, guBuf, actBuf, logitsBuf, tokBuf, outTokBuf, metaBuf, ropeBuf, stage, ...kBufs, ...vBufs, ...convState.values()]) b.destroy();
  }

  // Warm-up: forces shader/pipeline compilation now, so the first real prompt does not pay for it.
  const tw0 = performance.now();
  await generate({ embeds: new Uint16Array(300 * HID), M: 300, maxNew: 2, eos: -1 });
  await device.queue.onSubmittedWorkDone();
  stats.warmupMs = performance.now() - tw0;

  return { generate, debugLogits, stats, destroy, device };
}
