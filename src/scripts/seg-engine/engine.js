// src/scripts/seg-engine/engine.js
/**
 * Custom WebGPU inference engine for u2netp (fixed input size, default 320x320).
 * No ONNX Runtime: the ONNX graph is compiled to a flat list of WGSL dispatches
 * (see onnx.js / kernels.js), weights are repacked for the conv kernel, and
 * activations are kept in a small pool of reused storage buffers.
 *
 * API
 *   const engine = await createSegEngine({ modelBytes, f16: "auto", size: 320 });
 *   await engine.infer(rgba)      -> { mask, edge, fgCount }   (RGBA 320x320 in, RGBA masks out)
 *   await engine.forward(rgba)    -> network only (GPU pre-process + layers), waits for the GPU
 *   await engine.readOutput()     -> Float32Array of the sigmoid output d0 (size*size)
 *   await engine.profile(rgba)    -> per-dispatch GPU timings (needs "timestamp-query")
 *   engine.stats, engine.destroy()
 */
import { parseOnnx, compileGraph, tensorToNumbers } from "./onnx.js";
import { buildKernels, buildConvKernel, convTileBM, CONV_BN, CONV_BK } from "./kernels.js";

const align4 = (n) => Math.ceil(n / 4) * 4;
const prod = (a) => a.reduce((x, y) => x * y, 1);

// ---- fp16 helpers ---------------------------------------------------------
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
export function floatToHalfBits(x) {
  _f32[0] = x;
  const b = _u32[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = ((b >>> 23) & 0xff) - 127 + 15;
  const man = b & 0x7fffff;
  if (((b >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
  if (exp >= 31) return sign | 0x7bff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m2 = man | 0x800000;
    const shift = 14 - exp;
    let h = m2 >> shift;
    if ((m2 >> (shift - 1)) & 1) h++;
    return sign | h;
  }
  let h = sign | (exp << 10) | (man >> 13);
  if (man & 0x1000) h++;
  return h;
}
export function halfBitsToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

/** Split n elements into a 1-D or 2-D workgroup grid (65535 limit per dimension). */
function dispatch1D(n, wg = 256) {
  const groups = Math.max(1, Math.ceil(n / wg));
  const x = Math.min(groups, 65535);
  const y = Math.ceil(groups / x);
  return [x, y, 1];
}

function packBias(bias, OC, OCP) {
  const out = new Float32Array(OCP);
  if (bias) out.set(tensorToNumbers(bias).subarray(0, OC));
  return out;
}

/** Row-major [OCP][KPAD] (K contiguous), zero padded so tile loads need no bounds checks. */
function packConvWeights(weight, OC, K, OCP, KPAD, f16) {
  const src = tensorToNumbers(weight);
  const out = f16 ? new Uint16Array(OCP * KPAD) : new Float32Array(OCP * KPAD);
  for (let oc = 0; oc < OC; oc++) {
    for (let k = 0; k < K; k++) {
      const v = src[oc * K + k];
      out[oc * KPAD + k] = f16 ? floatToHalfBits(v) : v;
    }
  }
  return out;
}

async function checkModule(module, label) {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    throw new Error(`WGSL compile error in ${label}: ` + errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join(" | "));
  }
}

export async function createSegEngine(options) {
  const size = options.size || 320;
  const graph = parseOnnx(options.modelBytes);
  const plan = compileGraph(graph, { height: size, width: size });

  if (!navigator.gpu) throw new Error("WebGPU is not supported.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found.");

  const wantF16 = options.f16 === "auto" || options.f16 === undefined ? adapter.features.has("shader-f16") : !!options.f16;
  if (wantF16 && !adapter.features.has("shader-f16")) throw new Error("shader-f16 is not supported on this adapter.");
  const f16 = wantF16;
  const bytesPer = f16 ? 2 : 4;

  const requiredFeatures = [];
  if (f16) requiredFeatures.push("shader-f16");
  const hasTimestamps = adapter.features.has("timestamp-query");
  if (hasTimestamps) requiredFeatures.push("timestamp-query");

  const largest = Math.max(...plan.buffers.map((b) => b.elems)) * bytesPer;
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(largest, 128 * 1024 * 1024)),
    maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(largest, 256 * 1024 * 1024))
  };
  if (largest > requiredLimits.maxStorageBufferBindingSize) {
    throw new Error(`Largest activation buffer (${(largest / 1e6).toFixed(0)} MB) exceeds the device storage-buffer limit.`);
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  const lostPromise = device.lost.then((info) => {
    console.error("[seg-engine] device lost:", info.message);
  });
  void lostPromise;

  // ---- pipelines ----------------------------------------------------------
  const wgsl = buildKernels(f16);
  const pipelines = {};
  const modules = {};
  for (const name of Object.keys(wgsl)) {
    modules[name] = device.createShaderModule({ code: wgsl[name], label: name });
  }
  await Promise.all(Object.entries(modules).map(([name, m]) => checkModule(m, name)));
  await Promise.all(
    Object.keys(modules).map(async (name) => {
      pipelines[name] = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: modules[name], entryPoint: "main" },
        label: name
      });
    })
  );

  // conv pipelines: one per (output-channel tile, kernel size) used by the model
  const convPipelines = new Map();
  {
    const keys = new Map();
    for (const op of plan.ops) {
      if (op.type !== "conv") continue;
      const BM = convTileBM(op.OC);
      keys.set(`${BM}:${op.KH}x${op.KW}`, [BM, op.KH, op.KW]);
    }
    await Promise.all(
      [...keys.entries()].map(async ([key, [BM, KH, KW]]) => {
        const code = buildConvKernel(f16, BM, KH, KW);
        const module = device.createShaderModule({ code, label: `conv ${key}` });
        await checkModule(module, `conv ${key}`);
        convPipelines.set(
          key,
          await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" }, label: `conv ${key}` })
        );
      })
    );
  }

  // ---- buffers ------------------------------------------------------------
  const STORAGE = GPUBufferUsage.STORAGE;
  const actBuffers = plan.buffers.map((b, i) =>
    device.createBuffer({
      size: align4(b.elems * bytesPer),
      usage: STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: `act${i}`
    })
  );
  let weightBytes = 0;
  const ownedBuffers = [...actBuffers];
  const makeBuffer = (size, usage, label) => {
    const b = device.createBuffer({ size: Math.max(16, align4(size)), usage, label });
    ownedBuffers.push(b);
    return b;
  };
  const uploadBuffer = (typed, usage, label) => {
    const b = makeBuffer(typed.byteLength, usage | GPUBufferUsage.COPY_DST, label);
    device.queue.writeBuffer(b, 0, typed.buffer, typed.byteOffset, typed.byteLength);
    weightBytes += align4(typed.byteLength);
    return b;
  };
  const uniform = (u32, label) => {
    const b = makeBuffer(u32.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
    device.queue.writeBuffer(b, 0, u32.buffer, u32.byteOffset, u32.byteLength);
    return b;
  };

  const n = size * size;
  const rgbaBuf = makeBuffer(n * 4, STORAGE | GPUBufferUsage.COPY_DST, "rgba");
  const statsBuf = makeBuffer(8, STORAGE, "stats");
  const counterBuf = makeBuffer(4, STORAGE | GPUBufferUsage.COPY_SRC, "counter");
  const resultBuf = makeBuffer(n * 8, STORAGE | GPUBufferUsage.COPY_SRC, "result");
  const staging = makeBuffer(n * 8 + 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "staging");

  const bufOf = (t) => actBuffers[t.buffer];
  const bind = (pipeline, entries, label) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      label,
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } }))
    });

  // ---- build the dispatch list -----------------------------------------------
  const steps = []; // {name, type, pipeline, group, dispatch, post?}
  const addStep = (type, name, pipeline, group, dispatch, post = false) =>
    steps.push({ type, name, pipeline, group, dispatch, post });

  addStep(
    "pre",
    "preprocess",
    pipelines.pre,
    bind(pipelines.pre, [rgbaBuf, bufOf(plan.input), uniform(new Uint32Array([n, plan.input.elemOffset, 0, 0]), "pre.p")], "pre"),
    dispatch1D(n)
  );

  for (const op of plan.ops) {
    const label = `${op.idx}:${op.type}:${op.name || ""}`;
    if (op.type === "conv") {
      const [IC, H, W] = op.input.shape;
      const KK = op.KH * op.KW;
      const K = IC * KK;
      const BM = convTileBM(op.OC);
      const OCP = Math.ceil(op.OC / BM) * BM;
      // weights [OCP][KPAD]; the uniform slot after OC carries KPAD
      const KPAD = Math.ceil(K / CONV_BK) * CONV_BK;
      const wBuf = uploadBuffer(packConvWeights(op.weight, op.OC, K, OCP, KPAD, f16), STORAGE, `${label}.w`);
      const bBuf = uploadBuffer(packBias(op.bias, op.OC, OCP), STORAGE, `${label}.b`);
      const u = uniform(
        new Uint32Array([W, H, IC, op.OC, KPAD, op.pads[0], op.pads[1], op.dil[0], op.dil[1], op.input.elemOffset, op.output.elemOffset, op.act]),
        `${label}.p`
      );
      if (op.strides[0] !== 1 || op.strides[1] !== 1) throw new Error("Strided conv is not supported by the engine.");
      if (op.output.shape[1] !== H || op.output.shape[2] !== W) throw new Error("Only 'same' convs are supported by the engine.");
      const cp = convPipelines.get(`${BM}:${op.KH}x${op.KW}`);
      addStep(
        "conv",
        label,
        cp,
        bind(cp, [bufOf(op.input), wBuf, bBuf, bufOf(op.output), u], label),
        [Math.ceil((H * W) / CONV_BN), OCP / BM, 1]
      );
    } else if (op.type === "maxpool") {
      if (op.pads.some((p) => p !== 0)) throw new Error("Padded max-pool is not supported by the engine.");
      const [C, H, W] = op.input.shape;
      const [, OH, OW] = op.output.shape;
      const total = C * OH * OW;
      const u = uniform(new Uint32Array([C, H, W, OH, OW, op.k[0], op.k[1], op.strides[0], op.strides[1], op.input.elemOffset, op.output.elemOffset, total]), `${label}.p`);
      addStep("maxpool", label, pipelines.maxpool, bind(pipelines.maxpool, [bufOf(op.input), bufOf(op.output), u], label), dispatch1D(total));
    } else if (op.type === "resize") {
      if (op.mode !== "linear" || op.coord !== "pytorch_half_pixel") throw new Error(`Resize ${op.mode}/${op.coord} is not supported by the engine.`);
      const [C, H, W] = op.input.shape;
      const [, OH, OW] = op.output.shape;
      const total = C * OH * OW;
      const u = uniform(new Uint32Array([C, H, W, OH, OW, op.input.elemOffset, op.output.elemOffset, total, op.act, 0, 0, 0]), `${label}.p`);
      addStep("resize", label, pipelines.resize, bind(pipelines.resize, [bufOf(op.input), bufOf(op.output), u], label), dispatch1D(total));
    } else if (op.type === "add") {
      const total = prod(op.output.shape);
      const u = uniform(new Uint32Array([total, op.a.elemOffset, op.b.elemOffset, op.output.elemOffset, op.act, 0, 0, 0]), `${label}.p`);
      addStep("add", label, pipelines.add, bind(pipelines.add, [bufOf(op.a), bufOf(op.b), bufOf(op.output), u], label), dispatch1D(total));
    } else if (op.type === "act") {
      const total = prod(op.output.shape);
      const u = uniform(new Uint32Array([total, op.input.elemOffset, op.output.elemOffset, op.act]), `${label}.p`);
      addStep("act", label, pipelines.act, bind(pipelines.act, [bufOf(op.input), bufOf(op.output), u], label), dispatch1D(total));
    } else if (op.type === "copy") {
      const total = prod(op.src.shape);
      const dstOff = op.dst.elemOffset + op.dstCOff * op.src.shape[1] * op.src.shape[2];
      const u = uniform(new Uint32Array([total, op.src.elemOffset, dstOff, 0]), `${label}.p`);
      addStep("copy", label, pipelines.copy, bind(pipelines.copy, [bufOf(op.src), bufOf(op.dst), u], label), dispatch1D(total));
    } else throw new Error("Unhandled op " + op.type);
  }

  const out = plan.output;
  addStep(
    "minmax",
    "postprocess:minmax",
    pipelines.minmax,
    bind(pipelines.minmax, [bufOf(out), statsBuf, counterBuf, uniform(new Uint32Array([n, out.elemOffset, 0, 0]), "minmax.p")], "minmax"),
    [1, 1, 1],
    true
  );
  addStep(
    "mask",
    "postprocess:mask+edge",
    pipelines.mask,
    bind(pipelines.mask, [bufOf(out), statsBuf, resultBuf, counterBuf, uniform(new Uint32Array([size, size, out.elemOffset, n]), "mask.p")], "mask"),
    dispatch1D(n),
    true
  );

  const netSteps = steps.filter((s) => !s.post);

  const encodeSteps = (pass, list) => {
    for (const s of list) {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.group);
      pass.dispatchWorkgroups(s.dispatch[0], s.dispatch[1], s.dispatch[2]);
    }
  };

  const upload = (rgba) => {
    if (rgba.byteLength !== n * 4) throw new Error(`Expected ${n * 4} RGBA bytes, got ${rgba.byteLength}.`);
    device.queue.writeBuffer(rgbaBuf, 0, rgba.buffer, rgba.byteOffset, n * 4);
  };

  const activationBytes = actBuffers.reduce((s, b) => s + b.size, 0);
  const stats = {
    f16,
    ops: plan.ops.length,
    dispatches: steps.length,
    weightBytes,
    activationBytes,
    ioBytes: rgbaBuf.size + resultBuf.size + staging.size,
    features: [...device.features].sort(),
    hasTimestamps
  };

  // ---------------------------------------------------------------- public API
  const engine = {
    stats,
    device,
    plan,

    /** Full pipeline: upload, pre-process, network, post-process, read back. */
    async infer(rgba) {
      upload(rgba);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      encodeSteps(pass, steps);
      pass.end();
      enc.copyBufferToBuffer(resultBuf, 0, staging, 0, n * 8);
      enc.copyBufferToBuffer(counterBuf, 0, staging, n * 8, 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return {
        mask: new Uint8ClampedArray(copy, 0, n * 4),
        edge: new Uint8ClampedArray(copy, n * 4, n * 4),
        fgCount: new Uint32Array(copy, n * 8, 1)[0]
      };
    },

    /** Network only (pre-process + layers). Resolves when the GPU is idle. */
    async forward(rgba) {
      upload(rgba);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      encodeSteps(pass, netSteps);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    },

    /** Read the sigmoid output of the last run as Float32Array (size*size). */
    async readOutput() {
      const bytes = align4(n * bytesPer);
      const tmp = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bufOf(out), out.elemOffset * bytesPer, tmp, 0, bytes);
      device.queue.submit([enc.finish()]);
      await tmp.mapAsync(GPUMapMode.READ);
      const raw = tmp.getMappedRange().slice(0);
      tmp.unmap();
      tmp.destroy();
      if (!f16) return new Float32Array(raw, 0, n);
      const half = new Uint16Array(raw, 0, n);
      const res = new Float32Array(n);
      for (let i = 0; i < n; i++) res[i] = halfBitsToFloat(half[i]);
      return res;
    },

    /** Per-dispatch GPU time in ms (one pass per dispatch). Chrome quantizes timestamps unless dev features are on. */
    async profile(rgba) {
      if (!hasTimestamps) throw new Error("timestamp-query is not available on this device.");
      upload(rgba);
      const count = steps.length;
      const querySet = device.createQuerySet({ type: "timestamp", count: count * 2 });
      const resolve = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const read = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      steps.forEach((s, i) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } });
        encodeSteps(pass, [s]);
        pass.end();
      });
      enc.resolveQuerySet(querySet, 0, count * 2, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, read, 0, count * 16);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      querySet.destroy();
      resolve.destroy();
      read.destroy();
      return steps.map((s, i) => ({ name: s.name, type: s.type, ms: Number(t[i * 2 + 1] - t[i * 2]) / 1e6 }));
    },

    destroy() {
      for (const b of ownedBuffers) b.destroy();
      device.destroy();
    }
  };
  return engine;
}

