// src/scripts/seg-engine/onnx.js
/**
 * Minimal ONNX reader + compiler for the u2netp segmentation model.
 *
 * - parseOnnx(): decodes the protobuf wire format (no dependencies)
 * - compileGraph(): folds the shape-computation nodes (Shape/Gather/Unsqueeze/Slice/Concat/Cast/Constant)
 *   for a fixed input size and produces a flat list of tensor ops with buffer lifetimes.
 *
 * Tensor layout used by the engine: NCHW with N = 1, stored as [C][H][W].
 * Concat along channels is therefore a contiguous slice, so single-use producers
 * write straight into their slice of the concat output (no copy).
 */

// --------------------------------------------------------------------------
// Protobuf wire reader
// --------------------------------------------------------------------------
class Reader {
  constructor(buf) {
    this.b = buf;
    this.p = 0;
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  eof() {
    return this.p >= this.b.length;
  }
  varint() {
    const start = this.p;
    while (this.b[this.p++] & 0x80);
    const len = this.p - start;
    if (len <= 7) {
      let r = 0;
      let mul = 1;
      for (let i = 0; i < len; i++) {
        r += (this.b[start + i] & 0x7f) * mul;
        mul *= 128;
      }
      return r;
    }
    let big = 0n;
    for (let i = 0; i < len; i++) big |= BigInt(this.b[start + i] & 0x7f) << BigInt(7 * i);
    return Number(BigInt.asIntN(64, big));
  }
  fixed32() {
    const v = this.dv.getFloat32(this.p, true);
    this.p += 4;
    return v;
  }
  skip(wt) {
    if (wt === 0) this.varint();
    else if (wt === 1) this.p += 8;
    else if (wt === 2) this.p += this.varint();
    else if (wt === 5) this.p += 4;
    else throw new Error("Unsupported wire type " + wt);
  }
}

/** Iterate fields of a message. cb(fieldNumber, wireType, value, reader) */
function readMessage(buf, cb) {
  const r = new Reader(buf);
  while (!r.eof()) {
    const tag = r.varint();
    const field = Math.floor(tag / 8);
    const wt = tag % 8;
    if (wt === 0) cb(field, wt, r.varint());
    else if (wt === 2) {
      const len = r.varint();
      const sub = buf.subarray(r.p, r.p + len);
      r.p += len;
      cb(field, wt, sub);
    } else if (wt === 5) cb(field, wt, r.fixed32());
    else if (wt === 1) {
      cb(field, wt, null);
      r.p += 8;
    } else throw new Error("Unsupported wire type " + wt);
  }
}

const utf8 = new TextDecoder();

function packedVarints(sub) {
  const r = new Reader(sub);
  const out = [];
  while (!r.eof()) out.push(r.varint());
  return out;
}

function parseTensorProto(buf) {
  const t = { dims: [], dataType: 0, name: "", raw: null, floats: null, ints: null };
  readMessage(buf, (f, wt, v) => {
    if (f === 1) {
      if (wt === 2) t.dims.push(...packedVarints(v));
      else t.dims.push(v);
    } else if (f === 2) t.dataType = v;
    else if (f === 4 && wt === 2) {
      t.floats = new Float32Array(v.slice().buffer);
    } else if (f === 7) {
      t.ints = t.ints || [];
      if (wt === 2) t.ints.push(...packedVarints(v));
      else t.ints.push(v);
    } else if (f === 5) {
      t.ints = t.ints || [];
      if (wt === 2) t.ints.push(...packedVarints(v));
      else t.ints.push(v);
    } else if (f === 8) t.name = utf8.decode(v);
    else if (f === 9) t.raw = v;
  });
  return t;
}

function parseAttribute(buf) {
  const a = { name: "", f: null, i: null, s: null, t: null, floats: null, ints: null };
  readMessage(buf, (f, wt, v) => {
    if (f === 1) a.name = utf8.decode(v);
    else if (f === 2) a.f = v;
    else if (f === 3) a.i = v;
    else if (f === 4) a.s = utf8.decode(v);
    else if (f === 5) a.t = parseTensorProto(v);
    else if (f === 7) {
      a.floats = a.floats || [];
      if (wt === 2) {
        const fl = new Float32Array(v.slice().buffer);
        a.floats.push(...fl);
      } else a.floats.push(v);
    } else if (f === 8) {
      a.ints = a.ints || [];
      if (wt === 2) a.ints.push(...packedVarints(v));
      else a.ints.push(v);
    }
  });
  return a;
}

function parseNode(buf) {
  const n = { input: [], output: [], name: "", op: "", attrs: {} };
  readMessage(buf, (f, wt, v) => {
    if (f === 1) n.input.push(utf8.decode(v));
    else if (f === 2) n.output.push(utf8.decode(v));
    else if (f === 3) n.name = utf8.decode(v);
    else if (f === 4) n.op = utf8.decode(v);
    else if (f === 5) {
      const a = parseAttribute(v);
      n.attrs[a.name] = a;
    }
  });
  return n;
}

function parseValueInfoName(buf) {
  let name = "";
  readMessage(buf, (f, wt, v) => {
    if (f === 1) name = utf8.decode(v);
  });
  return name;
}

/** @param {Uint8Array} bytes ONNX model file */
export function parseOnnx(bytes) {
  const graph = { nodes: [], initializers: new Map(), inputs: [], outputs: [] };
  readMessage(bytes, (f, wt, v) => {
    if (f !== 7) return;
    readMessage(v, (gf, gwt, gv) => {
      if (gf === 1) graph.nodes.push(parseNode(gv));
      else if (gf === 5) {
        const t = parseTensorProto(gv);
        graph.initializers.set(t.name, t);
      } else if (gf === 11) graph.inputs.push(parseValueInfoName(gv));
      else if (gf === 12) graph.outputs.push(parseValueInfoName(gv));
    });
  });
  // graph inputs may list initializers as well (old IR versions)
  graph.inputs = graph.inputs.filter((n) => !graph.initializers.has(n));
  return graph;
}

// --------------------------------------------------------------------------
// Tensor data helpers
// --------------------------------------------------------------------------
const DT_FLOAT = 1;
const DT_INT32 = 6;
const DT_INT64 = 7;
const DT_FLOAT16 = 10;

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

/** Decode any supported TensorProto into a plain number array / Float32Array. */
export function tensorToNumbers(t) {
  const count = t.dims.reduce((a, b) => a * b, 1);
  if (t.dataType === DT_FLOAT) {
    if (t.raw) {
      const copy = t.raw.slice();
      return new Float32Array(copy.buffer, 0, count);
    }
    return t.floats ? Float32Array.from(t.floats) : new Float32Array(0);
  }
  if (t.dataType === DT_FLOAT16) {
    const out = new Float32Array(count);
    const src = new Uint16Array(t.raw.slice().buffer, 0, count);
    for (let i = 0; i < count; i++) out[i] = halfToFloat(src[i]);
    return out;
  }
  if (t.dataType === DT_INT64) {
    if (t.raw) {
      const big = new BigInt64Array(t.raw.slice().buffer, 0, count);
      return Array.from(big, (x) => Number(x));
    }
    return t.ints ? t.ints.slice() : [];
  }
  if (t.dataType === DT_INT32) {
    if (t.raw) return Array.from(new Int32Array(t.raw.slice().buffer, 0, count));
    return t.ints ? t.ints.slice() : [];
  }
  throw new Error("Unsupported tensor data type " + t.dataType + " for " + t.name);
}

// --------------------------------------------------------------------------
// Graph compiler
// --------------------------------------------------------------------------
const prod = (a) => a.reduce((x, y) => x * y, 1);

/**
 * @param {ReturnType<typeof parseOnnx>} graph
 * @param {{height?: number, width?: number}} [opts]
 */
export function compileGraph(graph, opts = {}) {
  const H0 = opts.height || 320;
  const W0 = opts.width || 320;

  // consumer counts (by name) to decide fusion / aliasing
  const consumers = new Map();
  for (const n of graph.nodes) for (const i of n.input) consumers.set(i, (consumers.get(i) || 0) + 1);
  for (const o of graph.outputs) consumers.set(o, (consumers.get(o) || 0) + 1);

  const tensors = []; // {id, shape:[C,H,W], view, producer, uses, ...}
  const ops = [];
  const byName = new Map(); // value name -> tensor
  const consts = new Map(); // value name -> {data:number[], dims:number[]}

  const newTensor = (shape, extra = {}) => {
    const t = { id: tensors.length, shape, view: null, def: -1, lastUse: -1, kind: "act", ...extra };
    tensors.push(t);
    return t;
  };
  const getT = (name) => {
    const t = byName.get(name);
    if (!t) throw new Error("Unknown tensor " + name);
    return t;
  };
  const getConst = (name) => {
    if (consts.has(name)) return consts.get(name);
    const init = graph.initializers.get(name);
    if (init) {
      const c = { data: Array.from(tensorToNumbers(init)), dims: init.dims.slice() };
      consts.set(name, c);
      return c;
    }
    return null;
  };

  // graph input
  const inName = graph.inputs[0];
  const inputT = newTensor([3, H0, W0], { kind: "input" });
  byName.set(inName, inputT);

  const emit = (op) => {
    op.idx = ops.length;
    ops.push(op);
    return op;
  };
  const attrInts = (n, name, dflt) => (n.attrs[name] && n.attrs[name].ints ? n.attrs[name].ints : dflt);
  const attrInt = (n, name, dflt) => (n.attrs[name] && n.attrs[name].i !== null ? n.attrs[name].i : dflt);
  const attrStr = (n, name, dflt) => (n.attrs[name] && n.attrs[name].s !== null ? n.attrs[name].s : dflt);

  const fusable = new Set(["conv", "resize", "add"]);

  for (const n of graph.nodes) {
    const out0 = n.output[0];
    switch (n.op) {
      case "Constant": {
        const a = n.attrs.value;
        if (!a || !a.t) throw new Error("Unsupported Constant form in " + n.name);
        const t = a.t;
        const count = prod(t.dims);
        const data = count === 0 ? [] : Array.from(tensorToNumbers(t));
        consts.set(out0, { data, dims: t.dims.slice() });
        break;
      }
      case "Shape": {
        const t = getT(n.input[0]);
        consts.set(out0, { data: [1, ...t.shape], dims: [t.shape.length + 1] });
        break;
      }
      case "Gather": {
        const d = getConst(n.input[0]);
        const idx = getConst(n.input[1]);
        if (!d || !idx) throw new Error("Dynamic Gather not supported");
        const data = idx.data.map((i) => d.data[i < 0 ? i + d.data.length : i]);
        consts.set(out0, { data, dims: idx.dims.slice() });
        break;
      }
      case "Unsqueeze": {
        const d = getConst(n.input[0]);
        if (!d) throw new Error("Dynamic Unsqueeze not supported");
        const axes = attrInts(n, "axes", []);
        const dims = d.dims.slice();
        for (const ax of [...axes].sort((a, b) => a - b)) dims.splice(ax < 0 ? ax + dims.length + 1 : ax, 0, 1);
        consts.set(out0, { data: d.data.slice(), dims });
        break;
      }
      case "Slice": {
        const d = getConst(n.input[0]);
        const starts = getConst(n.input[1]).data;
        const ends = getConst(n.input[2]).data;
        if (!d || d.dims.length !== 1) throw new Error("Only 1-D const Slice supported");
        const len = d.data.length;
        const clamp = (v) => Math.max(0, Math.min(len, v < 0 ? v + len : v));
        consts.set(out0, { data: d.data.slice(clamp(starts[0]), clamp(ends[0])), dims: [clamp(ends[0]) - clamp(starts[0])] });
        break;
      }
      case "Cast": {
        const d = getConst(n.input[0]);
        if (d) {
          consts.set(out0, { data: d.data.slice(), dims: d.dims.slice() });
        } else {
          // dtype is managed by the engine, so a Cast on an activation is a pass-through alias
          byName.set(out0, getT(n.input[0]));
        }
        break;
      }
      case "Concat": {
        const allConst = n.input.every((i) => getConst(i));
        if (allConst) {
          const data = [];
          for (const i of n.input) data.push(...getConst(i).data);
          consts.set(out0, { data, dims: [data.length] });
          break;
        }
        const ins = n.input.map(getT);
        const [, h, w] = ins[0].shape;
        for (const t of ins) if (t.shape[1] !== h || t.shape[2] !== w) throw new Error("Concat spatial mismatch");
        const C = ins.reduce((s, t) => s + t.shape[0], 0);
        const out = newTensor([C, h, w]);
        byName.set(out0, out);
        let cOff = 0;
        for (const [k, t] of ins.entries()) {
          const single = consumers.get(n.input[k]) === 1;
          const canAlias = single && t.kind === "act" && !t.view && t.producer && t.producer.type !== "concat_copy" && !t.aliased;
          if (canAlias) {
            t.view = { parent: out, cOff };
            t.aliased = true;
          } else {
            emit({ type: "copy", src: t, dst: out, dstCOff: cOff, name: n.name });
          }
          cOff += t.shape[0];
        }
        out.producer = { type: "concat" };
        break;
      }
      case "Conv": {
        const x = getT(n.input[0]);
        const w = graph.initializers.get(n.input[1]);
        const b = n.input[2] ? graph.initializers.get(n.input[2]) : null;
        if (!w) throw new Error("Conv weight must be an initializer");
        const [OC, ICg, KH, KW] = w.dims;
        const group = attrInt(n, "group", 1);
        if (group !== 1) throw new Error("Grouped conv not supported");
        if (ICg !== x.shape[0]) throw new Error("Conv channel mismatch");
        const pads = attrInts(n, "pads", [0, 0, 0, 0]);
        const strides = attrInts(n, "strides", [1, 1]);
        const dil = attrInts(n, "dilations", [1, 1]);
        const oh = Math.floor((x.shape[1] + pads[0] + pads[2] - dil[0] * (KH - 1) - 1) / strides[0]) + 1;
        const ow = Math.floor((x.shape[2] + pads[1] + pads[3] - dil[1] * (KW - 1) - 1) / strides[1]) + 1;
        const out = newTensor([OC, oh, ow]);
        const op = emit({
          type: "conv",
          input: x,
          output: out,
          weight: w,
          bias: b,
          OC,
          IC: ICg,
          KH,
          KW,
          pads,
          strides,
          dil,
          act: 0,
          name: n.name
        });
        out.producer = op;
        byName.set(out0, out);
        break;
      }
      case "MaxPool": {
        const x = getT(n.input[0]);
        const k = attrInts(n, "kernel_shape", [2, 2]);
        const strides = attrInts(n, "strides", [1, 1]);
        const pads = attrInts(n, "pads", [0, 0, 0, 0]);
        const ceil = attrInt(n, "ceil_mode", 0);
        const f = ceil ? Math.ceil : Math.floor;
        const oh = f((x.shape[1] + pads[0] + pads[2] - k[0]) / strides[0]) + 1;
        const ow = f((x.shape[2] + pads[1] + pads[3] - k[1]) / strides[1]) + 1;
        const out = newTensor([x.shape[0], oh, ow]);
        out.producer = emit({ type: "maxpool", input: x, output: out, k, strides, pads, name: n.name });
        byName.set(out0, out);
        break;
      }
      case "Resize": {
        const x = getT(n.input[0]);
        const sizesName = n.input[3];
        const scalesName = n.input[2];
        let oh;
        let ow;
        const sizes = sizesName ? getConst(sizesName) : null;
        const scales = scalesName ? getConst(scalesName) : null;
        if (sizes && sizes.data.length === 4) {
          oh = sizes.data[2];
          ow = sizes.data[3];
        } else if (scales && scales.data.length === 4) {
          oh = Math.floor(x.shape[1] * scales.data[2]);
          ow = Math.floor(x.shape[2] * scales.data[3]);
        } else throw new Error("Resize without static sizes/scales");
        const mode = attrStr(n, "mode", "nearest");
        const coord = attrStr(n, "coordinate_transformation_mode", "half_pixel");
        const out = newTensor([x.shape[0], oh, ow]);
        out.producer = emit({ type: "resize", input: x, output: out, mode, coord, act: 0, name: n.name });
        byName.set(out0, out);
        break;
      }
      case "Add": {
        const a = getT(n.input[0]);
        const b = getT(n.input[1]);
        if (prod(a.shape) !== prod(b.shape)) throw new Error("Add shape mismatch");
        const out = newTensor(a.shape.slice());
        out.producer = emit({ type: "add", a, b, output: out, act: 0, name: n.name });
        byName.set(out0, out);
        break;
      }
      case "Relu":
      case "Sigmoid": {
        const x = getT(n.input[0]);
        const code = n.op === "Relu" ? 1 : 2;
        const p = x.producer;
        if (p && fusable.has(p.type) && p.act === 0 && consumers.get(n.input[0]) === 1 && !x.view) {
          p.act = code; // fused into the producer
          byName.set(out0, x);
        } else {
          const out = newTensor(x.shape.slice());
          out.producer = emit({ type: "act", input: x, output: out, act: code, name: n.name });
          byName.set(out0, out);
        }
        break;
      }
      default:
        throw new Error("Unsupported op " + n.op + " (" + n.name + ")");
    }
  }

  const output = getT(graph.outputs[0]);
  output.kind = "output";

  // ---- lifetimes -------------------------------------------------------
  const rootOf = (t) => (t.view ? rootOf(t.view.parent) : t);
  const elemOffset = (t) => (t.view ? elemOffset(t.view.parent) + t.view.cOff * t.shape[1] * t.shape[2] : 0);
  for (const t of tensors) {
    t.root = rootOf(t);
    t.elemOffset = elemOffset(t);
  }
  const readsOf = (op) => {
    if (op.type === "add") return [op.a, op.b];
    if (op.type === "copy") return [op.src];
    return [op.input];
  };
  const writesOf = (op) => [op.type === "copy" ? op.dst : op.output];
  inputT.root.def = -1;
  ops.forEach((op, i) => {
    for (const t of writesOf(op)) {
      const r = t.root;
      if (r.def === -1 && r !== inputT.root) r.def = i;
      else if (r !== inputT.root) r.def = Math.min(r.def, i);
      r.lastUse = Math.max(r.lastUse, i);
    }
    for (const t of readsOf(op)) t.root.lastUse = Math.max(t.root.lastUse, i);
  });
  // output buffer is consumed by post-processing after all ops
  output.root.lastUse = ops.length + 1;

  // ---- buffer pool assignment ------------------------------------------
  const roots = tensors.filter((t) => t.root === t);
  const buffers = []; // {id, elems}
  const free = [];
  const live = [];
  const assign = (t) => {
    const need = prod(t.shape);
    let pick = -1;
    for (let i = 0; i < free.length; i++) {
      const b = buffers[free[i]];
      if (b.elems >= need && (pick === -1 || b.elems < buffers[free[pick]].elems)) pick = i;
    }
    let bufId;
    if (pick >= 0) bufId = free.splice(pick, 1)[0];
    else {
      bufId = buffers.length;
      buffers.push({ id: bufId, elems: need });
    }
    t.buffer = bufId;
    live.push(t);
  };
  const releaseBefore = (i) => {
    for (let k = live.length - 1; k >= 0; k--) {
      if (live[k].lastUse < i) {
        free.push(live[k].buffer);
        live.splice(k, 1);
      }
    }
  };
  assign(inputT);
  const byDef = roots.filter((t) => t !== inputT).sort((a, b) => a.def - b.def);
  let cursor = 0;
  for (let i = 0; i < ops.length; i++) {
    releaseBefore(i);
    while (cursor < byDef.length && byDef[cursor].def <= i) {
      assign(byDef[cursor]);
      cursor++;
    }
  }
  for (const t of tensors) {
    t.buffer = t.root.buffer;
  }

  const totalElems = buffers.reduce((s, b) => s + b.elems, 0);
  return { ops, tensors, buffers, input: inputT, output, totalActivationElems: totalElems, consts };
}
