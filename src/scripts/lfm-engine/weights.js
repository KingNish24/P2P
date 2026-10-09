// src/scripts/lfm-engine/weights.js
import { parseOnnx } from "../seg-engine/onnx.js";

/**
 * A data source for ONNX external data. read(offset, length) must resolve to a Uint8Array.
 *  - Blob / File: blobSource(blob)  (no full copy in RAM)
 *  - Node:        fileSource(fs, path)
 */
export function blobSource(blob) {
  return { read: async (offset, length) => new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer()) };
}

/**
 * Parse an ONNX graph and upload every initializer (except skipped names) to GPU storage buffers.
 * Tensors are streamed one by one, so peak RAM = largest tensor (lm_head ~ 33 MB).
 * @returns {Promise<{graph:any, tensors:Map<string,{buffer:GPUBuffer,dims:number[],dataType:number,bytes:number}>}>}
 */
export async function loadWeights(device, graphBytes, source, { skip = [], onProgress } = {}) {
  const graph = parseOnnx(graphBytes);
  const tensors = new Map();
  const skipSet = new Set(skip);
  const names = [...graph.initializers.keys()].filter((n) => !skipSet.has(n));
  let done = 0;
  for (const name of names) {
    const t = graph.initializers.get(name);
    let bytes;
    if (t.external) {
      bytes = await source.read(Number(t.external.offset || 0), Number(t.external.length));
    } else if (t.raw) {
      bytes = t.raw;
    } else {
      throw new Error("Unsupported initializer storage for " + name);
    }
    const size = Math.max(4, (bytes.byteLength + 3) & ~3);
    const buffer = device.createBuffer({ label: name, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (bytes.byteLength % 4 === 0) device.queue.writeBuffer(buffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
    else {
      const pad = new Uint8Array(size);
      pad.set(bytes);
      device.queue.writeBuffer(buffer, 0, pad);
    }
    tensors.set(name, { buffer, dims: t.dims.slice(), dataType: t.dataType, bytes: size });
    done++;
    if (onProgress && done % 16 === 0) onProgress(done / names.length);
  }
  if (onProgress) onProgress(1);
  return { graph, tensors };
}
