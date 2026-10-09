// U2NetP salient object detection worker (module worker, plain ES module)
// Receives 320x320 RGBA pixel buffer, returns mask + edge RGBA buffers.

let sessionPromise = null;
let ortModule = null;

const SIZE = 320;
const TOTAL_PIXELS = SIZE * SIZE;
// Reused across jobs (input tensor data never leaves the worker)
const floatData = new Float32Array(3 * TOTAL_PIXELS);

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!self.navigator || !self.navigator.gpu) {
        throw new Error("WebGPU is not supported by your browser or graphics hardware.");
      }
      // WebGPU-only: the webgpu bundle still needs its JSEP runtime files (large .wasm), served from CDN
      const ort = await import("onnxruntime-web/webgpu");
      if (ort.env && ort.env.wasm) {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
      }
      ortModule = ort;
      // u2netp.onnx is fp16 (fp32 input/output); no WASM fallback
      return ort.InferenceSession.create("/models/u2netp.onnx", {
        executionProviders: ["webgpu"]
      });
    })().catch((err) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

async function runSegment(id, rgbaBuffer) {
  const session = await getSession();
  const ort = ortModule;

  const rgba = new Uint8ClampedArray(rgbaBuffer);
  const totalPixels = TOTAL_PIXELS;

  // Normalize channels:
  // mean = [0.485, 0.456, 0.406]
  // std = [0.229, 0.224, 0.225]
  for (let i = 0; i < totalPixels; i++) {
    const p = i * 4;
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    floatData[i] = (r / 255.0 - 0.485) / 0.229;
    floatData[totalPixels + i] = (g / 255.0 - 0.456) / 0.224;
    floatData[2 * totalPixels + i] = (b / 255.0 - 0.406) / 0.225;
  }

  const inputTensor = new ort.Tensor("float32", floatData, [1, 3, SIZE, SIZE]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);

  const outputTensor = results[session.outputNames[0]];
  const outData = outputTensor.data;

  // Calculate min and max of output values
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < outData.length; i++) {
    const v = outData[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = (max - min) || 1;
  // (v - min) / range > 0.45  <=>  v > min + 0.45 * range
  const thr = min + 0.45 * range;

  const mask = new Uint8ClampedArray(totalPixels * 4);
  const edge = new Uint8ClampedArray(totalPixels * 4);
  let fgCount = 0;

  // Single pass: foreground pixels get a mask pixel; foreground pixels on the
  // image border or touching a background pixel also get an edge pixel.
  const mw = SIZE;
  const mh = SIZE;
  for (let y = 0; y < mh; y++) {
    const row = y * mw;
    for (let x = 0; x < mw; x++) {
      const idx = row + x;
      if (outData[idx] <= thr) continue;
      fgCount++;
      const p = idx * 4;
      mask[p] = 255;
      mask[p + 1] = 255;
      mask[p + 2] = 255;
      mask[p + 3] = 255;
      if (x === 0 || x === mw - 1 || y === 0 || y === mh - 1 ||
        outData[idx - 1] <= thr || outData[idx + 1] <= thr ||
        outData[idx - mw] <= thr || outData[idx + mw] <= thr) {
        edge[p] = 0;
        edge[p + 1] = 255;
        edge[p + 2] = 120;
        edge[p + 3] = 255;
      }
    }
  }

  if (typeof outputTensor.dispose === "function") outputTensor.dispose();

  self.postMessage(
    { type: "mask", id, mask: mask.buffer, edge: edge.buffer, fgCount },
    [mask.buffer, edge.buffer]
  );
}

// Serialize segment jobs: never run two inferences at once
const pendingJobs = [];
let draining = false;

function enqueue(job) {
  pendingJobs.push(job);
  if (!draining) drain();
}

async function drain() {
  draining = true;
  while (pendingJobs.length > 0) {
    const job = pendingJobs.shift();
    try {
      await runSegment(job.id, job.rgba);
    } catch (err) {
      self.postMessage({ type: "error", id: job.id, message: String(err) });
    }
  }
  draining = false;
}

self.onmessage = (event) => {
  const data = event.data;
  if (!data || typeof data.type !== "string") return;

  if (data.type === "init") {
    getSession()
      .then(() => {
        self.postMessage({ type: "ready" });
      })
      .catch((err) => {
        self.postMessage({ type: "error", message: String(err) });
      });
    return;
  }

  if (data.type === "segment") {
    try {
      enqueue({ id: data.id, rgba: data.rgba });
    } catch (err) {
      self.postMessage({ type: "error", id: data.id, message: String(err) });
    }
    return;
  }

  // unknown message type -> ignore
};
