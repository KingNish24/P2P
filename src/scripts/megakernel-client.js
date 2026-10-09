// src/scripts/megakernel-client.js
/**
 * MegaKernel Client Runtime
 * Loads the unified public/kernels/megakernel.json bundle locally from Cloudflare CDN.
 * Intercepts all kernel fetch requests to serve WGSL shaders and manifests instantly from memory
 * with zero network latency, zero calls to Hugging Face, and zero CORS errors.
 */

let megaKernelData = null;
let loadPromise = null;

export async function initMegaKernel() {
  if (megaKernelData) return megaKernelData;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      console.log("[MegaKernel] Loading bundled kernels from /kernels/megakernel.json...");
      const t0 = performance.now();
      const res = await fetch("/kernels/megakernel.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      megaKernelData = await res.json();
      const loadTime = performance.now() - t0;
      const count = Object.keys(megaKernelData).length;
      console.log(`[MegaKernel] ✅ Loaded ${count} kernels into memory in ${loadTime.toFixed(1)} ms!`);
    } catch (err) {
      console.warn("[MegaKernel] Failed to load local bundle, will fallback to proxy:", err);
      megaKernelData = {};
    }

    // Install global fetch interceptor
    installMegaKernelInterceptor();
    return megaKernelData;
  })();

  return loadPromise;
}

export function installMegaKernelInterceptor() {
  if (globalThis.__MEGAKERNEL_INTERCEPTED__) return;
  globalThis.__MEGAKERNEL_INTERCEPTED__ = true;

  const origin = typeof self !== "undefined" && self.location ? self.location.origin : "";
  const origFetch = globalThis.fetch;

  globalThis.fetch = function (resource, init) {
    let url = typeof resource === "string" ? resource : (resource && resource.url ? resource.url : "");

    // 1. Check if request is for Hugging Face kernels registry API
    if (url.includes("/api/kernels") && megaKernelData && Object.keys(megaKernelData).length > 0) {
      const kernelList = Object.keys(megaKernelData).map(id => ({ id }));
      return Promise.resolve(new Response(JSON.stringify(kernelList), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      }));
    }

    // 2. Check if request is for a specific kernel artifact (manifest, metadata, or wgsl template)
    const match = url.match(/kernels\/(webgpu-kernels\/[^\/]+)\/resolve\/[^\/]+\/build\/webgpu\/(.+)$/);
    if (match && megaKernelData) {
      const repoId = match[1];
      const fileName = match[2];
      const kernelRepo = megaKernelData[repoId];
      if (kernelRepo && kernelRepo[fileName] !== undefined) {
        return Promise.resolve(new Response(kernelRepo[fileName], {
          status: 200,
          headers: {
            "Content-Type": fileName.endsWith(".json") ? "application/json" : "text/plain; charset=utf-8",
            "Access-Control-Allow-Origin": "*"
          }
        }));
      }
    }

    // 3. Fallback: rewrite any direct huggingface.co requests through /hf/ proxy
    if (origin && !url.startsWith(origin) && url.startsWith("https://huggingface.co/")) {
      const proxiedUrl = url.replace("https://huggingface.co/", `${origin}/hf/`);
      if (typeof resource === "string") {
        resource = proxiedUrl;
      } else if (resource instanceof Request) {
        resource = new Request(proxiedUrl, resource);
      }
    }

    return origFetch(resource, init);
  };
}

// Auto-install interceptor immediately
installMegaKernelInterceptor();
