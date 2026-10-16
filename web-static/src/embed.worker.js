// Runs EmbeddingGemma 2 off the main thread. WebGPU when available, WASM otherwise.
import { env } from "@huggingface/transformers";
import { loadEmbedder } from "./embed-core.js";

env.allowLocalModels = false; // weights come from the Hugging Face CDN once, then the browser cache
let emb = null;

async function init() {
  const progress_callback = (p) => self.postMessage({ progress: p });
  const gpu = self.navigator?.gpu && (await self.navigator.gpu.requestAdapter().catch(() => null));
  if (gpu) {
    try { return await loadEmbedder({ device: "webgpu", progress_callback }); }
    catch (e) { self.postMessage({ progress: { status: "fallback", message: String(e) } }); }
  }
  return loadEmbedder({ device: "wasm", progress_callback });
}

self.onmessage = async ({ data: { id, op, args } }) => {
  try {
    if (op === "init") {
      emb ||= await init();
      return self.postMessage({ id, result: { device: emb.device } });
    }
    emb ||= await init();
    const result = op === "texts" ? await emb.embedTexts(...args) : await emb.embedImage(...args);
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({ id, error: e?.message || String(e) });
  }
};
