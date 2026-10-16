// EmbeddingGemma 2 (ONNX, transformers.js): shared by the browser worker and Node scripts.
import { AutoConfig, AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";
import { EMBED_DIM, MODEL_DTYPE, MODEL_ID, TASK_PREFIXES, VISION_TOKENS } from "./config.js";

export async function loadEmbedder({ device = "wasm", progress_callback } = {}) {
  const config = await AutoConfig.from_pretrained(MODEL_ID);
  config.audio_config = null; // text + vision only (~285 MB at q4)
  const model = await AutoModel.from_pretrained(MODEL_ID, { config, device, dtype: MODEL_DTYPE, progress_callback });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  processor.image_processor.max_soft_tokens = VISION_TOKENS;

  // Matryoshka: keep the first 256 dims, then re-normalise.
  const toVecs = (t) => {
    const s = t.slice(null, [0, EMBED_DIM]).normalize(2, -1);
    const [n, d] = s.dims, data = s.data;
    return Array.from({ length: n }, (_, i) => Float32Array.from(data.subarray(i * d, (i + 1) * d)));
  };

  return {
    device,
    async embedTexts(texts, task = "classification") {
      const { sentence_embedding } = await model(await processor(texts.map((t) => TASK_PREFIXES[task] + t)));
      return toVecs(sentence_embedding);
    },
    /** rgba: Uint8ClampedArray (w*h*4). One image per call keeps WebGPU under its dispatch limit. */
    async embedImage(rgba, width, height) {
      const img = new RawImage(rgba, width, height, 4).rgb();
      const { sentence_embedding } = await model(await processor(null, img));
      return toVecs(sentence_embedding)[0];
    },
  };
}
