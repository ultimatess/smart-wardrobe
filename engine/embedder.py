"""EmbeddingGemma 2 wrapper: one shared 256d (Matryoshka) space for images and text."""
import hashlib
import time

import numpy as np
from PIL import Image, ImageOps

from engine import config

try:  # HEIC/HEIF camera-roll photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

# Text-only task prefixes (images take no prefix). See the model card.
TASK_PREFIXES = {
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "search": "task: search result | query: ",
    "similarity": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "none": "",
}


def load_image(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect phone orientation
    return img.convert("RGB")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class WardrobeEmbedder:
    def __init__(self, model_id: str = config.MODEL_ID, dim: int = config.EMBED_DIM):
        import torch
        from sentence_transformers import SentenceTransformer

        self.dim = dim
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        # bf16 on CUDA when supported; never fp16 (activations overflow -> NaN). On Apple MPS fp32 is
        # faster than bf16 (no native bf16 on M1/M2: 670 vs 840 ms/image measured).
        dtype = torch.float32
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        print(f"Loading {model_id} on {self.device} ({dtype}, dim={self.dim})...")
        t0 = time.perf_counter()
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},  # text + vision only (440M)
            processor_kwargs={"image_seq_length": config.VISION_TOKENS, "max_soft_tokens": config.VISION_TOKENS},
        )
        print(f"Model ready in {time.perf_counter() - t0:.1f}s")

    def _encode(self, inputs, batch_size: int = 8) -> np.ndarray:
        out = self.model.encode(
            inputs,
            batch_size=batch_size,
            truncate_dim=self.dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _l2(np.asarray(out, dtype=np.float32))

    def embed_pil(self, images: list[Image.Image]) -> np.ndarray:
        return self._encode([{"image": im} for im in images], batch_size=4)

    def embed_images(self, image_paths) -> np.ndarray:
        return self.embed_pil([load_image(p) for p in image_paths])

    def embed_texts(self, prompts: list[str], task: str = "classification") -> np.ndarray:
        prefix = TASK_PREFIXES[task]
        return self._encode([prefix + p for p in prompts], batch_size=32)


class FakeEmbedder:
    """Deterministic stand-in for tests: text -> hashed bag-of-words; images -> pixel stats.

    Shares one space so that identical words produce overlapping vectors, which is enough
    to exercise dedup, classification plumbing and recommender math without model weights.
    """

    def __init__(self, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.device = "cpu"

    def _word_vec(self, word: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(word.lower().encode()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    def embed_texts(self, prompts: list[str], task: str = "classification") -> np.ndarray:
        vecs = []
        for p in prompts:
            words = [w.strip(".,+/()") for w in p.split() if len(w) > 2]
            vecs.append(sum((self._word_vec(w) for w in words), np.zeros(self.dim, np.float32)))
        return _l2(np.stack(vecs))

    def embed_pil(self, images: list[Image.Image]) -> np.ndarray:
        vecs = []
        for im in images:
            small = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).ravel() / 255.0
            v = np.zeros(self.dim, np.float32)
            v[: small.size] = small - small.mean()
            v[-1] = 1.0
            vecs.append(v)
        return _l2(np.stack(vecs))

    def embed_images(self, image_paths) -> np.ndarray:
        return self.embed_pil([load_image(p) for p in image_paths])


_EMBEDDER = None


def get_embedder():
    """Process-wide singleton (model load is the slow part)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import os

        _EMBEDDER = FakeEmbedder() if os.environ.get("WARDROBE_FAKE_EMBEDDER") else WardrobeEmbedder()
    return _EMBEDDER
