# Smart Wardrobe: Local-First Architecture & Implementation Spec

## 1. Executive Summary & Core Philosophy
A 100% privacy-preserving, local-first Smart Wardrobe application powered by **EmbeddingGemma 2**. By aligning images and text in a unified vector space (truncated to **256 dimensions** via Matryoshka Representation Learning), the system performs garment detection, zero-shot metadata classification, deduplication, and context-aware daily outfit recommendation entirely offline with **zero runtime LLM token costs**.

---

## 2. Locked Product Decisions ("The Golden Spec")

| Dimension | Locked Decision | Engineering Implementation |
| :--- | :--- | :--- |
| **1. Onboarding** | **Passive Camera Roll Ingestion + Face Anchoring** | Scan existing gallery photos ("in-the-wild" selfies, group shots, events). Match the user's face against 1 reference selfie (`MediaPipe`) and crop the body bounding box below the face. |
| **2. Granularity** | **Hybrid: Outfit-First, Split-on-Demand** | Index full "Proven Outfits" (Whole Looks) as the primary unit for high-trust recommendations; store upper/lower torso bounding boxes as a secondary layer for future mix-and-match. |
| **3. Target Platform** | **Local-First Web/PWA Sandbox $\rightarrow$ Mobile (LiteRT)** | Build Gate 1–3 as a Python/FastAPI + Local Web UI workspace; port to Android/iOS via LiteRT + `sqlite-vec` for production background gallery scanning. |
| **4. Taxonomy** | **Bi-Cultural & Climate-Aware (Indian + Western + Fusion)** | Pre-compute 150+ text anchor vectors covering Indian ethnic (kurtas, sarees, bandhgalas, silk dhotis), fusion, and Western wear, mapped to humid/monsoon/mild climates. |
| **5. Daily Context** | **Auto-Weather + 1-Tap Vibe Pill** | Combine coarse local weather (temp/humidity/rain) + day of week with a 1-Tap UI toggle (`[🏢 Work]`, `[☕ Casual]`, `[✨ Festive]`, `[✈️ Travel]`). |
| **6. Killer Feature** | **Anti-Repeat Radar** | Tracks wear timestamps and GPS/location clusters from EXIF metadata to penalize recently worn outfits and warn against repeating the same look with the same group/venue. |

---

## 3. Zero-Token Local Pipeline Architecture

1. **Garment Isolation & Deduplication:**
   * Local detector isolates the user's outfit crop from full-body/group photos.
   * Cosine similarity ($\ge 0.92$) merges duplicate photos of the same outfit worn across different days and increments `wear_count`.
2. **Zero-Shot Metadata Extraction:**
   * Pre-compute 256d text embeddings for the fixed garment taxonomy (`taxonomy_anchors.bin`).
   * Compute dot-product between cropped image vectors and taxonomy vectors to assign category, fabric, color, and formality in $< 20\text{ ms}$.
3. **Temporal & Context Tagging:**
   * Parse photo **EXIF timestamps and GPS** locally.
   * Map timestamps to seasons/time-of-day and cluster GPS coordinates to infer venue context (e.g., office vs. travel vs. function hall).
4. **Dynamic Collections:**
   * Combine K-Means vector clustering with SQLite relational filters to auto-populate collections (*"Monsoon Office Wear"*, *"Ethnic & Festive"*, *"Seldom Worn Gems"*).
5. **Daily Recommendation Engine:**
   * Construct a composite query vector:
     $$\vec{Q}_{\text{day}} = \text{Norm}\left(\alpha \cdot \vec{E}_{\text{weather+vibe}} + (1 - \alpha) \cdot \vec{U}_{\text{style\_centroid}}\right)$$
   * Score candidate outfits using cosine similarity penalized by an exponential wear-recency decay function:
     $$\text{Score}(i) = \left(\vec{Q}_{\text{day}} \cdot \vec{V}_i\right) \times \left(1 - e^{-\lambda \cdot \Delta t_i}\right)$$
     where $\Delta t_i$ is days since outfit $i$ was last worn.

---

## 4. The 4-Gate Validation Loop (Minimum Token Spend)

* **Gate 1: The 50-Photo Python Spike (Zero UI)**
  * *Action:* Run the local ingestion & embedding script on 50 real camera-roll photos.
  * *Pass Criteria:* $> 85\%$ precision in grouping duplicate outfits across different days and matching primary garment categories.
* **Gate 2: Taxonomy & Cold-Start Calibration**
  * *Action:* Benchmark the 150-label Indian + Western anchor bank against diverse outfits using EmbeddingGemma 2 task prefixes.
  * *Pass Criteria:* Zero-shot top-1 or top-2 label matches human tagging on $9/10$ items.
* **Gate 3: Recommendation Math Backtest**
  * *Action:* Feed 5 mock daily scenarios (e.g., *"34°C humid Wednesday + Work"*, *"Rainy Sunday evening + Festive"*) into the decay-weighted scoring formula in CLI.
  * *Pass Criteria:* Top-3 ranked outfits are weather-appropriate, match the vibe pill, and respect the Anti-Repeat Radar.
* **Gate 4: Spec-Locked UI & Mobile Packaging**
  * *Action:* Generate the FastAPI endpoints and local frontend dashboard against the frozen SQLite schema.

---

## 5. Project Workspace Structure

```text
smart-wardrobe/
├── pyproject.toml              # Managed by uv (dependencies & scripts)
├── SPEC.md                     # This specification file
├── data/
│   ├── reference_selfie.jpg    # 1 anchor photo of the user's face
│   ├── raw_gallery/            # Drop 50 test camera-roll photos here
│   ├── crops/                  # Auto-generated person/outfit bounding boxes
│   ├── taxonomy_anchors.bin    # Pre-computed 256d text vectors (generated once)
│   └── wardrobe.db             # SQLite + sqlite-vec single-file database
├── engine/
│   ├── __init__.py
│   ├── db.py                   # SQLite + sqlite-vec schema & vector queries
│   ├── embedder.py             # EmbeddingGemma 2 (256d Matryoshka wrapper)
│   ├── ingest.py               # EXIF parser + MediaPipe person crop + dedup
│   ├── taxonomy.py             # Indian + Western zero-shot anchor definitions
│   └── recommender.py          # Context + anti-repeat decay scoring math
├── api/
│   └── server.py               # Local FastAPI server for the PWA / UI
└── tests/
    └── gate1_eval.py           # CLI validation harness (precision & latency)
```

---

## 6. Environment Setup Commands

### Step 1: Initialize Repository with `uv`
```bash
mkdir smart-wardrobe && cd smart-wardrobe
uv init --python 3.12
uv add torch torchvision transformers sentence-transformers pillow pillow-heif exifread sqlite-vec mediapipe fastapi uvicorn numpy
mkdir -p data/raw_gallery data/crops engine api tests
touch engine/__init__.py
```

### Step 2: Cache Model Weights for 100% Offline Execution
```bash
uv run huggingface-cli login
# After first download of google/embeddinggemma-2, lock offline mode:
export HF_HUB_OFFLINE=1
```

---

## 7. Core Foundation Code

### A. Database Layer (`engine/db.py`)
```python
import sqlite3
import sqlite_vec
import struct
from pathlib import Path

DB_PATH = Path("data/wardrobe.db")

def serialize_f32(vector: list[float]) -> bytes:
    """Packs a float list into raw bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def init_db():
    conn = get_db()
    with conn:
        # 1. Relational table for Proven Outfits & EXIF context
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outfits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_image_path TEXT UNIQUE,
                crop_image_path TEXT,
                primary_category TEXT,      -- e.g., 'Navy Linen Kurta', 'Smart Casual Shirt+Chinos'
                vibe_tag TEXT,              -- 'Work', 'Casual', 'Festive', 'Travel'
                season_tag TEXT,            -- 'Summer/Humid', 'Monsoon', 'Mild/Winter'
                wear_count INTEGER DEFAULT 1,
                last_worn_timestamp TEXT,   -- ISO-8601 from EXIF
                lat REAL,
                lon REAL
            )
        """)

        # 2. Virtual Vector Table (256d Matryoshka truncated for speed)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_outfits USING vec0(
                outfit_id INTEGER PRIMARY KEY,
                embedding float[256] distance_metric=cosine
            )
        """)

        # 3. Wear Log (powers the Anti-Repeat Radar)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wear_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                outfit_id INTEGER,
                worn_at TEXT,
                location_cluster TEXT,
                FOREIGN KEY(outfit_id) REFERENCES outfits(id)
            )
        """)
    conn.close()
    print(f"Initialized local vector DB at {DB_PATH.resolve()}")

if __name__ == "__main__":
    init_db()
```

### B. EmbeddingGemma 2 Wrapper (`engine/embedder.py`)
```python
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

class WardrobeEmbedder:
    def __init__(self, model_id: str = "google/embeddinggemma-2", dim: int = 256):
        self.dim = dim
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Loading {model_id} on {self.device} (target dim={self.dim})...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        ).to(self.device).eval()

    def _truncate_and_norm(self, embeds: torch.Tensor) -> list[list[float]]:
        # Slice to first 256 Matryoshka dimensions and L2-normalize
        sliced = embeds[:, :self.dim].float()
        normed = F.normalize(sliced, p=2, dim=-1)
        return normed.cpu().tolist()

    @torch.inference_mode()
    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return self._truncate_and_norm(outputs)

    @torch.inference_mode()
    def embed_texts(self, prompts: list[str]) -> list[list[float]]:
        # Task prefix aligns text queries with visual garment representations
        formatted = [f"task: garment classification | query: {p}" for p in prompts]
        inputs = self.processor(text=formatted, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        return self._truncate_and_norm(outputs)
```

---

## 8. Gate 1 Implementation Checklist

- [ ] Initialize `smart-wardrobe` repo with `uv` and Python 3.12
- [ ] Run `uv run python engine/db.py` to create `data/wardrobe.db` with `sqlite-vec`
- [ ] Place 1 reference selfie at `data/reference_selfie.jpg` and 20–50 gallery photos in `data/raw_gallery/`
- [ ] Implement `engine/taxonomy.py` to pre-compute `data/taxonomy_anchors.bin`
- [ ] Implement `engine/ingest.py` (EXIF parser + person cropper + cosine deduplication)
- [ ] Run `tests/gate1_eval.py` and verify $>85\%$ classification and deduplication accuracy