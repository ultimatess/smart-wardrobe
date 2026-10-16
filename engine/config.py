"""Central paths and tunables. Override any path with the matching env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("WARDROBE_DATA_DIR", ROOT / "data"))

DB_PATH = DATA_DIR / "wardrobe.db"
RAW_GALLERY = DATA_DIR / "raw_gallery"
CROPS_DIR = DATA_DIR / "crops"
MODELS_DIR = DATA_DIR / "models"
REFERENCE_SELFIE = DATA_DIR / "reference_selfie.jpg"
TAXONOMY_BIN = DATA_DIR / "taxonomy_anchors.bin"
TAXONOMY_META = DATA_DIR / "taxonomy_anchors.json"

MODEL_ID = os.environ.get("WARDROBE_MODEL_ID", "google/embeddinggemma-2")
EMBED_DIM = 256
# Vision soft tokens per image (70-1120). 280 = model default; 140 is ~2x faster on an M1 with
# near-identical dedup separation. Changing it changes image vectors: re-ingest (--reset) after.
VISION_TOKENS = int(os.environ.get("WARDROBE_VISION_TOKENS", 280))

# Ingestion
DEDUP_THRESHOLD = 0.92        # cosine >= this merges two photos into one outfit
FACE_MATCH_THRESHOLD = 0.80   # identity score >= this treats a face as the user
MIN_FACE_PX = 40              # ignore faces smaller than this (background people)

# Recommendation
ALPHA = 0.7                   # weight of the day-context vector vs. style centroid
DECAY_LAMBDA = 0.15           # per-day recovery rate in (1 - e^{-lambda * dt})
VENUE_REPEAT_DAYS = 45        # same look + same venue inside this window is flagged...
VENUE_RECENT_LOOKS = 5        # ...if it is also among the last N distinct looks that venue has seen
VENUE_REPEAT_PENALTY = 0.5    # multiplier applied when flagged
VIBE_WEIGHT = 0.6             # share of the vibe phrase vs. weather phrase inside E_weather+vibe
VIBE_BOOST = 0.25             # context-fit bonus when the outfit's vibe tag matches the pill
SEASON_BOOST = 0.10           # context-fit bonus when the outfit's season tag matches the weather

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
