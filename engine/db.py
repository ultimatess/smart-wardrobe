"""SQLite + sqlite-vec schema and vector queries (the frozen Gate 4 schema)."""
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import sqlite_vec

from engine import config


def serialize_f32(vector) -> bytes:
    """Packs a float list into raw bytes for sqlite-vec."""
    vector = [float(x) for x in vector]
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_f32(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def get_db(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


SCHEMA = f"""
-- 1. Proven Outfits (whole looks) & EXIF context of the most recent wear
CREATE TABLE IF NOT EXISTS outfits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_image_path TEXT UNIQUE,      -- first photo the outfit was seen in
    crop_image_path TEXT,
    primary_category TEXT,              -- e.g. 'Navy Linen Kurta', 'Shirt + Chinos'
    fabric_tag TEXT,
    color_tag TEXT,
    vibe_tag TEXT,                      -- 'Work', 'Casual', 'Festive', 'Travel'
    season_tag TEXT,                    -- 'Summer/Humid', 'Monsoon', 'Mild/Winter'
    labels_json TEXT,                   -- top-k zero-shot labels + scores per axis
    wear_count INTEGER DEFAULT 1,
    last_worn_timestamp TEXT,           -- ISO-8601 from EXIF
    lat REAL,
    lon REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 2. Vector table (256d Matryoshka truncated)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_outfits USING vec0(
    outfit_id INTEGER PRIMARY KEY,
    embedding float[{config.EMBED_DIM}] distance_metric=cosine
);

-- 3. Wear log (powers the Anti-Repeat Radar)
CREATE TABLE IF NOT EXISTS wear_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outfit_id INTEGER,
    worn_at TEXT,
    location_cluster TEXT,
    lat REAL,
    lon REAL,
    source TEXT DEFAULT 'exif',         -- 'exif' (gallery) | 'manual' (tapped in UI)
    FOREIGN KEY(outfit_id) REFERENCES outfits(id) ON DELETE CASCADE
);

-- 4. Every ingested photo (idempotent re-scans + dedup audit trail)
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE,
    outfit_id INTEGER,                  -- NULL when skipped
    status TEXT,                        -- 'new_outfit' | 'merged' | 'skipped_no_user' | 'error'
    similarity REAL,                    -- cosine to matched outfit when merged
    taken_at TEXT,
    lat REAL,
    lon REAL,
    detail TEXT,
    FOREIGN KEY(outfit_id) REFERENCES outfits(id) ON DELETE SET NULL
);

-- 5. Secondary layer: upper/lower torso boxes for future mix-and-match
CREATE TABLE IF NOT EXISTS garment_regions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outfit_id INTEGER,
    photo_path TEXT,
    region TEXT,                        -- 'upper' | 'lower'
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    crop_image_path TEXT,
    FOREIGN KEY(outfit_id) REFERENCES outfits(id) ON DELETE CASCADE
);

-- 6. User-confirmed venue names (override the inferred Home/Office/Travel guess)
CREATE TABLE IF NOT EXISTS venue_names (
    location_cluster TEXT PRIMARY KEY,
    label TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wear_outfit ON wear_logs(outfit_id, worn_at);
"""


def init_db(path: Path | None = None, verbose: bool = True) -> None:
    conn = get_db(path)
    with conn:
        conn.executescript(SCHEMA)
    conn.close()
    if verbose:
        print(f"Initialized local vector DB at {Path(path or config.DB_PATH).resolve()}")


# ---------------------------------------------------------------- vector ops

def insert_outfit(conn: sqlite3.Connection, row: dict, embedding) -> int:
    cols = ", ".join(row)
    qs = ", ".join("?" for _ in row)
    cur = conn.execute(f"INSERT INTO outfits ({cols}) VALUES ({qs})", tuple(row.values()))
    outfit_id = cur.lastrowid
    conn.execute(
        "INSERT INTO vec_outfits (outfit_id, embedding) VALUES (?, ?)",
        (outfit_id, serialize_f32(embedding)),
    )
    return outfit_id


def update_embedding(conn: sqlite3.Connection, outfit_id: int, embedding) -> None:
    conn.execute(
        "UPDATE vec_outfits SET embedding = ? WHERE outfit_id = ?",
        (serialize_f32(embedding), outfit_id),
    )


def nearest_outfits(conn: sqlite3.Connection, embedding, k: int = 5) -> list[tuple[int, float]]:
    """Returns [(outfit_id, cosine_similarity)] best-first."""
    if conn.execute("SELECT COUNT(*) FROM vec_outfits").fetchone()[0] == 0:
        return []
    rows = conn.execute(
        "SELECT outfit_id, distance FROM vec_outfits WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(embedding), k),
    ).fetchall()
    return [(r["outfit_id"], 1.0 - r["distance"]) for r in rows]


def all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    rows = conn.execute("SELECT outfit_id, embedding FROM vec_outfits ORDER BY outfit_id").fetchall()
    if not rows:
        return [], np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    ids = [r["outfit_id"] for r in rows]
    return ids, np.stack([deserialize_f32(r["embedding"]) for r in rows])


def get_embedding(conn: sqlite3.Connection, outfit_id: int) -> np.ndarray | None:
    r = conn.execute("SELECT embedding FROM vec_outfits WHERE outfit_id = ?", (outfit_id,)).fetchone()
    return deserialize_f32(r["embedding"]) if r else None


def outfit_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["labels"] = json.loads(d.pop("labels_json") or "{}")
    return d


def delete_outfit(conn: sqlite3.Connection, outfit_id: int) -> None:
    conn.execute("DELETE FROM vec_outfits WHERE outfit_id = ?", (outfit_id,))
    conn.execute("DELETE FROM outfits WHERE id = ?", (outfit_id,))


if __name__ == "__main__":
    init_db()
