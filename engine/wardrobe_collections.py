"""Dynamic collections: SQLite relational rules + K-Means style clusters over outfit vectors."""
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from engine import db
from engine.taxonomy import get_anchor_bank

ETHNIC_STYLES = ("indian", "fusion")

RULES = [
    ("Monsoon Office Wear", "season_tag = 'Monsoon' AND vibe_tag = 'Work'", ()),
    ("Summer Breathables", "season_tag = 'Summer/Humid' AND fabric_tag IN ('linen','cotton','khadi','chiffon',"
                           "'rayon','chikankari embroidered cotton')", ()),
    ("Ethnic & Festive", "json_extract(labels_json,'$.style') IN (?, ?) OR vibe_tag = 'Festive'", ETHNIC_STYLES),
    ("Western Workwear", "json_extract(labels_json,'$.style') = 'western' AND vibe_tag = 'Work'", ()),
    ("Weekend Casuals", "vibe_tag = 'Casual'", ()),
    ("Travel Ready", "vibe_tag = 'Travel'", ()),
    ("Most Loved", "wear_count >= 3", ()),
]


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means (cosine) with k-means++ init. Returns (labels, unit centroids)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = max(1, min(k, n))
    cent = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = 1 - np.max(x @ np.stack(cent).T, axis=1)
        p = np.clip(d, 0, None) ** 2
        cent.append(x[rng.choice(n, p=p / p.sum())] if p.sum() > 0 else x[rng.integers(n)])
    c = np.stack(cent)
    labels = np.zeros(n, dtype=int)
    for it in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if it and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                m = members.sum(axis=0)
                c[j] = m / np.linalg.norm(m)
    return labels, c


def build_collections(conn: sqlite3.Connection, now: datetime | None = None, k: int | None = None) -> list[dict]:
    now = now or datetime.now()
    out = []
    for name, where, params in RULES:
        ids = [r[0] for r in conn.execute(f"SELECT id FROM outfits WHERE {where} ORDER BY wear_count DESC", params)]
        if ids:
            out.append({"name": name, "kind": "rule", "outfit_ids": ids})

    cutoff = (now - timedelta(days=60)).isoformat()
    gems = [r[0] for r in conn.execute(
        "SELECT id FROM outfits WHERE wear_count <= 2 AND last_worn_timestamp < ? ORDER BY last_worn_timestamp",
        (cutoff,))]
    if gems:
        out.append({"name": "Seldom Worn Gems", "kind": "rule", "outfit_ids": gems})

    ids, vecs = db.all_embeddings(conn)
    if len(ids) >= 4:
        k = k or max(2, min(8, round(np.sqrt(len(ids) / 2))))
        labels, cents = kmeans(vecs, k)
        bank = get_anchor_bank()
        for j in range(len(cents)):
            members = [ids[i] for i in np.where(labels == j)[0]]
            if len(members) < 2:
                continue
            cat = bank.scores(cents[j], "category")[0][0]
            vibe = bank.scores(cents[j], "vibe")[0][0]
            out.append({"name": f"{vibe}: {cat.title()} & similar", "kind": "cluster", "outfit_ids": members})
    return out
