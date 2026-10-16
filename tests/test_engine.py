"""Unit tests (fake embedder, temp data dir): uv run pytest"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from engine import config, db
from engine.context import haversine_km, season_for, season_from_weather
from engine.embedder import FakeEmbedder
from engine.ingest import Box, Ingestor, _boxes_from_face
from engine.recommender import DayContext, Recommender, recency_factor, style_centroid
from engine.taxonomy import anchor_rows, build_anchor_bank, derive_tags
from engine.wardrobe_collections import build_collections, kmeans


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "w.db"
    db.init_db(path, verbose=False)
    c = db.get_db(path)
    yield c
    c.close()


@pytest.fixture(scope="session")
def emb():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def bank(emb):
    return build_anchor_bank(emb, force=True)


def unit(seed, dim=config.EMBED_DIM):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_taxonomy_has_150_plus_bicultural_anchors():
    rows = anchor_rows()
    assert len(rows) >= 150
    styles = {r.get("style") for r in rows if r["axis"] == "category"}
    assert {"indian", "western", "fusion"} <= styles
    labels = {r["label"] for r in rows}
    for must in ("silk saree", "bandhgala suit", "silk dhoti with kurta", "linen kurta", "shirt and chinos"):
        assert must in labels


def test_anchor_bank_roundtrip_and_cache(emb, bank):
    assert bank.matrix.shape == (len(anchor_rows()), config.EMBED_DIM)
    assert np.allclose(np.linalg.norm(bank.matrix, axis=1), 1, atol=1e-5)
    cached = build_anchor_bank(emb)
    assert np.allclose(cached.matrix, bank.matrix)


def test_derive_tags_uses_category_priors(bank):
    vec = bank.matrix[bank.axes["category"][[r["label"] for r in bank.rows if r["axis"] == "category"].index("sherwani")]]
    tags = derive_tags(bank.classify(vec), bank, exif_season="Mild/Winter")
    assert tags["category"] == "sherwani"
    assert tags["vibe_tag"] == "Festive"
    assert tags["season_tag"] == "Mild/Winter"
    assert "Sherwani" in tags["primary_category"]


def test_vector_knn_returns_cosine(conn):
    a, b = unit(1), unit(2)
    with conn:
        ia = db.insert_outfit(conn, {"primary_category": "A"}, a)
        db.insert_outfit(conn, {"primary_category": "B"}, b)
    (best, sim), *_ = db.nearest_outfits(conn, a, k=2)
    assert best == ia and sim == pytest.approx(1.0, abs=1e-5)


def _photo(tmp_path, name, color, noise_seed=None):
    arr = np.full((400, 260, 3), color, dtype=np.uint8)
    if noise_seed is not None:  # simulate a second photo of the same outfit
        arr = np.clip(arr + np.random.default_rng(noise_seed).integers(-4, 5, arr.shape), 0, 255).astype(np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_ingest_dedup_merges_and_counts_wears(conn, tmp_path, emb, bank):
    ing = Ingestor(conn, embedder=emb, locator=False, bank=bank)
    d0 = datetime(2026, 7, 1, 9)
    ex = lambda d: {"taken_at": d.isoformat(), "lat": 12.97, "lon": 77.59, "time_source": "exif"}
    r1 = ing.ingest_photo(_photo(tmp_path, "a1.jpg", (30, 40, 120)), ex(d0))
    r2 = ing.ingest_photo(_photo(tmp_path, "a2.jpg", (30, 40, 120), 1), ex(d0 + timedelta(days=9)))
    r3 = ing.ingest_photo(_photo(tmp_path, "a3.jpg", (30, 40, 120), 2), ex(d0 + timedelta(days=9, hours=3)))
    r4 = ing.ingest_photo(_photo(tmp_path, "b1.jpg", (200, 30, 30)), ex(d0 + timedelta(days=2)))
    assert r1["status"] == "new_outfit"
    assert r2["status"] == "merged" and r2["outfit_id"] == r1["outfit_id"] and r2["similarity"] >= 0.92
    assert r3["status"] == "merged"
    assert r4["status"] == "new_outfit" and r4["outfit_id"] != r1["outfit_id"]
    o = conn.execute("SELECT * FROM outfits WHERE id=?", (r1["outfit_id"],)).fetchone()
    assert o["wear_count"] == 2  # same-day photo (r3) is not a second wear
    assert o["last_worn_timestamp"] == (d0 + timedelta(days=9)).isoformat()
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 4
    clusters = {r[0] for r in conn.execute("SELECT location_cluster FROM wear_logs")}
    assert clusters == {"loc-1"}


def test_face_heuristic_box_is_below_face():
    boxes = _boxes_from_face(Box(100, 50, 160, 120), 600, 900)
    assert boxes["body"].y0 == 120 and boxes["body"].y1 > 400
    assert boxes["upper"].y1 <= boxes["lower"].y0 + 1


def test_recency_factor():
    assert recency_factor(None) == 1.0
    assert recency_factor(0) == 0.0
    assert recency_factor(1) == pytest.approx(1 - math.exp(-config.DECAY_LAMBDA))
    assert recency_factor(3) < recency_factor(30) < 1


def test_seasons_and_weather():
    assert season_for("2026-04-15T10:00:00") == "Summer/Humid"
    assert season_for("2026-08-01T10:00:00") == "Monsoon"
    assert season_for("2026-12-25T10:00:00") == "Mild/Winter"
    assert season_for("2026-12-25T10:00:00", lat=-33.9) == "Summer/Humid"
    assert season_from_weather(34, 80, False) == "Summer/Humid"
    assert season_from_weather(26, 90, True) == "Monsoon"
    assert season_from_weather(18, 40, False) == "Mild/Winter"
    assert haversine_km(12.97, 77.59, 13.08, 80.27) == pytest.approx(290, rel=0.05)


def _seed_wardrobe(conn, emb):
    items = [
        ("navy linen kurta office", "Work", "Summer/Humid", "linen"),
        ("maroon silk sherwani wedding", "Festive", "Mild/Winter", "silk"),
        ("blue denim jeans t-shirt weekend", "Casual", "Monsoon", "denim"),
        ("grey formal shirt trousers office", "Work", "Monsoon", "cotton"),
    ]
    vecs = emb.embed_texts([d for d, *_ in items], task="search")
    ids = []
    with conn:
        for (d, vibe, season, fabric), v in zip(items, vecs):
            ids.append(db.insert_outfit(conn, {"primary_category": d, "vibe_tag": vibe, "season_tag": season,
                                               "fabric_tag": fabric, "labels_json": "{}"}, v))
    return ids


def test_recommender_anti_repeat_and_venue_radar(conn, emb):
    ids = _seed_wardrobe(conn, emb)
    now = datetime(2026, 8, 5, 9)
    with conn:
        conn.execute("INSERT INTO wear_logs (outfit_id, worn_at, location_cluster) VALUES (?,?,?)",
                     (ids[3], (now - timedelta(days=1)).isoformat(), "loc-1"))
        conn.execute("INSERT INTO wear_logs (outfit_id, worn_at, location_cluster) VALUES (?,?,?)",
                     (ids[0], (now - timedelta(days=20)).isoformat(), "loc-1"))
    rec = Recommender(conn, embedder=emb)
    res = rec.recommend(DayContext(vibe="Work", when=now, temp_c=27, humidity=85, rain=True,
                                   venue_cluster="loc-1"), top_k=4)
    by_id = {r["outfit_id"]: r for r in res["results"]}
    assert by_id[ids[3]]["breakdown"]["recency"] < 0.2          # worn yesterday -> heavily decayed
    assert "worn yesterday" in by_id[ids[3]]["warnings"]
    assert by_id[ids[0]]["breakdown"]["venue_radar"] == config.VENUE_REPEAT_PENALTY
    assert any("risky in rain" in w for w in by_id[ids[1]]["warnings"])
    assert res["results"][0]["outfit_id"] != ids[3]


def test_recommender_ignores_future_wears_for_backtest(conn, emb):
    ids = _seed_wardrobe(conn, emb)
    now = datetime(2026, 8, 5, 9)
    with conn:
        conn.execute("INSERT INTO wear_logs (outfit_id, worn_at) VALUES (?,?)",
                     (ids[0], (now + timedelta(days=1)).isoformat()))
    res = Recommender(conn, embedder=emb).recommend(DayContext(vibe="Work", when=now), top_k=4)
    assert next(r for r in res["results"] if r["outfit_id"] == ids[0])["breakdown"]["recency"] == 1.0


def test_style_centroid_weights_by_wears():
    a, b = unit(3), unit(4)
    c = style_centroid([1, 2], np.stack([a, b]), {1: 20, 2: 1})
    assert c @ a > c @ b


def test_kmeans_separates_clusters():
    rng = np.random.default_rng(0)
    centers = np.stack([unit(10), unit(11)])
    x = np.concatenate([centers[0] + 0.05 * rng.standard_normal((10, config.EMBED_DIM)),
                        centers[1] + 0.05 * rng.standard_normal((10, config.EMBED_DIM))]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels, _ = kmeans(x, 2)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1 and labels[0] != labels[10]


def test_collections_rules(conn, emb, bank, monkeypatch):
    import engine.taxonomy as tx
    monkeypatch.setattr(tx, "_BANK", bank)
    _seed_wardrobe(conn, emb)
    names = {c["name"] for c in build_collections(conn)}
    assert "Monsoon Office Wear" in names
