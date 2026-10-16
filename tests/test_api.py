"""API smoke tests against the temp data dir (fake embedder)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from engine import config, db
from engine.embedder import get_embedder


@pytest.fixture(scope="module")
def client():
    from api.server import app

    with TestClient(app) as c:
        conn = db.get_db()
        emb = get_embedder()
        crop = config.CROPS_DIR / "seed.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 90), (20, 30, 90)).save(crop)
        with conn:
            for name, vibe in [("navy linen kurta", "Work"), ("red silk saree", "Festive")]:
                db.insert_outfit(conn, {"primary_category": name, "vibe_tag": vibe, "season_tag": "Monsoon",
                                        "fabric_tag": "linen", "labels_json": "{}", "crop_image_path": str(crop),
                                        "last_worn_timestamp": "2026-09-01T10:00:00"},
                                 emb.embed_texts([name], task="search")[0])
        conn.close()
        yield c


def test_health_and_list(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["outfits"] == 2 and "Work" in h["vibes"]
    items = client.get("/api/outfits").json()
    assert {o["primary_category"] for o in items} == {"navy linen kurta", "red silk saree"}
    assert "source_image_path" not in items[0] and items[0]["crop_url"].endswith("/crop")
    assert client.get(items[0]["crop_url"]).headers["content-type"] == "image/jpeg"


def test_semantic_search(client):
    hits = client.get("/api/outfits", params={"q": "red silk saree"}).json()
    assert hits[0]["primary_category"] == "red silk saree"


def test_recommend_wear_patch_flow(client):
    rec = client.get("/api/recommend", params={"vibe": "Work", "temp_c": 30, "humidity": 80, "k": 2}).json()
    assert len(rec["results"]) == 2 and "office" in rec["context"]
    oid = rec["results"][0]["outfit_id"]
    after = client.post(f"/api/outfits/{oid}/wear", json={}).json()
    assert after["wear_count"] == 2 and after["wears"][0]["source"] == "manual"
    rec2 = client.get("/api/recommend", params={"vibe": "Work", "k": 2}).json()
    worn = next(r for r in rec2["results"] if r["outfit_id"] == oid)
    assert worn["breakdown"]["recency"] < 0.01
    patched = client.patch(f"/api/outfits/{oid}", json={"vibe_tag": "Travel"}).json()
    assert patched["vibe_tag"] == "Travel"
    assert client.patch(f"/api/outfits/{oid}", json={"vibe_tag": "Party"}).status_code == 422


def test_collections_and_venues(client):
    assert isinstance(client.get("/api/collections").json(), list)
    assert client.get("/api/venues").status_code == 200
    assert client.get("/api/weather").json()["available"] is False  # network off by default


def test_upload_rejects_non_images(client):
    r = client.post("/api/upload", files=[("files", ("evil.sh", io.BytesIO(b"x"), "text/plain"))])
    assert r.status_code == 415


def test_pwa_shell(client):
    assert "Smart Wardrobe" in client.get("/").text
    assert client.get("/static/manifest.webmanifest").status_code == 200
