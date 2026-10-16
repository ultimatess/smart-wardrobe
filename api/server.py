"""Local FastAPI server for the Smart Wardrobe PWA. Binds to 127.0.0.1 only.

Run:  uv run uvicorn api.server:app --port 8000      (or: uv run wardrobe-serve)
"""
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import config, db
from engine.wardrobe_collections import build_collections
from engine.context import fetch_weather, venue_labels
from engine.recommender import DayContext, Recommender
from engine.taxonomy import SEASONS, VIBES

WEB_DIR = config.ROOT / "web"
MODEL_LOCK = threading.Lock()  # one inference at a time on the shared MPS/CPU model

_state: dict = {}


@asynccontextmanager
async def lifespan(_app):
    db.init_db(verbose=False)
    yield


app = FastAPI(title="Smart Wardrobe (local)", version="0.1.0", lifespan=lifespan)


def get_conn():
    conn = db.get_db()
    try:
        yield conn
    finally:
        conn.close()


def _embedder():
    from engine.embedder import get_embedder

    return get_embedder()


def _ingestor(conn):
    from engine.ingest import Ingestor, PersonLocator

    if "locator" not in _state:
        _state["locator"] = PersonLocator()
    return Ingestor(conn, embedder=_embedder(), locator=_state["locator"])


def _public(o: dict) -> dict:
    o = dict(o)
    for k in ("crop_image_path", "source_image_path"):
        o.pop(k, None)
    o["crop_url"] = f"/api/outfits/{o.get('id') or o.get('outfit_id')}/crop"
    return o


# ---------------------------------------------------------------- outfits

@app.get("/api/health")
def health(conn=Depends(get_conn)):
    n = conn.execute("SELECT COUNT(*) FROM outfits").fetchone()[0]
    return {"ok": True, "outfits": n, "model_loaded": "locator" in _state or _embedder_loaded(),
            "reference_selfie": config.REFERENCE_SELFIE.exists(), "vibes": VIBES, "seasons": SEASONS}


def _embedder_loaded() -> bool:
    from engine import embedder

    return embedder._EMBEDDER is not None


@app.get("/api/outfits")
def list_outfits(vibe: str | None = None, season: str | None = None, q: str | None = None,
                 limit: int = 200, conn=Depends(get_conn)):
    where, params = [], []
    if vibe:
        where.append("vibe_tag = ?")
        params.append(vibe)
    if season:
        where.append("season_tag = ?")
        params.append(season)
    sql = "SELECT * FROM outfits" + (" WHERE " + " AND ".join(where) if where else "")
    rows = {r["id"]: db.outfit_dict(r) for r in conn.execute(sql + " ORDER BY last_worn_timestamp DESC", params)}
    if q:  # semantic text search over outfit images
        with MODEL_LOCK:
            qv = _embedder().embed_texts([q], task="search")[0]
        hits = db.nearest_outfits(conn, qv, k=min(limit, 4096))
        return [{**_public(rows[i]), "match": round(s, 4)} for i, s in hits if i in rows][:limit]
    return [_public(o) for o in list(rows.values())[:limit]]


@app.get("/api/outfits/{outfit_id}")
def get_outfit(outfit_id: int, conn=Depends(get_conn)):
    r = conn.execute("SELECT * FROM outfits WHERE id = ?", (outfit_id,)).fetchone()
    if not r:
        raise HTTPException(404, "outfit not found")
    venues = venue_labels(conn)
    wears = [dict(w) | {"venue": venues.get(w["location_cluster"])} for w in conn.execute(
        "SELECT worn_at, location_cluster, source FROM wear_logs WHERE outfit_id=? ORDER BY worn_at DESC",
        (outfit_id,))]
    photos = [dict(p) for p in conn.execute(
        "SELECT taken_at, status, similarity FROM photos WHERE outfit_id=? ORDER BY taken_at", (outfit_id,))]
    regions = [dict(g) for g in conn.execute(
        "SELECT id, region, x0, y0, x1, y1 FROM garment_regions WHERE outfit_id=?", (outfit_id,))]
    return _public(db.outfit_dict(r)) | {"wears": wears, "photos": photos, "regions": regions}


@app.get("/api/outfits/{outfit_id}/crop")
def outfit_crop(outfit_id: int, conn=Depends(get_conn)):
    r = conn.execute("SELECT crop_image_path FROM outfits WHERE id = ?", (outfit_id,)).fetchone()
    if not r or not r[0] or not Path(r[0]).exists():
        raise HTTPException(404, "crop not found")
    return FileResponse(r[0], media_type="image/jpeg")


class OutfitPatch(BaseModel):
    primary_category: str | None = None
    vibe_tag: Literal["Work", "Casual", "Festive", "Travel"] | None = None
    season_tag: Literal["Summer/Humid", "Monsoon", "Mild/Winter"] | None = None
    fabric_tag: str | None = None
    color_tag: str | None = None


@app.patch("/api/outfits/{outfit_id}")
def patch_outfit(outfit_id: int, body: OutfitPatch, conn=Depends(get_conn)):
    changes = body.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(400, "nothing to update")
    with conn:
        cur = conn.execute(f"UPDATE outfits SET {', '.join(f'{k} = ?' for k in changes)} WHERE id = ?",
                           (*changes.values(), outfit_id))
    if not cur.rowcount:
        raise HTTPException(404, "outfit not found")
    return get_outfit(outfit_id, conn)


@app.delete("/api/outfits/{outfit_id}")
def remove_outfit(outfit_id: int, conn=Depends(get_conn)):
    with conn:
        db.delete_outfit(conn, outfit_id)
    return {"deleted": outfit_id}


class WearBody(BaseModel):
    when: datetime | None = None
    venue_cluster: str | None = None


@app.post("/api/outfits/{outfit_id}/wear")
def log_wear(outfit_id: int, body: WearBody | None = None, conn=Depends(get_conn)):
    body = body or WearBody()
    when = (body.when or datetime.now()).replace(microsecond=0).isoformat()
    with conn:
        cur = conn.execute(
            "UPDATE outfits SET wear_count = wear_count + 1,"
            " last_worn_timestamp = MAX(COALESCE(last_worn_timestamp, ''), ?) WHERE id = ?", (when, outfit_id))
        if not cur.rowcount:
            raise HTTPException(404, "outfit not found")
        conn.execute("INSERT INTO wear_logs (outfit_id, worn_at, location_cluster, source) VALUES (?,?,?, 'manual')",
                     (outfit_id, when, body.venue_cluster))
    return get_outfit(outfit_id, conn)


# ---------------------------------------------------------------- ingestion

@app.post("/api/ingest")
def ingest(conn=Depends(get_conn)):
    """Scan data/raw_gallery for photos not yet ingested."""
    with MODEL_LOCK:
        ing = _ingestor(conn)
        results = ing.ingest_dir(config.RAW_GALLERY)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"processed": len(results), "counts": counts,
            "results": [{k: v for k, v in r.items() if k != "path"} | {"file": Path(r["path"]).name}
                        for r in results]}


def _safe_name(name: str) -> str:
    base = Path(name or "upload.jpg").name
    if Path(base).suffix.lower() not in config.IMAGE_EXTS:
        raise HTTPException(415, f"unsupported file type: {base}")
    return base


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...), conn=Depends(get_conn)):
    config.RAW_GALLERY.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = config.RAW_GALLERY / _safe_name(f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
    return ingest(conn)


@app.post("/api/reference-selfie")
def reference_selfie(file: UploadFile = File(...)):
    _safe_name(file.filename)
    from engine.embedder import load_image

    tmp = config.REFERENCE_SELFIE.with_suffix(".upload")
    with open(tmp, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        load_image(tmp).save(config.REFERENCE_SELFIE, "JPEG", quality=92)
    finally:
        tmp.unlink(missing_ok=True)
    _state.pop("locator", None)  # re-anchor on next ingest
    return {"ok": True}


# ---------------------------------------------------------------- daily context

@app.get("/api/recommend")
def recommend(vibe: Literal["Work", "Casual", "Festive", "Travel"] = "Casual",
              temp_c: float | None = None, humidity: float | None = None, rain: bool = False,
              venue: str | None = None, when: datetime | None = None, k: int = 3, conn=Depends(get_conn)):
    ctx = DayContext(vibe=vibe, when=when or datetime.now(), temp_c=temp_c, humidity=humidity, rain=rain,
                     venue_cluster=venue)
    with MODEL_LOCK:
        res = Recommender(conn, embedder=_embedder()).recommend(ctx, top_k=k)
    res["results"] = [_public(r) for r in res["results"]]
    return res


@app.get("/api/weather")
def weather(lat: float | None = None, lon: float | None = None, conn=Depends(get_conn)):
    if lat is None or lon is None:  # default to the most recent geotagged wear (usually home)
        r = conn.execute("SELECT lat, lon FROM wear_logs WHERE lat IS NOT NULL ORDER BY worn_at DESC LIMIT 1").fetchone()
        if r:
            lat, lon = r["lat"], r["lon"]
    w = fetch_weather(lat, lon) if lat is not None else None
    if w is None:
        return {"available": False,
                "hint": "Set WARDROBE_ALLOW_WEATHER=1 to fetch coarse weather, or enter it manually."}
    return {"available": True, **w}


@app.get("/api/venues")
def venues(conn=Depends(get_conn)):
    labels = venue_labels(conn)
    rows = conn.execute("SELECT location_cluster c, COUNT(*) n, MAX(worn_at) last FROM wear_logs "
                        "WHERE location_cluster IS NOT NULL GROUP BY c ORDER BY n DESC").fetchall()
    return [{"cluster": r["c"], "label": labels.get(r["c"], r["c"]), "wears": r["n"], "last": r["last"]}
            for r in rows]


class VenueName(BaseModel):
    label: str


@app.put("/api/venues/{cluster}")
def name_venue(cluster: str, body: VenueName, conn=Depends(get_conn)):
    label = body.label.strip()[:40]
    if not label:
        raise HTTPException(400, "label required")
    with conn:
        conn.execute("INSERT INTO venue_names (location_cluster, label) VALUES (?, ?)"
                     " ON CONFLICT(location_cluster) DO UPDATE SET label = excluded.label", (cluster, label))
    return venues(conn)


@app.get("/api/collections")
def collections(conn=Depends(get_conn)):
    with MODEL_LOCK:
        cols = build_collections(conn)
    return cols


@app.get("/api/stats")
def stats(conn=Depends(get_conn)):
    photos = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM photos GROUP BY status")}
    by_vibe = {r["vibe_tag"]: r["n"] for r in conn.execute("SELECT vibe_tag, COUNT(*) n FROM outfits GROUP BY 1")}
    return {"photos": photos, "outfits_by_vibe": by_vibe,
            "total_wears": conn.execute("SELECT COUNT(*) FROM wear_logs").fetchone()[0]}


# ---------------------------------------------------------------- PWA shell

@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/sw.js")
def service_worker():
    return FileResponse(WEB_DIR / "sw.js", media_type="application/javascript")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
