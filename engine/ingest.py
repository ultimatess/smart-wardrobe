"""Passive camera-roll ingestion: EXIF -> face anchoring -> body crop -> embed -> dedup -> tag.

Usage:  uv run python -m engine.ingest [--gallery DIR] [--reset]
"""
import argparse
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from engine import config, db
from engine.context import assign_location_cluster, parse_exif, season_for
from engine.embedder import get_embedder, load_image
from engine.taxonomy import derive_tags, get_anchor_bank

MAX_SIDE = 1600              # downscale before detection/cropping
ACCEPT_FACELESS_SINGLE = True  # mirror selfies: no visible face but exactly one person -> user


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def clip(self, w: int, h: int) -> "Box":
        return Box(max(0, int(self.x0)), max(0, int(self.y0)), min(w, int(self.x1)), min(h, int(self.y1)))

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_tuple(self):
        return (self.x0, self.y0, self.x1, self.y1)


# ---------------------------------------------------------------- MediaPipe person isolation

class PersonLocator:
    """Face detection + identity match against the reference selfie + pose-based body box."""

    def __init__(self, reference: Path | None = config.REFERENCE_SELFIE):
        import os

        os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe/LiteRT info logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        m = config.MODELS_DIR
        self.face_det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(m / "blaze_face_short_range.tflite")),
            min_detection_confidence=0.5))
        self.face_emb = vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=str(m / "image_embedder.tflite")),
            l2_normalize=True, quantize=False))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(m / "pose_landmarker_lite.task")),
            num_poses=6, min_pose_detection_confidence=0.5))
        self.ref_vec = None
        if reference and Path(reference).exists():
            img = _downscale(load_image(reference))
            faces = self._faces(img)
            if not faces:
                raise RuntimeError(f"No face found in reference selfie {reference}")
            self.ref_vec = self._face_vec(img, max(faces, key=lambda b: b.area))

    def _mp_image(self, img: Image.Image):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img)))

    def _faces(self, img: Image.Image) -> list[Box]:
        res = self.face_det.detect(self._mp_image(img))
        out = []
        for d in res.detections:
            bb = d.bounding_box
            if min(bb.width, bb.height) >= config.MIN_FACE_PX:
                out.append(Box(bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height))
        return out

    def _face_vec(self, img: Image.Image, face: Box) -> np.ndarray:
        w, h = face.x1 - face.x0, face.y1 - face.y0
        crop = img.crop(Box(face.x0 - 0.2 * w, face.y0 - 0.2 * h, face.x1 + 0.2 * w, face.y1 + 0.2 * h)
                        .clip(*img.size).as_tuple()).resize((224, 224))
        return np.asarray(self.face_emb.embed(self._mp_image(crop)).embeddings[0].embedding, np.float32)

    def _poses(self, img: Image.Image) -> list[np.ndarray]:
        res = self.pose.detect(self._mp_image(img))
        w, h = img.size
        return [np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in p]) for p in res.pose_landmarks]

    def locate(self, img: Image.Image) -> dict:
        """Returns {'body': Box, 'upper': Box|None, 'lower': Box|None, 'identity': float|None, 'reason': str}
        or {'body': None, 'reason': ...} when the user is not in the photo."""
        w, h = img.size
        faces = self._faces(img)
        poses = self._poses(img)

        face, identity = None, None
        if faces:
            if self.ref_vec is not None:
                scored = [(float(self._face_vec(img, f) @ self.ref_vec), f) for f in faces]
                identity, face = max(scored, key=lambda s: s[0])
                if identity < config.FACE_MATCH_THRESHOLD:
                    return {"body": None, "identity": identity, "reason": "no face matched reference"}
            else:  # single-user mode: largest face is the user
                face = max(faces, key=lambda b: b.area)
        elif not (ACCEPT_FACELESS_SINGLE and len(poses) == 1):
            return {"body": None, "identity": None, "reason": "no face (and not a single-person photo)"}

        # Pick the pose belonging to the anchored face (nose inside the face box).
        pose = None
        if face is not None:
            fw, fh = face.x1 - face.x0, face.y1 - face.y0
            grown = Box(face.x0 - 0.3 * fw, face.y0 - 0.3 * fh, face.x1 + 0.3 * fw, face.y1 + 0.3 * fh)
            pose = next((p for p in poses if grown.contains(p[0, 0], p[0, 1])), None)
        elif poses:
            pose = poses[0]

        if pose is not None:
            reason = "pose" if face is not None else "pose; unanchored single person (no face visible)"
            return {**_boxes_from_pose(pose, face, w, h), "identity": identity, "reason": reason}
        return {**_boxes_from_face(face, w, h), "identity": identity, "reason": "face-heuristic"}


def _boxes_from_pose(p: np.ndarray, face: Box | None, w: int, h: int) -> dict:
    vis = p[:, 2] > 0.3
    sh = p[[11, 12]]          # shoulders
    hips = p[[23, 24]]
    legs = p[[25, 26, 27, 28]]  # knees, ankles
    xs = p[vis, 0] if vis.sum() >= 4 else p[:, 0]
    shoulder_w = abs(sh[0, 0] - sh[1, 0]) or w * 0.2
    top = face.y1 if face else sh[:, 1].min() - 0.25 * shoulder_w
    hip_y = hips[:, 1].mean()
    leg_vis = legs[legs[:, 2] > 0.3]
    bottom = leg_vis[:, 1].max() + 0.1 * shoulder_w if len(leg_vis) else hip_y + 1.2 * (hip_y - top)
    x0, x1 = xs.min() - 0.35 * shoulder_w, xs.max() + 0.35 * shoulder_w
    body = Box(x0, top, x1, bottom).clip(w, h)
    upper = Box(x0, top, x1, hip_y + 0.05 * shoulder_w).clip(w, h)
    lower = Box(x0, hip_y - 0.05 * shoulder_w, x1, bottom).clip(w, h) if hip_y < h * 0.95 else None
    return {"body": body, "upper": upper, "lower": lower}


def _boxes_from_face(face: Box, w: int, h: int) -> dict:
    fw, fh = face.x1 - face.x0, face.y1 - face.y0
    cx = (face.x0 + face.x1) / 2
    body = Box(cx - 1.8 * fw, face.y1, cx + 1.8 * fw, face.y1 + 7.0 * fh).clip(w, h)
    mid = face.y1 + 3.0 * fh
    upper = Box(body.x0, body.y0, body.x1, min(mid, body.y1))
    lower = Box(body.x0, mid, body.x1, body.y1) if mid < body.y1 - fh else None
    return {"body": body, "upper": upper, "lower": lower}


def _downscale(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


# ---------------------------------------------------------------- pipeline

class Ingestor:
    def __init__(self, conn: sqlite3.Connection, embedder=None, locator=None, bank=None):
        self.conn = conn
        self.embedder = embedder or get_embedder()
        self.bank = bank or get_anchor_bank(self.embedder)
        self.locator = locator if locator is not None else PersonLocator()
        self.timings: dict[str, list[float]] = {"locate": [], "embed": [], "classify": [], "total": []}

    def ingest_dir(self, gallery: Path = config.RAW_GALLERY) -> list[dict]:
        paths = [p for p in sorted(Path(gallery).rglob("*")) if p.suffix.lower() in config.IMAGE_EXTS]
        done = {r["path"] for r in self.conn.execute("SELECT path FROM photos")}
        todo = [(parse_exif(p), p) for p in paths if str(p.resolve()) not in done]
        todo.sort(key=lambda t: t[0]["taken_at"])  # chronological, so last_worn ends up correct
        results = []
        for i, (exif, p) in enumerate(todo, 1):
            r = self.ingest_photo(p, exif)
            results.append(r)
            print(f"[{i}/{len(todo)}] {p.name}: {r['status']}"
                  + (f" -> outfit {r['outfit_id']} ({r.get('primary_category', '')})" if r.get("outfit_id") else "")
                  + (f" sim={r['similarity']:.3f}" if r.get("similarity") else "")
                  + (f" [{r['detail']}]" if r.get("detail") else ""))
        return results

    def ingest_photo(self, path: Path, exif: dict | None = None) -> dict:
        t0 = time.perf_counter()
        path = Path(path).resolve()
        exif = exif or parse_exif(path)
        try:
            img = _downscale(load_image(path))
            t = time.perf_counter()
            loc = self.locator.locate(img) if self.locator else {"body": Box(0, 0, *img.size), "upper": None,
                                                                  "lower": None, "identity": None, "reason": "full"}
            self.timings["locate"].append(time.perf_counter() - t)
            if loc.get("identity") is not None:
                loc["reason"] = f"{loc['reason']}; identity={loc['identity']:.2f}"
            if loc["body"] is None or loc["body"].area < 64 * 64:
                return self._record_photo(path, None, "skipped_no_user", None, exif, loc.get("reason"))

            crop = img.crop(loc["body"].as_tuple())
            t = time.perf_counter()
            vec = self.embedder.embed_pil([crop])[0]
            self.timings["embed"].append(time.perf_counter() - t)
        except Exception as e:  # corrupt file, unsupported codec...
            return self._record_photo(path, None, "error", None, exif, f"{type(e).__name__}: {e}")

        with self.conn:
            match = db.nearest_outfits(self.conn, vec, k=1)
            if match and match[0][1] >= config.DEDUP_THRESHOLD:
                outfit_id, sim = match[0]
                self._merge(outfit_id, vec, exif)
                status, row = "merged", {}
            else:
                sim = match[0][1] if match else None
                t = time.perf_counter()
                labels = self.bank.classify(vec, top_k=5)
                tags = derive_tags(labels, self.bank, exif_season=season_for(exif["taken_at"], exif["lat"]))
                self.timings["classify"].append(time.perf_counter() - t)
                crop_path = self._save_crop(crop, path)
                row = {
                    "source_image_path": str(path), "crop_image_path": str(crop_path),
                    "primary_category": tags["primary_category"], "fabric_tag": tags["fabric_tag"],
                    "color_tag": tags["color_tag"], "vibe_tag": tags["vibe_tag"], "season_tag": tags["season_tag"],
                    "labels_json": json.dumps({**labels, "category_raw": tags["category"], "style": tags["style"]}),
                    "wear_count": 1, "last_worn_timestamp": exif["taken_at"], "lat": exif["lat"], "lon": exif["lon"],
                }
                outfit_id = db.insert_outfit(self.conn, row, vec)
                self._log_wear(outfit_id, exif)
                status = "new_outfit"
            for region in ("upper", "lower"):
                b = loc.get(region)
                if b is not None and b.area > 32 * 32:
                    rp = self._save_crop(img.crop(b.as_tuple()), path, suffix=region)
                    self.conn.execute(
                        "INSERT INTO garment_regions (outfit_id, photo_path, region, x0, y0, x1, y1, crop_image_path)"
                        " VALUES (?,?,?,?,?,?,?,?)", (outfit_id, str(path), region, *b.as_tuple(), str(rp)))
            out = self._record_photo(path, outfit_id, status, sim, exif, loc["reason"], commit=False)
        self.timings["total"].append(time.perf_counter() - t0)
        out["primary_category"] = row.get("primary_category") or self.conn.execute(
            "SELECT primary_category FROM outfits WHERE id=?", (outfit_id,)).fetchone()[0]
        return out

    def _merge(self, outfit_id: int, vec: np.ndarray, exif: dict) -> None:
        o = self.conn.execute("SELECT wear_count, last_worn_timestamp FROM outfits WHERE id=?",
                              (outfit_id,)).fetchone()
        n = o["wear_count"]
        # Running centroid keeps the outfit vector robust to lighting/pose variation.
        old = db.get_embedding(self.conn, outfit_id)
        centroid = old * n + vec
        db.update_embedding(self.conn, outfit_id, centroid / np.linalg.norm(centroid))
        # Several photos from the same day are one wear, not several.
        day = exif["taken_at"][:10]
        same_day = self.conn.execute(
            "SELECT 1 FROM wear_logs WHERE outfit_id=? AND substr(worn_at,1,10)=?", (outfit_id, day)).fetchone()
        if same_day:
            return
        last = max(filter(None, [o["last_worn_timestamp"], exif["taken_at"]]))
        self.conn.execute(
            "UPDATE outfits SET wear_count = wear_count + 1, last_worn_timestamp = ?,"
            " lat = COALESCE(?, lat), lon = COALESCE(?, lon) WHERE id = ?",
            (last, exif["lat"] if last == exif["taken_at"] else None,
             exif["lon"] if last == exif["taken_at"] else None, outfit_id))
        self._log_wear(outfit_id, exif)

    def _log_wear(self, outfit_id: int, exif: dict, source: str = "exif") -> None:
        cluster = assign_location_cluster(self.conn, exif["lat"], exif["lon"])
        self.conn.execute(
            "INSERT INTO wear_logs (outfit_id, worn_at, location_cluster, lat, lon, source) VALUES (?,?,?,?,?,?)",
            (outfit_id, exif["taken_at"], cluster, exif["lat"], exif["lon"], source))

    def _save_crop(self, crop: Image.Image, src: Path, suffix: str = "body") -> Path:
        config.CROPS_DIR.mkdir(parents=True, exist_ok=True)
        out = config.CROPS_DIR / f"{src.stem}_{hashlib.sha1(str(src).encode()).hexdigest()[:8]}_{suffix}.jpg"
        crop.save(out, "JPEG", quality=88)
        return out

    def _record_photo(self, path, outfit_id, status, sim, exif, detail, commit=True) -> dict:
        def write():
            self.conn.execute(
                "INSERT OR REPLACE INTO photos (path, outfit_id, status, similarity, taken_at, lat, lon, detail)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (str(path), outfit_id, status, sim, exif["taken_at"], exif["lat"], exif["lon"], detail))
        if commit:
            with self.conn:
                write()
        else:
            write()
        return {"path": str(path), "outfit_id": outfit_id, "status": status, "similarity": sim, "detail": detail}


def reset_db() -> None:
    for p in (config.DB_PATH,):
        p.unlink(missing_ok=True)
    for c in config.CROPS_DIR.glob("*.jpg"):
        c.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", type=Path, default=config.RAW_GALLERY)
    ap.add_argument("--reset", action="store_true", help="wipe wardrobe.db and crops first")
    args = ap.parse_args()
    if args.reset:
        reset_db()
    db.init_db()
    conn = db.get_db()
    ing = Ingestor(conn)
    res = ing.ingest_dir(args.gallery)
    counts = {}
    for r in res:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n_outfits = conn.execute("SELECT COUNT(*) FROM outfits").fetchone()[0]
    print(f"\nProcessed {len(res)} photos: {counts}. Wardrobe now has {n_outfits} outfits.")
    if ing.timings["total"]:
        print("Mean latency (ms): " + ", ".join(
            f"{k}={1000 * np.mean(v):.0f}" for k, v in ing.timings.items() if v))


if __name__ == "__main__":
    main()
