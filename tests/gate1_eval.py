"""Gate 1 - the 50-photo spike: dedup precision, category accuracy, face anchoring, latency.

    uv run python tests/gate1_eval.py                 # first run writes a labels template
    uv run python tests/gate1_eval.py --labels data/gate1_labels.csv

Labels CSV columns:
    filename   - file in data/raw_gallery
    outfit     - any string; photos of the same outfit share it. '-' = the user is NOT in the photo
    category   - optional taxonomy category label (see data/taxonomy_anchors.json), e.g. 'silk saree'

Pass criteria (spec section 4): > 85 % pairwise precision grouping duplicate outfits across days,
and > 85 % top-1 category match. Runs against a scratch DB; data/wardrobe.db is never touched.
"""
import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import config, db  # noqa: E402
from engine.context import parse_exif  # noqa: E402

PASS = 0.85


class RecordingEmbedder:
    """Wraps the real embedder to keep every per-photo crop vector for threshold sweeps."""

    def __init__(self, inner):
        self.inner, self.last = inner, None

    def embed_pil(self, images):
        self.last = self.inner.embed_pil(images)
        return self.last

    def __getattr__(self, name):
        return getattr(self.inner, name)


def write_template(gallery: Path, labels: Path) -> None:
    files = sorted(p.name for p in gallery.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS)
    labels.parent.mkdir(parents=True, exist_ok=True)
    with open(labels, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "outfit", "category"])
        for f in files:
            w.writerow([f, "", ""])
    print(f"Wrote a labels template for {len(files)} photos to {labels}.\n"
          "Fill in 'outfit' (same value = same outfit, '-' = you are not in it) and optionally 'category',\n"
          "then re-run this script.")


def pairwise(pred: dict[str, int], truth: dict[str, str]) -> tuple[float, float, int, int, int]:
    tp = fp = fn = 0
    for a, b in itertools.combinations(sorted(set(pred) & set(truth)), 2):
        same_pred, same_true = pred[a] == pred[b], truth[a] == truth[b]
        tp += same_pred and same_true
        fp += same_pred and not same_true
        fn += same_true and not same_pred
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall, tp, fp, fn


def sweep(vecs: dict[str, np.ndarray], truth: dict[str, str]) -> list[tuple]:
    names = sorted(set(vecs) & set(truth))
    if len(names) < 2:
        return []
    m = np.stack([vecs[n] for n in names])
    sims = m @ m.T
    rows = []
    for th in np.arange(0.80, 0.985, 0.01):
        tp = fp = fn = 0
        for i, j in itertools.combinations(range(len(names)), 2):
            p, t = sims[i, j] >= th, truth[names[i]] == truth[names[j]]
            tp += p and t
            fp += p and not t
            fn += t and not p
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((round(float(th), 2), prec, rec, f1))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", type=Path, default=config.RAW_GALLERY)
    ap.add_argument("--labels", type=Path, default=config.DATA_DIR / "gate1_labels.csv")
    ap.add_argument("--threshold", type=float, default=config.DEDUP_THRESHOLD)
    args = ap.parse_args()

    if not args.labels.exists():
        write_template(args.gallery, args.labels)
        return 2
    with open(args.labels) as fh:
        labels = [r for r in csv.DictReader(fh) if r.get("filename") and r.get("outfit", "").strip()]
    if not labels:
        print(f"{args.labels} has no labelled rows yet.")
        return 2

    config.DEDUP_THRESHOLD = args.threshold
    scratch = Path(tempfile.mkdtemp(prefix="gate1-"))
    config.CROPS_DIR = scratch / "crops"
    db_path = scratch / "gate1.db"
    db.init_db(db_path, verbose=False)
    conn = db.get_db(db_path)

    from engine.embedder import get_embedder
    from engine.ingest import Ingestor
    from engine.taxonomy import get_anchor_bank

    rec = RecordingEmbedder(get_embedder())
    bank = get_anchor_bank(rec.inner)
    ing = Ingestor(conn, embedder=rec, bank=bank)

    paths = []
    for r in labels:
        p = (args.gallery / r["filename"]).resolve()
        if not p.exists():
            print(f"  ! missing {p}")
            continue
        paths.append((parse_exif(p), p, r))
    paths.sort(key=lambda t: t[0]["taken_at"])

    vecs, pred, results = {}, {}, {}
    for exif, p, r in paths:
        rec.last = None
        res = ing.ingest_photo(p, exif)
        results[r["filename"]] = res
        if rec.last is not None:
            vecs[r["filename"]] = rec.last[0]
        if res["outfit_id"]:
            pred[r["filename"]] = res["outfit_id"]

    present = {r["filename"]: r["outfit"].strip() for r in labels if r["outfit"].strip() != "-"}
    absent = {r["filename"] for r in labels if r["outfit"].strip() == "-"}

    # --- face anchoring
    found = sum(1 for f in present if results.get(f, {}).get("outfit_id"))
    rejected = sum(1 for f in absent if f in results and not results[f]["outfit_id"])
    print("\n== Face anchoring")
    print(f"  user photos kept     {found}/{len(present)}")
    print(f"  non-user rejected    {rejected}/{len(absent)}")
    for f in present:
        if f in results and not results[f]["outfit_id"]:
            print(f"    missed {f}: {results[f]['status']} [{results[f]['detail']}]")
    for f in absent:
        if f in results and results[f]["outfit_id"]:
            print(f"    false accept {f}: [{results[f]['detail']}]")
    pred = {f: o for f, o in pred.items() if f in present}  # dedup is scored on the user's photos only

    # --- dedup
    prec, recall, tp, fp, fn = pairwise(pred, present)
    n_true = len(set(present[f] for f in pred))
    n_pred = len(set(pred.values()))
    print(f"\n== Dedup @ cosine >= {args.threshold:.2f}")
    print(f"  pairwise precision   {prec:.1%}   (tp={tp} fp={fp})")
    print(f"  pairwise recall      {recall:.1%}   (fn={fn})")
    print(f"  outfits predicted    {n_pred}  vs labelled {n_true}")
    rows = sweep(vecs, present)
    if rows:
        best = max(rows, key=lambda r: (r[1] > PASS, r[3]))
        print("  threshold sweep (per-photo vectors):  th  prec  recall  f1")
        for th, p_, r_, f1 in rows:
            mark = " <- best" if th == best[0] else ""
            print(f"      {th:.2f}  {p_:5.1%}  {r_:5.1%}  {f1:.3f}{mark}")

    # --- category
    cat_rows = [(f, r["category"].strip().lower()) for r in labels
                if (f := r["filename"]) in vecs and r.get("category", "").strip()]
    top1 = top2 = 0
    known = {r["label"] for r in bank.rows if r["axis"] == "category"}
    if cat_rows:
        print("\n== Category (zero-shot)")
        for f, want in cat_rows:
            if want not in known:
                print(f"  ! '{want}' is not a taxonomy label (row {f}) - counted as a miss")
            got = [lab for lab, _ in bank.scores(vecs[f], "category")[:2]]
            top1 += got[0] == want
            top2 += want in got
            if got[0] != want:
                print(f"  miss {f}: want '{want}', got {got}")
        print(f"  top-1  {top1}/{len(cat_rows)} = {top1 / len(cat_rows):.1%}")
        print(f"  top-2  {top2}/{len(cat_rows)} = {top2 / len(cat_rows):.1%}")

    # --- latency
    print("\n== Latency (mean ms / photo)")
    for k, v in ing.timings.items():
        if v:
            print(f"  {k:<10} {1000 * np.mean(v):7.1f}")
    cls_ms = 1000 * np.mean(ing.timings["classify"]) if ing.timings["classify"] else 0
    print(f"  zero-shot classify target < 20 ms: {'OK' if cls_ms < 20 else 'SLOW'}")

    dedup_ok = prec > PASS
    cat_ok = (top1 / len(cat_rows) > PASS) if cat_rows else None
    print("\n== GATE 1:", "PASS" if dedup_ok and cat_ok is not False else "FAIL",
          f"(dedup precision {'ok' if dedup_ok else 'below 85%'}; category "
          f"{'not labelled' if cat_ok is None else 'ok' if cat_ok else 'below 85%'})")
    print(f"Scratch artefacts: {scratch}")
    return 0 if dedup_ok and cat_ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
