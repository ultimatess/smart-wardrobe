"""Gate 2 - taxonomy & cold-start calibration of the 150+ label anchor bank.

    uv run python tests/gate2_calibrate.py --labels data/gate2_labels.csv

CSV columns: filename (relative to --gallery), category (a taxonomy category label).
Benchmarks each EmbeddingGemma 2 task prefix for the anchor texts and reports zero-shot
top-1 / top-2 agreement with human tags. Pass: top-1 or top-2 match on >= 9/10 items.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import config  # noqa: E402
from engine.embedder import TASK_PREFIXES, get_embedder, load_image  # noqa: E402
from engine.taxonomy import anchor_rows  # noqa: E402

PASS = 0.9


def category_matrix(embedder, task: str) -> tuple[list[str], np.ndarray]:
    rows = [r for r in anchor_rows() if r["axis"] == "category"]
    mats = []
    for r in rows:
        v = embedder.embed_texts(r["prompts"], task=task).mean(axis=0)
        mats.append(v / np.linalg.norm(v))
    return [r["label"] for r in rows], np.stack(mats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", type=Path, default=config.RAW_GALLERY)
    ap.add_argument("--labels", type=Path, default=config.DATA_DIR / "gate2_labels.csv")
    ap.add_argument("--no-crop", action="store_true", help="embed full photos instead of person crops")
    args = ap.parse_args()
    if not args.labels.exists():
        print(f"Missing {args.labels}. Create it with columns: filename,category")
        return 2
    with open(args.labels) as fh:
        items = [(r["filename"], r["category"].strip().lower()) for r in csv.DictReader(fh) if r.get("category")]

    emb = get_embedder()
    locator = None
    if not args.no_crop:
        from engine.ingest import PersonLocator, _downscale

        locator = PersonLocator()
    images, kept = [], []
    for f, want in items:
        img = load_image(args.gallery / f)
        if locator:
            img = _downscale(img)
            loc = locator.locate(img)
            if loc["body"] is None:
                print(f"  ! {f}: user not located ({loc['reason']}) - using full photo")
            else:
                img = img.crop(loc["body"].as_tuple())
        images.append(img)
        kept.append((f, want))
    vecs = emb.embed_pil(images)

    results = {}
    for task in ("classification", "clustering", "similarity", "none"):
        labels, mat = category_matrix(emb, task)
        unknown = {w for _, w in kept if w not in labels}
        if unknown:
            print(f"  ! not taxonomy labels (always a miss): {sorted(unknown)}")
        sims = vecs @ mat.T
        top = np.argsort(-sims, axis=1)[:, :2]
        t1 = sum(labels[top[i, 0]] == w for i, (_, w) in enumerate(kept))
        t2 = sum(w in (labels[top[i, 0]], labels[top[i, 1]]) for i, (_, w) in enumerate(kept))
        results[task] = (t1, t2, top, labels)
        print(f"prefix={task!r:<17} ({TASK_PREFIXES[task]!r})  top-1 {t1}/{len(kept)}  top-1|2 {t2}/{len(kept)}")

    best = max(results, key=lambda k: (results[k][1], results[k][0]))
    t1, t2, top, labels = results[best]
    print(f"\nBest prefix: {best}. Misses:")
    for i, (f, w) in enumerate(kept):
        got = [labels[top[i, 0]], labels[top[i, 1]]]
        if w not in got:
            print(f"  {f}: want '{w}', got {got}")
    ok = t2 / len(kept) >= PASS
    print(f"\n== GATE 2: {'PASS' if ok else 'FAIL'} ({t2}/{len(kept)} top-1|2 with '{best}' prefix; need >= 90%)")
    if best != "classification":
        print("   Note: production anchors use the 'classification' prefix; switch in engine/taxonomy.py if this holds.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
