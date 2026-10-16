"""One-time setup: fetch MediaPipe task models + EmbeddingGemma 2 weights, build the anchor bank.

    uv run python -m engine.bootstrap

After this completes everything runs offline (export HF_HUB_OFFLINE=1).
"""
import urllib.request

from engine import config

MEDIAPIPE = {
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/"
        "blaze_face_short_range.tflite",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task",
    "image_embedder.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/"
        "mobilenet_v3_small.tflite",
}


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for d in (config.RAW_GALLERY, config.CROPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for name, url in MEDIAPIPE.items():
        dest = config.MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name}")
        urllib.request.urlretrieve(url, dest)

    from huggingface_hub import snapshot_download

    print(f"fetch   {config.MODEL_ID} (~1.5 GB, cached under ~/.cache/huggingface)")
    snapshot_download(config.MODEL_ID)

    from engine import db
    from engine.embedder import get_embedder
    from engine.taxonomy import build_anchor_bank

    db.init_db()
    bank = build_anchor_bank(get_embedder(), force=True)
    print(f"ok      {len(bank)} taxonomy anchors -> {config.TAXONOMY_BIN}")
    print("\nDone. From now on: export HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
