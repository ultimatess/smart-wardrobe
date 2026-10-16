# Smart Wardrobe — browser build (GitHub Pages)

The same pipeline as the Python engine, running **entirely in the browser**. No server, and photos
never leave the device.

| Python build | Browser build |
| :--- | :--- |
| EmbeddingGemma 2 (PyTorch, fp32) | `onnx-community/embeddinggemma-2-ONNX`, q4, transformers.js on WebGPU (WASM fallback), in a Web Worker |
| MediaPipe Python | MediaPipe Tasks Web (face detector, pose landmarker, image embedder) |
| SQLite + sqlite-vec | IndexedDB, brute-force cosine in JS (instant for thousands of outfits) |
| exifread | exifr |
| FastAPI | `src/app-api.js` (same operations, called directly by the UI) |
| `taxonomy_anchors.bin` | Anchor rows exported from `engine/taxonomy.py`, embedded in-browser once and cached |

## Develop

```bash
cd web-static
npm ci                 # .npmrc pins the public npm registry
npm run dev            # http://localhost:5173
npm test               # logic tests (vitest + fake-indexeddb)
npm run gate3          # Gate 3 backtest with the q4 model in Node (same scenarios as tests/gate3_backtest.py)
npm run build && npm run e2e -- <photo-dir> [--selfie face.jpg]   # headless Chrome smoke test
```

If you change the taxonomy in Python, re-export it with `uv run python -m engine.export_web` (from the
repo root). Browsers rebuild their cached anchor bank automatically when the taxonomy hash changes.

## Deploy to GitHub Pages

`.github/workflows/pages.yml` runs the tests, builds, and deploys `web-static/dist` on every push to
`main`. One-time setup: in the repo go to **Settings → Pages → Build and deployment → Source**, and choose
**GitHub Actions**. The site is served from `https://<user>.github.io/<repo>/`. The build uses
relative paths, so any sub-path works.

## What leaves the device

- **Once:** about 316 MB of model weights from the Hugging Face CDN (then cached; measured repeat load is about 2 s with 0 bytes over the network).
- **Only if you press "Auto weather" and confirm:** a location rounded to 1 decimal (~11 km), sent to open-meteo.com.
- **Never:** photos, crops, vectors, wear history or the selfie. These live in IndexedDB. Use **Export backup** before clearing site data or switching devices.

## Measured (M1 Pro, Chrome, WebGPU)

| | |
| :--- | :--- |
| First visit model download + init | ~50 s |
| Repeat visit model ready | ~2 s |
| Per photo: locate / embed / classify | ~60 ms / ~1.4 s / <1 ms |
| q4 vs Python fp32 embedding parity | cosine 0.989 (text), 0.991 (image) |
| Gate 3 backtest (q4) | PASS 5/5, same top-3 as Python |
| Face identity (same person / stranger) | 0.94 / 0.33–0.37, identical to Python |

## Limitations

- **Duplicate threshold margin:** the q4 model scores duplicates slightly lower (an edited copy matched at 0.928 vs 0.947 in fp32), close to the 0.92 dedup threshold. Calibrate with Gate 1 on real photos.
- **HEIC** decodes only in Safari. In Chrome or Firefox, export as JPEG first.
- **No background scanning:** the browser can only process photos you pick. Background camera-roll scanning needs the native (LiteRT) app.
- **WebGPU** is needed for good speed. On WASM, image embedding takes several seconds per photo.
