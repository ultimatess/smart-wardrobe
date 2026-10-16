# Smart Wardrobe (local-first)

Privacy-preserving wardrobe + daily outfit recommender built on **EmbeddingGemma 2** (256d Matryoshka),
MediaPipe and SQLite + `sqlite-vec`. Photos, vectors and wear history never leave the device.
See [SPEC.md](SPEC.md) for the product spec.

There are two builds that share one taxonomy and the same scoring logic:
- **Python (this folder):** FastAPI + local PWA, used for the Gate 1–3 harnesses.
- **Browser ([web-static/](web-static/README.md)):** fully client-side, deployable to GitHub Pages.

## Setup (once, needs network)

```bash
uv sync
uv run python -m engine.bootstrap      # MediaPipe models, EmbeddingGemma 2 weights, 157 taxonomy anchors, DB
export HF_HUB_OFFLINE=1                # everything after this runs offline
```

## Use

```bash
cp ~/Pictures/me.jpg data/reference_selfie.jpg      # 1 clear face photo (optional, enables face anchoring)
cp <20-50 camera-roll photos> data/raw_gallery/
uv run python -m engine.ingest                      # add --reset to rebuild from scratch
uv run uvicorn api.server:app --port 8000           # open http://127.0.0.1:8000
```

The PWA has Today (vibe pills, weather, venue, top-3 with score breakdown, "wearing this today"),
Wardrobe (semantic search, filters, tag correction), Collections, and Import (selfie + photo upload).

## Validation gates

| Gate | Command | Status |
| :--- | :--- | :--- |
| 1. 50-photo spike | `uv run python tests/gate1_eval.py` (first run writes `data/gate1_labels.csv` to fill in) | Harness ready - **needs your labelled photos** |
| 2. Taxonomy calibration | `uv run python tests/gate2_calibrate.py --labels data/gate2_labels.csv` | Harness ready - **needs 10+ labelled outfits** |
| 3. Recommendation backtest | `uv run python tests/gate3_backtest.py` (or `--db data/wardrobe.db`) | **PASS 5/5** on mock bi-cultural wardrobe |
| 4. UI + API | `uv run uvicorn api.server:app` | Built; `uv run pytest` covers API + engine (19 tests) |

Gate 1 also prints a dedup threshold sweep and per-photo face identity scores, so `DEDUP_THRESHOLD`
and `FACE_MATCH_THRESHOLD` in `engine/config.py` can be calibrated from your own photos.

## Layout

```
engine/config.py        paths + every tunable (thresholds, alpha, lambda, radar)
engine/db.py            schema (outfits, vec_outfits, wear_logs, photos, garment_regions, venue_names)
engine/embedder.py      EmbeddingGemma 2 via sentence-transformers (+ FakeEmbedder for tests)
engine/taxonomy.py      157 anchors: 92 Indian/Western/Fusion categories, fabric, colour, pattern, vibe, season
engine/ingest.py        EXIF -> face anchor -> pose body crop (+upper/lower) -> embed -> dedup -> tags
engine/context.py       EXIF/GPS parsing, seasons, venue clusters, opt-in weather
engine/recommender.py   Q_day + decay scoring, context fit, Anti-Repeat Radar
engine/wardrobe_collections.py   rule collections + spherical k-means clusters
api/server.py           FastAPI (127.0.0.1 only); web/ = PWA shell
engine/export_web.py    exports the taxonomy to web-static/src/taxonomy.json
web-static/             browser-only build (see its README); .github/workflows/pages.yml deploys it
tests/                  pytest suite + gate1/2/3 CLIs
```

## Where the build deviates from the spec, and why

1. **Embedder API.** The spec's `AutoModel.get_image_features` sketch doesn't match the released model.
   This build uses `SentenceTransformer.encode` with `{"image": ...}` inputs, `truncate_dim=256`, and
   renormalisation, as the model card describes. Audio encoder isn't loaded (440M instead of 740M).
2. **fp32 on Apple MPS.** On an M1, bf16 measured slower (840 vs 670 ms per image). Use
   `WARDROBE_VISION_TOKENS=140` for ~2x speed with near-identical dedup separation.
3. **Context vector.** One sentence for vibe + weather let weather words dominate ("rain jacket" won
   a Work day). `E_weather+vibe` is now `Norm(0.6*E(vibe) + 0.4*E(weather+day))`.
4. **Context fit multiplier.** Embedding cosines span only ~0.72-0.88, so explicit tag agreement is
   added on top of the spec score: +0.25 vibe match, +0.10 season match, 0.7x for rain/heat-unsafe
   fabrics. Every result returns the full breakdown.
5. **Anti-Repeat Radar.** A pure day window flagged the whole work wardrobe at the office. A look is
   now flagged only if it's among the **last 5 distinct looks at that venue** within 45 days. Home is
   exempt, and users can name venues (`PUT /api/venues/{cluster}`) to override the inferred labels.
6. **Same-day photos are one wear.** Merged duplicates from the same day don't bump `wear_count`.
   Outfit vectors are a running centroid of all their photos.
7. **Extra tables.** `photos` (idempotent rescans + audit), `garment_regions` (upper/lower boxes, the
   split-on-demand layer), and `venue_names`.

## Known limitations / risks to check in Gate 1-2

- **Face anchoring isn't face recognition.** MediaPipe doesn't do identity matching, so this build
  compares MobileNet embeddings of face crops against the selfie. On samples it scored 0.94 for the
  same person vs 0.35-0.39 for a stranger, but calibrate the 0.80 threshold on your photos.
  Single-person photos with no visible face (mirror selfies) are accepted as "unanchored".
- **Fabric is the weakest zero-shot axis** (wool suits read as silk/denim). It only appears in the
  outfit name when confident, and tags are editable in the UI.
- **Western suit vs bandhgala** confusion seen on one sample; Gate 2 will show whether it's systematic.
- Photos without EXIF (e.g. WhatsApp forwards) fall back to file mtime for the wear date.
- Weather is manual by default. `WARDROBE_ALLOW_WEATHER=1` enables Open-Meteo with a 1-decimal
  (~11 km) rounded coordinate, the only network call the app can make.
