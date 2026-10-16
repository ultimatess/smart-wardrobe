// Port of engine/ingest.py Ingestor for the browser: File -> EXIF -> locate -> crop -> embed -> dedup -> tag.
import exifr from "exifr";
import { DEDUP_THRESHOLD, MAX_SIDE } from "./config.js";
import { assignLocationCluster, seasonFor } from "./context.js";
import { embedImageData } from "./embedder.js";
import { area, makeCanvas } from "./locator.js";
import * as store from "./store.js";
import { deriveTags } from "./taxonomy.js";
import { dot, l2, toLocalISO } from "./vec.js";

export const photoKey = (f) => `${f.name}|${f.size}|${f.lastModified}`;

export async function readExif(file) {
  let tags = null;
  try { tags = await exifr.parse(file, { gps: true, pick: ["DateTimeOriginal", "CreateDate", "ModifyDate", "latitude", "longitude"] }); }
  catch { /* no EXIF */ }
  const dt = tags?.DateTimeOriginal || tags?.CreateDate || tags?.ModifyDate;
  const ok = dt instanceof Date && !isNaN(dt);
  const lat = Number.isFinite(tags?.latitude) ? +tags.latitude.toFixed(6) : null;
  const lon = Number.isFinite(tags?.longitude) ? +tags.longitude.toFixed(6) : null;
  return {
    taken_at: toLocalISO(ok ? dt : new Date(file.lastModified)),
    time_source: ok ? "exif" : "mtime",
    lat: lat === 0 && lon === 0 ? null : lat, lon: lat === 0 && lon === 0 ? null : lon,
  };
}

/** Decode (EXIF-rotated) and downscale to MAX_SIDE onto a canvas. */
export async function decode(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const c = makeCanvas(Math.round(bmp.width * s), Math.round(bmp.height * s));
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return c;
}

export function cropCanvas(src, b) {
  const c = makeCanvas(b.x1 - b.x0, b.y1 - b.y0);
  c.getContext("2d").drawImage(src, b.x0, b.y0, c.width, c.height, 0, 0, c.width, c.height);
  return c;
}
// Stored as raw bytes: WebKit (Safari and every iOS browser) can't put Blobs in IndexedDB.
const toJpeg = (c) => new Promise((res) => c.toBlob(res, "image/jpeg", 0.88)).then((b) => b.arrayBuffer());
const imageData = (c) => c.getContext("2d").getImageData(0, 0, c.width, c.height);

export class Ingestor {
  constructor({ locator, bank }) {
    Object.assign(this, { locator, bank, timings: { locate: [], embed: [], classify: [] } });
  }

  /** files: File[]; onEach({i, n, file, result}) */
  async ingestFiles(files, onEach) {
    const done = new Set((await store.all("photos")).map((p) => p.key));
    const todo = [];
    for (const f of files) if (!done.has(photoKey(f))) todo.push([await readExif(f), f]);
    todo.sort((a, b) => (a[0].taken_at < b[0].taken_at ? -1 : 1));
    const out = [];
    for (let i = 0; i < todo.length; i++) {
      const [exif, file] = todo[i];
      const result = await this.ingestPhoto(file, exif);
      out.push(result);
      onEach?.({ i: i + 1, n: todo.length, file, result });
    }
    return out;
  }

  async ingestPhoto(file, exif) {
    const key = photoKey(file);
    const record = async (outfit_id, status, similarity, detail) => {
      const p = { key, name: file.name, outfit_id, status, similarity, taken_at: exif.taken_at, lat: exif.lat,
                  lon: exif.lon, detail: [detail, exif.time_source === "mtime" ? "no EXIF date" : null].filter(Boolean).join("; ") };
      await store.put("photos", p);
      return p;
    };
    let canvas, loc, vec;
    try {
      canvas = await decode(file);
      let t = performance.now();
      loc = this.locator.locate(canvas);
      this.timings.locate.push(performance.now() - t);
      if (loc.identity != null) loc.reason = `${loc.reason}; identity=${loc.identity.toFixed(2)}`;
      if (!loc.body || area(loc.body) < 64 * 64) return record(null, "skipped_no_user", null, loc.reason);
      t = performance.now();
      vec = await embedImageData(imageData(cropCanvas(canvas, loc.body)));
      this.timings.embed.push(performance.now() - t);
    } catch (e) {
      return record(null, "error", null, `${e.name}: ${e.message}`);
    }

    const [outfits, wears] = await Promise.all([store.all("outfits"), store.all("wears")]);
    let best = null, sim = null;
    for (const o of outfits) {
      const s = dot(o.embedding, vec);
      if (sim == null || s > sim) { sim = s; best = o; }
    }
    let outfit_id, status;
    if (best && sim >= DEDUP_THRESHOLD) {
      outfit_id = best.id;
      status = "merged";
      await this.merge(best, vec, exif, wears);
    } else {
      const t = performance.now();
      const labels = this.bank.classify(vec, 5);
      const tags = deriveTags(labels, this.bank, seasonFor(exif.taken_at, exif.lat));
      this.timings.classify.push(performance.now() - t);
      const crop = await toJpeg(cropCanvas(canvas, loc.body));
      outfit_id = await store.put("outfits", {
        source_name: file.name, crop, embedding: vec,
        primary_category: tags.primary_category, fabric_tag: tags.fabric_tag, color_tag: tags.color_tag,
        vibe_tag: tags.vibe_tag, season_tag: tags.season_tag,
        labels: { ...labels, category_raw: tags.category, style: tags.style },
        wear_count: 1, last_worn_timestamp: exif.taken_at, lat: exif.lat, lon: exif.lon,
        created_at: toLocalISO(new Date()),
      });
      await this.logWear(outfit_id, exif, wears);
      status = "new_outfit";
    }
    for (const region of ["upper", "lower"]) {
      const b = loc[region];
      if (b && area(b) > 32 * 32)
        await store.put("regions", { outfit_id, photo: file.name, region, box: b, crop: await toJpeg(cropCanvas(canvas, b)) });
    }
    const p = await record(outfit_id, status, sim, loc.reason);
    return { ...p, primary_category: (await store.get("outfits", outfit_id)).primary_category };
  }

  async merge(o, vec, exif, wears) {
    // Running centroid keeps the outfit vector robust to lighting/pose variation.
    const n = o.wear_count;
    const c = l2(o.embedding.map((v, i) => v * n + vec[i]));
    const day = exif.taken_at.slice(0, 10);
    const sameDay = wears.some((w) => w.outfit_id === o.id && w.worn_at.slice(0, 10) === day);
    const next = { ...o, embedding: c };
    if (!sameDay) { // several photos from the same day are one wear
      const isLatest = !o.last_worn_timestamp || exif.taken_at > o.last_worn_timestamp;
      next.wear_count = n + 1;
      if (isLatest) Object.assign(next, { last_worn_timestamp: exif.taken_at, lat: exif.lat ?? o.lat, lon: exif.lon ?? o.lon });
    }
    await store.put("outfits", next);
    if (!sameDay) await this.logWear(o.id, exif, wears);
  }

  async logWear(outfit_id, exif, wears, source = "exif") {
    const location_cluster = assignLocationCluster(wears, exif.lat, exif.lon);
    const w = { outfit_id, worn_at: exif.taken_at, location_cluster, lat: exif.lat, lon: exif.lon, source };
    w.id = await store.put("wears", w);
    wears.push(w);
  }
}
