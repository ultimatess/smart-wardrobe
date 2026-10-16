// Local replacement for api/server.py: same operations, backed by IndexedDB + in-browser models.
import { fetchWeather, venueLabels } from "./context.js";
import { buildCollections } from "./collections.js";
import { embedTexts, initModel, onModelProgress } from "./embedder.js";
import { Ingestor, decode } from "./ingest.js";
import { PersonLocator } from "./locator.js";
import { DayContext, contextVector, recommend as score } from "./recommender.js";
import * as store from "./store.js";
import { AnchorBank, SEASONS, TAXONOMY_HASH, VIBES, buildAnchorMatrix } from "./taxonomy.js";
import { dot, toLocalISO } from "./vec.js";

export { onModelProgress, SEASONS, VIBES };
let bank = null, locator = null, device = null;
const status = new Set();
export const onStatus = (f) => status.add(f);
const say = (m) => status.forEach((f) => f(m));

/** Loads model (download on first visit), anchor bank and MediaPipe. Idempotent. */
export async function ready() {
  if (bank && locator) return { device };
  say("Loading EmbeddingGemma 2 (first visit downloads ~285 MB)…");
  ({ device } = await initModel());
  if (!bank) {
    const cached = await store.get("meta", "anchors");
    if (cached?.hash === TAXONOMY_HASH && cached.device === device) bank = new AnchorBank(cached.matrix);
    else {
      say("Building the 157-label taxonomy (one time)…");
      const matrix = await buildAnchorMatrix(embedTexts, (p) => say(`Building taxonomy… ${Math.round(p * 100)}%`));
      await store.put("meta", { key: "anchors", hash: TAXONOMY_HASH, device, matrix });
      bank = new AnchorBank(matrix);
    }
  }
  if (!locator) {
    say("Loading MediaPipe…");
    locator = await PersonLocator.create();
    locator.refVec = (await store.get("meta", "selfie"))?.vec ?? null;
  }
  say("");
  return { device };
}

// ------------------------------------------------------------------ outfits
export async function health() {
  const [outfits, selfie] = await Promise.all([store.all("outfits"), store.get("meta", "selfie")]);
  return { outfits: outfits.length, reference_selfie: !!selfie, vibes: VIBES, seasons: SEASONS, device };
}

const urls = new Map();
export function cropUrl(o) {
  const k = `${o.id}:${o.crop?.size}`;
  if (!urls.has(k) && o.crop) urls.set(k, URL.createObjectURL(o.crop));
  return urls.get(k) || "";
}
const pub = (o) => ({ ...o, crop_url: cropUrl(o), embedding: undefined, crop: undefined });

export async function listOutfits({ vibe, season, q } = {}) {
  let rows = (await store.all("outfits")).filter((o) => (!vibe || o.vibe_tag === vibe) && (!season || o.season_tag === season));
  if (q) {
    await ready();
    const [qv] = await embedTexts([q], "search");
    return rows.map((o) => ({ ...pub(o), match: dot(qv, o.embedding) })).sort((a, b) => b.match - a.match);
  }
  return rows.sort((a, b) => ((a.last_worn_timestamp || "") < (b.last_worn_timestamp || "") ? 1 : -1)).map(pub);
}

export async function getOutfit(id) {
  const o = await store.get("outfits", id);
  if (!o) throw new Error("outfit not found");
  const [wears, photos, names] = await Promise.all([store.all("wears"), store.all("photos"), store.all("venues")]);
  const venues = venueLabels(wears, Object.fromEntries(names.map((v) => [v.cluster, v.label])));
  return {
    ...pub(o),
    wears: wears.filter((w) => w.outfit_id === id).sort((a, b) => (a.worn_at < b.worn_at ? 1 : -1))
      .map((w) => ({ ...w, venue: venues[w.location_cluster] })),
    photos: photos.filter((p) => p.outfit_id === id),
  };
}

export async function patchOutfit(id, changes) {
  const o = await store.get("outfits", id);
  const allowed = ["primary_category", "vibe_tag", "season_tag", "fabric_tag", "color_tag"];
  for (const k of allowed) if (changes[k] != null) o[k] = changes[k];
  if (!VIBES.includes(o.vibe_tag) || !SEASONS.includes(o.season_tag)) throw new Error("invalid tag");
  await store.put("outfits", o);
  return getOutfit(id);
}

export const removeOutfit = (id) => store.deleteOutfit(id);

export async function logWear(id, { when = new Date(), venue = null } = {}) {
  const o = await store.get("outfits", id);
  const worn_at = toLocalISO(when);
  o.wear_count += 1;
  if (!o.last_worn_timestamp || worn_at > o.last_worn_timestamp) o.last_worn_timestamp = worn_at;
  await store.put("outfits", o);
  await store.put("wears", { outfit_id: id, worn_at, location_cluster: venue, lat: null, lon: null, source: "manual" });
  return getOutfit(id);
}

// ------------------------------------------------------------------ ingest
export async function ingestFiles(files, onEach) {
  await ready();
  const ing = new Ingestor({ locator, bank });
  const results = await ing.ingestFiles([...files], onEach);
  const counts = {};
  results.forEach((r) => (counts[r.status] = (counts[r.status] || 0) + 1));
  return { processed: results.length, counts, results, timings: ing.timings };
}

export async function setSelfie(file) {
  await ready();
  const canvas = await decode(file);
  const vec = locator.selfieVec(canvas);
  await store.put("meta", { key: "selfie", vec });
  locator.refVec = vec;
}

// ------------------------------------------------------------------ daily context
export async function recommendDay({ vibe, temp_c = null, humidity = null, rain = false, venue = null, when = new Date(), k = 3 }) {
  await ready();
  const [outfits, wears, names] = await Promise.all([store.all("outfits"), store.all("wears"), store.all("venues")]);
  const ctx = new DayContext({ vibe, when, temp_c, humidity, rain, venue });
  const res = score(ctx, await contextVector(ctx, embedTexts),
    { outfits, wears, venueNames: Object.fromEntries(names.map((v) => [v.cluster, v.label])) }, k);
  const byId = Object.fromEntries(outfits.map((o) => [o.id, o]));
  res.results = res.results.map((r) => ({ ...r, crop_url: cropUrl(byId[r.outfit_id]) }));
  return res;
}

export async function venues() {
  const [wears, names] = await Promise.all([store.all("wears"), store.all("venues")]);
  const labels = venueLabels(wears, Object.fromEntries(names.map((v) => [v.cluster, v.label])));
  const agg = {};
  for (const w of wears) if (w.location_cluster) {
    const a = (agg[w.location_cluster] ||= { cluster: w.location_cluster, wears: 0, last: "" });
    a.wears++;
    if (w.worn_at > a.last) a.last = w.worn_at;
  }
  return Object.values(agg).map((a) => ({ ...a, label: labels[a.cluster] || a.cluster })).sort((a, b) => b.wears - a.wears);
}

export async function nameVenue(cluster, label) {
  label = label.trim().slice(0, 40);
  if (!label) throw new Error("label required");
  await store.put("venues", { cluster, label });
  return venues();
}

export async function collections() {
  const outfits = await store.all("outfits");
  return buildCollections(outfits, bank);
}

/** Opt-in only: uses browser geolocation (or the latest geotagged wear) rounded to ~11 km. */
export async function weather() {
  let lat, lon;
  try {
    const pos = await new Promise((res, rej) => navigator.geolocation.getCurrentPosition(res, rej, { timeout: 8000, maximumAge: 36e5 }));
    ({ latitude: lat, longitude: lon } = pos.coords);
  } catch {
    const w = (await store.all("wears")).filter((x) => x.lat != null).sort((a, b) => (a.worn_at < b.worn_at ? 1 : -1))[0];
    if (!w) throw new Error("No location available. Allow location access or enter the weather manually.");
    ({ lat, lon } = w);
  }
  return fetchWeather(lat, lon);
}

// ------------------------------------------------------------------ backup (browser storage can be evicted)
const b64 = (blob) => new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(blob); });

export async function exportBackup() {
  const dump = { format: "smart-wardrobe-backup", version: 1, exported_at: new Date().toISOString() };
  for (const s of ["outfits", "wears", "photos", "regions", "venues"]) {
    dump[s] = await Promise.all((await store.all(s)).map(async (r) => ({
      ...r, embedding: r.embedding ? Array.from(r.embedding) : undefined, crop: r.crop ? await b64(r.crop) : undefined,
    })));
  }
  const selfie = await store.get("meta", "selfie");
  if (selfie) dump.selfie = Array.from(selfie.vec);
  return new Blob([JSON.stringify(dump)], { type: "application/json" });
}

export async function importBackup(file) {
  const dump = JSON.parse(await file.text());
  if (dump.format !== "smart-wardrobe-backup") throw new Error("Not a Smart Wardrobe backup file");
  for (const s of ["outfits", "wears", "photos", "regions", "venues"])
    for (const r of dump[s] || []) {
      if (r.embedding) r.embedding = Float32Array.from(r.embedding);
      if (r.crop) r.crop = await (await fetch(r.crop)).blob();
      await store.put(s, r);
    }
  if (dump.selfie) {
    await store.put("meta", { key: "selfie", vec: Float32Array.from(dump.selfie) });
    if (locator) locator.refVec = Float32Array.from(dump.selfie);
  }
}

export async function wipeAll() {
  await store.wipe();
  bank = null;
  if (locator) locator.refVec = null;
}
