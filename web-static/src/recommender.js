// Port of engine/recommender.py (see its docstring for the scoring rationale).
import * as C from "./config.js";
import { seasonFromWeather, venueLabels } from "./context.js";
import { HOT, VIBES } from "./taxonomy.js";
import { DAY_MS, dot, mix, parseLocal, toLocalISO } from "./vec.js";

export const VIBE_PHRASES = {
  Work: "smart professional office workwear",
  Casual: "relaxed casual everyday outfit",
  Festive: "festive traditional celebration outfit for a function",
  Travel: "comfortable practical travel outfit",
};
export const RAIN_UNSAFE = new Set(["silk", "raw silk", "velvet", "brocade", "organza", "leather", "tweed", "wool"]);
export const HEAT_UNSAFE = new Set(["wool", "tweed", "velvet", "leather", "brocade"]);

export class DayContext {
  constructor({ vibe = "Casual", when = new Date(), temp_c = null, humidity = null, rain = false, venue = null } = {}) {
    if (!VIBES.includes(vibe)) throw new Error(`vibe must be one of ${VIBES}`);
    Object.assign(this, { vibe, when, temp_c, humidity, rain, venue });
  }
  get season() { return seasonFromWeather(this.temp_c, this.humidity, this.rain); }
  phrases() {
    const w = [];
    if (this.temp_c != null) {
      const t = this.temp_c;
      const feel = t >= 33 ? "very hot" : t >= 28 ? "hot" : t >= 23 ? "warm" : t >= 16 ? "mild" : "cold";
      w.push(`${feel} ${Math.round(t)} degree`);
    }
    if (this.humidity != null && this.humidity >= 70) w.push("humid");
    if (this.rain) w.push("rainy monsoon");
    const h = this.when.getHours();
    const tod = h < 12 ? "morning" : h < 17 ? "afternoon" : "evening";
    const day = w.length ? `outfit for ${w.join(" ")} weather` : "outfit";
    const weekday = this.when.toLocaleDateString("en-US", { weekday: "long" });
    return [VIBE_PHRASES[this.vibe], `${day} on a ${weekday} ${tod}`];
  }
  describe() { return this.phrases().join(" | "); }
}

const calendarDays = (a, b) =>
  Math.round((new Date(b.getFullYear(), b.getMonth(), b.getDate()) - new Date(a.getFullYear(), a.getMonth(), a.getDate())) / DAY_MS);

export const recencyFactor = (dt, lam = C.DECAY_LAMBDA) => (dt == null ? 1 : 1 - Math.exp(-lam * Math.max(dt, 0)));

export function styleCentroid(outfits, wearCounts) {
  return mix(outfits.map((o) => [Math.log1p(wearCounts[o.id] || 1), o.embedding]));
}

/** embedTexts(texts, task) -> Float32Array[] ; data = { outfits, wears, venueNames } */
export async function contextVector(ctx, embedTexts) {
  const [ev, ed] = await embedTexts(ctx.phrases(), "search");
  return mix([[C.VIBE_WEIGHT, ev], [1 - C.VIBE_WEIGHT, ed]]);
}

export function recommend(ctx, eCtx, { outfits, wears, venueNames = {} }, topK = 3,
                          { alpha = C.ALPHA, lam = C.DECAY_LAMBDA } = {}) {
  if (!outfits.length) return { context: ctx.describe(), season: ctx.season, results: [] };
  const when = toLocalISO(ctx.when);
  const history = {};
  for (const w of [...wears].filter((w) => w.worn_at < when).sort((a, b) => (a.worn_at < b.worn_at ? 1 : -1)))
    (history[w.outfit_id] ||= []).push(w);
  const venues = venueLabels(wears, venueNames);

  const recentAtVenue = new Set();
  if (ctx.venue && venues[ctx.venue] !== "Home") { // nobody judges repeats at home
    const since = toLocalISO(new Date(ctx.when - C.VENUE_REPEAT_DAYS * DAY_MS));
    const last = {};
    for (const w of wears)
      if (w.location_cluster === ctx.venue && w.worn_at >= since && w.worn_at < when)
        last[w.outfit_id] = w.worn_at > (last[w.outfit_id] || "") ? w.worn_at : last[w.outfit_id];
    Object.entries(last).sort((a, b) => (a[1] < b[1] ? 1 : -1)).slice(0, C.VENUE_RECENT_LOOKS)
      .forEach(([id]) => recentAtVenue.add(Number(id)));
  }

  const counts = Object.fromEntries(outfits.map((o) => [o.id, (history[o.id] || []).length || 1]));
  const q = mix([[alpha, eCtx], [1 - alpha, styleCentroid(outfits, counts)]]);
  const season = ctx.season;

  const results = outfits.map((o) => {
    const past = history[o.id] || [];
    const dt = past.length ? (ctx.when - parseLocal(past[0].worn_at)) / DAY_MS : null;
    const decay = recencyFactor(dt, lam);
    const sim = dot(q, o.embedding);
    const base = sim * decay;
    let fit = 1;
    const reasons = [], warnings = [];
    if (o.vibe_tag === ctx.vibe) { fit += C.VIBE_BOOST; reasons.push(`${ctx.vibe} look`); }
    if (season && o.season_tag === season) { fit += C.SEASON_BOOST; reasons.push(`suits ${season}`); }
    const fabric = (o.fabric_tag || "").toLowerCase();
    if (ctx.rain && RAIN_UNSAFE.has(fabric)) { fit *= 0.7; warnings.push(`${fabric} is risky in rain`); }
    if (season === HOT && HEAT_UNSAFE.has(fabric)) { fit *= 0.7; warnings.push(`${fabric} runs hot`); }
    let radar = 1;
    if (dt != null && dt < 7) {
      const days = calendarDays(parseLocal(past[0].worn_at), ctx.when);
      warnings.push(days === 0 ? "worn today" : days === 1 ? "worn yesterday" : `worn ${days} days ago`);
    }
    if (recentAtVenue.has(o.id)) {
      const hit = past.find((p) => p.location_cluster === ctx.venue);
      radar = C.VENUE_REPEAT_PENALTY;
      const days = Math.floor((ctx.when - parseLocal(hit.worn_at)) / DAY_MS);
      warnings.push(`same look at ${venues[ctx.venue] || ctx.venue} ${days}d ago`);
    }
    return {
      outfit_id: o.id, primary_category: o.primary_category, vibe_tag: o.vibe_tag, season_tag: o.season_tag,
      fabric_tag: o.fabric_tag, wear_count: o.wear_count,
      days_since_worn: dt == null ? null : Math.round(dt * 10) / 10,
      // multipliers must push negative similarities further down, not up
      score: base >= 0 ? base * fit * radar : base / (fit * radar),
      breakdown: { similarity: +sim.toFixed(4), recency: +decay.toFixed(4), context_fit: +fit.toFixed(3), venue_radar: radar },
      reasons, warnings,
    };
  });
  results.sort((a, b) => b.score - a.score);
  return { context: ctx.describe(), season, results: results.slice(0, topK) };
}

