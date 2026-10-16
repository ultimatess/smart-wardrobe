// Pure-logic tests for the browser build (mirrors tests/test_engine.py). Run: npm test
import "fake-indexeddb/auto";
import { describe, expect, it } from "vitest";
import { buildCollections, kmeans } from "../src/collections.js";
import { assignLocationCluster, haversineKm, seasonFor, seasonFromWeather, venueLabels } from "../src/context.js";
import { boxesFromFace } from "../src/locator.js";
import { DayContext, recencyFactor, recommend, styleCentroid } from "../src/recommender.js";
import * as store from "../src/store.js";
import { AnchorBank, ROWS, deriveTags } from "../src/taxonomy.js";
import { l2, toLocalISO } from "../src/vec.js";
import * as C from "../src/config.js";

const unit = (seed) => {
  let s = seed;
  return l2(Float32Array.from({ length: C.EMBED_DIM }, () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5));
};
const at = (y, m, d, h = 9) => new Date(y, m - 1, d, h);

describe("taxonomy", () => {
  it("has 150+ bi-cultural anchors exported from Python", () => {
    expect(ROWS.length).toBeGreaterThanOrEqual(150);
    const styles = new Set(ROWS.filter((r) => r.axis === "category").map((r) => r.style));
    ["indian", "western", "fusion"].forEach((s) => expect(styles.has(s)).toBe(true));
  });
  it("derives festive/winter tags from a sherwani vector", () => {
    const bank = new AnchorBank(ROWS.map((_, i) => unit(i + 1)));
    const idx = ROWS.findIndex((r) => r.axis === "category" && r.label === "sherwani");
    const tags = deriveTags(bank.classify(bank.matrix[idx]), bank, "Mild/Winter");
    expect(tags.category).toBe("sherwani");
    expect(tags.vibe_tag).toBe("Festive");
    expect(tags.season_tag).toBe("Mild/Winter");
    expect(tags.primary_category).toMatch(/Sherwani/);
  });
});

describe("context", () => {
  it("maps seasons and weather like the Python build", () => {
    expect(seasonFor("2026-04-15T10:00:00")).toBe("Summer/Humid");
    expect(seasonFor("2026-08-01T10:00:00")).toBe("Monsoon");
    expect(seasonFor("2026-12-25T10:00:00")).toBe("Mild/Winter");
    expect(seasonFor("2026-12-25T10:00:00", -33.9)).toBe("Summer/Humid");
    expect(seasonFromWeather(34, 80, false)).toBe("Summer/Humid");
    expect(seasonFromWeather(26, 90, true)).toBe("Monsoon");
    expect(haversineKm(12.97, 77.59, 13.08, 80.27)).toBeCloseTo(290, -1);
  });
  it("clusters nearby GPS and tolerates wears without coordinates", () => {
    const wears = [{ location_cluster: "loc-1", lat: 12.97, lon: 77.59, worn_at: "2026-08-01T20:00:00" },
                   { location_cluster: "loc-1", lat: null, lon: null, worn_at: "2026-08-02T20:00:00" }];
    expect(assignLocationCluster(wears, 12.971, 77.591)).toBe("loc-1");
    expect(assignLocationCluster(wears, 13.5, 77.59)).toBe("loc-2");
    expect(venueLabels(wears, { "loc-1": "Parents" })["loc-1"]).toBe("Parents");
  });
});

describe("recommender", () => {
  const outfits = [
    { id: 1, primary_category: "linen kurta", vibe_tag: "Work", season_tag: "Summer/Humid", fabric_tag: "linen", wear_count: 1, embedding: unit(11) },
    { id: 2, primary_category: "silk sherwani", vibe_tag: "Festive", season_tag: "Mild/Winter", fabric_tag: "silk", wear_count: 1, embedding: unit(12) },
    { id: 3, primary_category: "formal shirt", vibe_tag: "Work", season_tag: "Monsoon", fabric_tag: "cotton", wear_count: 1, embedding: unit(13) },
    { id: 4, primary_category: "jeans tee", vibe_tag: "Casual", season_tag: "Monsoon", fabric_tag: "denim", wear_count: 1, embedding: unit(14) },
  ];
  const now = at(2026, 8, 5);
  const wears = [
    { outfit_id: 3, worn_at: toLocalISO(at(2026, 8, 4, 11)), location_cluster: "office" },
    { outfit_id: 1, worn_at: toLocalISO(at(2026, 7, 16, 11)), location_cluster: "office" },
    { outfit_id: 2, worn_at: toLocalISO(at(2026, 8, 6, 11)), location_cluster: "hall" }, // future: ignored
  ];
  it("applies decay, venue radar, weather guard; ignores future wears", () => {
    const ctx = new DayContext({ vibe: "Work", when: now, temp_c: 27, humidity: 85, rain: true, venue: "office" });
    const res = recommend(ctx, unit(99), { outfits, wears, venueNames: { office: "Office" } }, 10);
    const by = Object.fromEntries(res.results.map((r) => [r.outfit_id, r]));
    expect(by[3].breakdown.recency).toBeLessThan(0.2);
    expect(by[3].warnings).toContain("worn yesterday");
    expect(by[1].breakdown.venue_radar).toBe(C.VENUE_REPEAT_PENALTY);
    expect(by[2].warnings.some((w) => w.includes("risky in rain"))).toBe(true);
    expect(by[2].breakdown.recency).toBe(1);
    expect(res.results[0].outfit_id).not.toBe(3);
  });
  it("exempts Home from the radar", () => {
    const ctx = new DayContext({ vibe: "Work", when: now, venue: "office" });
    const res = recommend(ctx, unit(99), { outfits, wears, venueNames: { office: "Home" } }, 10);
    expect(res.results.every((r) => r.breakdown.venue_radar === 1)).toBe(true);
  });
  it("recency and centroid math", () => {
    expect(recencyFactor(null)).toBe(1);
    expect(recencyFactor(0)).toBe(0);
    expect(recencyFactor(1)).toBeCloseTo(1 - Math.exp(-C.DECAY_LAMBDA));
    const c = styleCentroid([{ id: 1, embedding: unit(3) }, { id: 2, embedding: unit(4) }], { 1: 20, 2: 1 });
    const d = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
    expect(d(c, unit(3))).toBeGreaterThan(d(c, unit(4)));
  });
  it("collections + kmeans", () => {
    const names = buildCollections(outfits, null).map((c) => c.name);
    expect(names).toContain("Monsoon Office Wear");
    const a = unit(21), b = unit(22);
    const jitter = (v, s) => l2(v.map((x, i) => x + 0.02 * Math.sin(i * s)));
    const x = [...[1, 2, 3, 4].map((s) => jitter(a, s)), ...[5, 6, 7, 8].map((s) => jitter(b, s))];
    const { labels } = kmeans(x, 2);
    expect(new Set(labels.slice(0, 4)).size).toBe(1);
    expect(labels[0]).not.toBe(labels[4]);
  });
  it("face heuristic body box sits below the face", () => {
    const b = boxesFromFace({ x0: 100, y0: 50, x1: 160, y1: 120 }, 600, 900);
    expect(b.body.y0).toBe(120);
    expect(b.body.y1).toBeGreaterThan(400);
  });
});

describe("store", () => {
  it("persists and cascades deletes", async () => {
    const id = await store.put("outfits", { primary_category: "x", embedding: unit(1) });
    await store.put("wears", { outfit_id: id, worn_at: "2026-08-01T10:00:00" });
    await store.put("photos", { key: "a|1|1", outfit_id: id, status: "new_outfit" });
    expect((await store.get("outfits", id)).embedding).toBeInstanceOf(Float32Array);
    await store.deleteOutfit(id);
    expect(await store.get("outfits", id)).toBeUndefined();
    expect((await store.all("wears")).length).toBe(0);
    expect((await store.get("photos", "a|1|1")).status).toBe("deleted");
  });
});
