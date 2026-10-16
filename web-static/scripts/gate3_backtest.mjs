// Gate 3 backtest for the browser build: same mock wardrobe + scenarios as tests/gate3_backtest.py,
// scored by the JS recommender with the q4 ONNX model (Node, CPU).  Run: npm run gate3
import { readFileSync } from "node:fs";
import { loadEmbedder } from "../src/embed-core.js";
import { DayContext, HEAT_UNSAFE, RAIN_UNSAFE, contextVector, recommend } from "../src/recommender.js";
import { toLocalISO } from "../src/vec.js";

// Pull MOCK / ANCHOR / SCENARIOS out of the Python harness so the two never drift.
const py = readFileSync(new URL("../../tests/gate3_backtest.py", import.meta.url), "utf8");
const block = py.slice(py.indexOf("MOCK = ["), py.indexOf("]\n\nANCHOR")) + "]";
const HOT = "Summer/Humid", WET = "Monsoon", MILD = "Mild/Winter";
const MOCK = Function("HOT", "WET", "MILD", `return ${block.slice(7).replace(/\(/g, "[").replace(/\)/g, "]")};`)(HOT, WET, MILD);
const VENUE_NAMES = { home: "Home", office: "Office", hall: "Function hall", cafe: "Cafe", airport: "Airport" };

const ANCHOR = new Date(2026, 7, 12, 8, 30); // Wed
const at = (days, hours = 0) => new Date(ANCHOR.getTime() + days * 864e5 + hours * 36e5);
const SCENARIOS = [
  ["34C humid Wednesday + Work", { vibe: "Work", when: ANCHOR, temp_c: 34, humidity: 80, rain: false, venue: "office" }],
  ["Rainy Sunday evening + Festive", { vibe: "Festive", when: at(4, 10), temp_c: 26, humidity: 92, rain: true, venue: "hall" }],
  ["Mild 18C Saturday + Casual", { vibe: "Casual", when: at(3, 2), temp_c: 18, humidity: 45, rain: false, venue: "cafe" }],
  ["Warm 29C Friday + Travel", { vibe: "Travel", when: at(2, -3), temp_c: 29, humidity: 60, rain: false, venue: "airport" }],
  ["Monsoon Thursday + Work (office radar)", { vibe: "Work", when: at(1), temp_c: 27, humidity: 88, rain: true, venue: "office" }],
];

const emb = await loadEmbedder({ device: "cpu" });
const vecs = await emb.embedTexts(MOCK.map((m) => m[0]), "document");
const outfits = [], wears = [];
MOCK.forEach(([desc, vibe, season, fabric, ws], i) => {
  outfits.push({ id: i + 1, primary_category: desc.split(",")[0], vibe_tag: vibe, season_tag: season,
                 fabric_tag: fabric, wear_count: ws.length, embedding: vecs[i], labels: {} });
  for (const [d, venue] of ws)
    wears.push({ outfit_id: i + 1, worn_at: toLocalISO(at(-d, venue === "office" ? 2 : 11)), location_cluster: venue });
});

let passed = 0;
for (const [name, opts] of SCENARIOS) {
  const ctx = new DayContext(opts);
  const full = recommend(ctx, await contextVector(ctx, emb.embedTexts), { outfits, wears, venueNames: VENUE_NAMES }, 1e4);
  const top = full.results.slice(0, 3);
  const fails = [];
  if (top.filter((r) => r.vibe_tag === ctx.vibe).length < 2) fails.push("vibe: fewer than 2/3 match the pill");
  for (const r of top) {
    const f = r.fabric_tag.toLowerCase();
    if ((ctx.rain && RAIN_UNSAFE.has(f)) || (ctx.season === HOT && HEAT_UNSAFE.has(f))) fails.push(`weather: ${r.primary_category}`);
    if (r.days_since_worn != null && r.days_since_worn < 3) fails.push(`anti-repeat: ${r.primary_category}`);
  }
  if (top[0].breakdown.venue_radar < 1) fails.push("venue radar: #1 repeats at this venue");
  const flagged = full.results.filter((r) => r.breakdown.venue_radar < 1);
  if (["office", "hall"].includes(ctx.venue) && !flagged.length) fails.push("venue radar never fired");
  passed += !fails.length;
  console.log(`\n### ${name}  [season=${full.season}]`);
  top.forEach((r, i) => console.log(`  ${i + 1}. ${r.primary_category.padEnd(58)} ${r.vibe_tag.padEnd(8)} score=${r.score.toFixed(3)}`));
  if (flagged.length) console.log(`    radar held back: ${flagged.map((r) => r.primary_category).join(", ")}`);
  console.log("    ->", fails.length ? "FAIL: " + fails.join("; ") : "PASS");
}
console.log(`\n== GATE 3 (browser build, ${emb.device}, q4): ${passed === SCENARIOS.length ? "PASS" : "FAIL"} (${passed}/${SCENARIOS.length})`);
process.exit(passed === SCENARIOS.length ? 0 : 1);
