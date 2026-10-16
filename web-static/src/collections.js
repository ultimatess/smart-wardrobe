// Port of engine/wardrobe_collections.py
import { DAY_MS, dot, l2, toLocalISO } from "./vec.js";

const BREATHABLE = new Set(["linen", "cotton", "khadi", "chiffon", "rayon", "chikankari embroidered cotton"]);
const RULES = [
  ["Monsoon Office Wear", (o) => o.season_tag === "Monsoon" && o.vibe_tag === "Work"],
  ["Summer Breathables", (o) => o.season_tag === "Summer/Humid" && BREATHABLE.has(o.fabric_tag)],
  ["Ethnic & Festive", (o) => ["indian", "fusion"].includes(o.labels?.style) || o.vibe_tag === "Festive"],
  ["Western Workwear", (o) => o.labels?.style === "western" && o.vibe_tag === "Work"],
  ["Weekend Casuals", (o) => o.vibe_tag === "Casual"],
  ["Travel Ready", (o) => o.vibe_tag === "Travel"],
  ["Most Loved", (o) => o.wear_count >= 3],
];

/** Seeded RNG so clusters are stable between visits. */
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/** Spherical k-means with k-means++ init. */
export function kmeans(x, k, iters = 50, seed = 0) {
  const n = x.length, rand = rng(seed);
  k = Math.max(1, Math.min(k, n));
  const cent = [x[Math.floor(rand() * n)]];
  while (cent.length < k) {
    const d = x.map((v) => Math.max(0, 1 - Math.max(...cent.map((c) => dot(v, c)))) ** 2);
    const tot = d.reduce((a, b) => a + b, 0);
    let r = rand() * tot, i = 0;
    while (tot > 0 && i < n - 1 && (r -= d[i]) > 0) i++;
    cent.push(x[tot > 0 ? i : Math.floor(rand() * n)]);
  }
  let c = cent.map((v) => Float32Array.from(v));
  let labels = new Array(n).fill(-1);
  for (let it = 0; it < iters; it++) {
    const next = x.map((v) => c.reduce((bi, cj, j) => (dot(v, cj) > dot(v, c[bi]) ? j : bi), 0));
    if (next.every((l, i) => l === labels[i])) break;
    labels = next;
    c = c.map((cj, j) => {
      const m = x.filter((_, i) => labels[i] === j);
      if (!m.length) return cj;
      const s = new Float32Array(cj.length);
      m.forEach((v) => v.forEach((val, d) => (s[d] += val)));
      return l2(s);
    });
  }
  return { labels, centroids: c };
}

export function buildCollections(outfits, bank, now = new Date()) {
  const out = [];
  const byWear = [...outfits].sort((a, b) => b.wear_count - a.wear_count);
  for (const [name, pred] of RULES) {
    const ids = byWear.filter(pred).map((o) => o.id);
    if (ids.length) out.push({ name, kind: "rule", outfit_ids: ids });
  }
  const cutoff = toLocalISO(new Date(now - 60 * DAY_MS));
  const gems = outfits.filter((o) => o.wear_count <= 2 && (o.last_worn_timestamp || "") < cutoff)
    .sort((a, b) => (a.last_worn_timestamp < b.last_worn_timestamp ? -1 : 1)).map((o) => o.id);
  if (gems.length) out.push({ name: "Seldom Worn Gems", kind: "rule", outfit_ids: gems });

  if (outfits.length >= 4 && bank) {
    const k = Math.max(2, Math.min(8, Math.round(Math.sqrt(outfits.length / 2))));
    const { labels, centroids } = kmeans(outfits.map((o) => o.embedding), k);
    centroids.forEach((cj, j) => {
      const members = outfits.filter((_, i) => labels[i] === j).map((o) => o.id);
      if (members.length < 2) return;
      const cat = bank.scores(cj, "category")[0][0];
      const vibe = bank.scores(cj, "vibe")[0][0];
      const t = cat.replace(/\b[a-z]/g, (ch) => ch.toUpperCase());
      out.push({ name: `${vibe}: ${t} & similar`, kind: "cluster", outfit_ids: members });
    });
  }
  return out;
}
