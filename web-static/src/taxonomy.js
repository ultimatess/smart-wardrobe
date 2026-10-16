// Zero-shot anchor bank. Rows come from engine/taxonomy.py via `uv run python -m engine.export_web`.
import TAXONOMY from "./taxonomy.json" with { type: "json" };
import { dot, l2, mean } from "./vec.js";

export const { vibes: VIBES, seasons: SEASONS } = TAXONOMY;
export const [HOT, WET, MILD] = SEASONS;
export const TAXONOMY_HASH = TAXONOMY.hash;
export const ROWS = TAXONOMY.rows;

const SOFTMAX_T = 0.02;
const FABRIC_NAME_MIN_P = 0.5;

/** Embeds every prompt (classification prefix), averages per row. embedTexts(texts) -> Float32Array[] */
export async function buildAnchorMatrix(embedTexts, onProgress) {
  const flat = [], owner = [];
  ROWS.forEach((r, i) => r.prompts.forEach((p) => { flat.push(p); owner.push(i); }));
  const vecs = [];
  const B = 32;
  for (let i = 0; i < flat.length; i += B) {
    vecs.push(...(await embedTexts(flat.slice(i, i + B), "classification")));
    onProgress?.(Math.min(1, (i + B) / flat.length));
  }
  return ROWS.map((_, i) => mean(vecs.filter((_, j) => owner[j] === i)));
}

export class AnchorBank {
  constructor(matrix) {
    this.matrix = matrix.map(l2);
    this.axes = {};
    ROWS.forEach((r, i) => (this.axes[r.axis] ||= []).push(i));
  }
  byLabel(axis, label) {
    return ROWS.find((r) => r.axis === axis && r.label === label);
  }
  scores(vec, axis) {
    return this.axes[axis]
      .map((i) => [ROWS[i].label, dot(this.matrix[i], vec)])
      .sort((a, b) => b[1] - a[1]);
  }
  classify(vec, topK = 5) {
    const out = {};
    for (const axis of Object.keys(this.axes)) out[axis] = this.scores(vec, axis).slice(0, topK);
    return out;
  }
}

function softmax(axisScores) {
  const m = Math.max(...axisScores.map(([, v]) => v));
  const e = axisScores.map(([l, v]) => [l, Math.exp((v - m) / SOFTMAX_T)]);
  const z = e.reduce((s, [, v]) => s + v, 0);
  return e.map(([l, v]) => [l, v / z]);
}
function priorVotes(axisScores, bank, axis, attr) {
  const votes = {};
  for (const [label, p] of softmax(axisScores)) {
    const targets = bank.byLabel(axis, label)?.[attr] || [];
    for (const t of targets) votes[t] = (votes[t] || 0) + p / targets.length;
  }
  return votes;
}
function combine(...weighted) {
  const total = {};
  for (const [w, dist] of weighted) for (const [k, v] of Object.entries(dist)) total[k] = (total[k] || 0) + w * v;
  return Object.entries(total).sort((a, b) => b[1] - a[1])[0][0];
}
const title = (s) => s.toLowerCase().replace(/\b[a-z]/g, (c) => c.toUpperCase()); // == Python str.title()

/** Port of engine.taxonomy.derive_tags */
export function deriveTags(labels, bank, exifSeason = null) {
  const [cat, catS] = labels.category[0];
  const color = labels.color[0][0];
  const fabric = labels.fabric[0][0];
  const fabricWord = fabric.split(" ").at(-1);
  const fabricP = softmax(labels.fabric)[0][1];
  const noun = cat.includes(fabricWord) || fabricP < FABRIC_NAME_MIN_P ? cat : `${fabricWord} ${cat}`;
  const primary = title(`${color.split(" ")[0]} ${noun}`);
  const vibe = combine([1.0, Object.fromEntries(softmax(labels.vibe))],
    [1.5, priorVotes(labels.category, bank, "category", "vibes")]);
  const season = combine([1.0, Object.fromEntries(softmax(labels.season))],
    [0.6, priorVotes(labels.category, bank, "category", "seasons")],
    [0.6, priorVotes(labels.fabric, bank, "fabric", "seasons")],
    [0.8, exifSeason ? { [exifSeason]: 1 } : {}]);
  return {
    primary_category: primary, category: cat, category_score: catS, fabric_tag: fabric, color_tag: color,
    vibe_tag: vibe, season_tag: season, style: bank.byLabel("category", cat).style,
  };
}
