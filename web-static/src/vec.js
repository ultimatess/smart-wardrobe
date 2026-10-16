export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
export const l2 = (a) => {
  const n = Math.sqrt(dot(a, a)) || 1;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / n;
  return out;
};
/** sum_k w_k * v_k, then L2-normalised */
export const mix = (pairs) => {
  const out = new Float32Array(pairs[0][1].length);
  for (const [w, v] of pairs) for (let i = 0; i < out.length; i++) out[i] += w * v[i];
  return l2(out);
};
export const mean = (vs) => mix(vs.map((v) => [1, v]));

// Local-time ISO without zone, same format the Python build stores ("2026-08-12T08:30:00").
export const toLocalISO = (d) => {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};
export const parseLocal = (s) => new Date(s); // no 'Z' -> parsed as local time
export const DAY_MS = 864e5;
