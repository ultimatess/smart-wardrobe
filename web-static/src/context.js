// Port of engine/context.py: seasons, venue clusters, opt-in weather.
import { CLUSTER_RADIUS_KM, TRAVEL_RADIUS_KM } from "./config.js";
import { HOT, MILD, WET } from "./taxonomy.js";
import { parseLocal } from "./vec.js";

export function seasonFor(when, lat = null) {
  const d = typeof when === "string" ? parseLocal(when) : when;
  let m = d.getMonth() + 1;
  if (lat != null && lat < 0) m = ((m + 5) % 12) + 1;
  if (m >= 3 && m <= 6) return HOT;
  if (m >= 7 && m <= 9) return WET;
  return MILD;
}

export function seasonFromWeather(tempC, humidity, rain) {
  if (rain) return WET;
  if (tempC == null) return null;
  if (tempC >= 28 || (tempC >= 25 && (humidity || 0) >= 70)) return HOT;
  return MILD;
}

export function haversineKm(lat1, lon1, lat2, lon2) {
  const r = Math.PI / 180;
  const dp = (lat2 - lat1) * r, dl = (lon2 - lon1) * r;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dl / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(a));
}

const centroid = (pts) => pts.length
  ? [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length] : null;

function clusterCenters(wears) {
  const c = {};
  for (const w of wears) {
    if (!w.location_cluster) continue;
    c[w.location_cluster] ||= [];
    if (w.lat != null && w.lon != null) c[w.location_cluster].push([w.lat, w.lon]);
  }
  return c;
}

/** Greedy leader clustering against existing wear coordinates (500 m radius). */
export function assignLocationCluster(wears, lat, lon) {
  if (lat == null || lon == null) return null;
  const centers = Object.entries(clusterCenters(wears)).filter(([, pts]) => pts.length)
    .map(([c, pts]) => [c, centroid(pts)]);
  let best = null, bestD = Infinity;
  for (const [c, [la, lo]] of centers) {
    const d = haversineKm(lat, lon, la, lo);
    if (d < bestD) { best = c; bestD = d; }
  }
  if (best && bestD <= CLUSTER_RADIUS_KM) return best;
  const used = new Set(wears.map((w) => w.location_cluster).filter(Boolean));
  let n = centers.length + 1;
  while (used.has(`loc-${n}`)) n++;
  return `loc-${n}`;
}

/** Home / Office / Travel / Venue-N per cluster; user-chosen names (venueNames) win. */
export function venueLabels(wears, venueNames = {}) {
  const home = {}, office = {};
  const centers = clusterCenters(wears);
  for (const w of wears) {
    if (!w.location_cluster) continue;
    const d = parseLocal(w.worn_at);
    const weekday = d.getDay() >= 1 && d.getDay() <= 5;
    const t = weekday && d.getHours() >= 9 && d.getHours() < 18 ? office : home;
    t[w.location_cluster] = (t[w.location_cluster] || 0) + 1;
  }
  const argmax = (o) => Object.entries(o).sort((a, b) => b[1] - a[1])[0];
  const h = argmax(home)?.[0];
  if (h) delete office[h];
  const o = argmax(office);
  const officeC = o && o[1] >= 2 ? o[0] : null;
  const hc = h ? centroid(centers[h]) : null;
  const labels = {};
  for (const [c, pts] of Object.entries(centers)) {
    const cc = centroid(pts);
    if (c === h) labels[c] = "Home";
    else if (c === officeC) labels[c] = "Office";
    else if (hc && cc && haversineKm(...cc, ...hc) > TRAVEL_RADIUS_KM) labels[c] = "Travel";
    else labels[c] = `Venue ${c.split("-").at(-1)}`;
  }
  return { ...labels, ...venueNames };
}

/** Opt-in coarse weather. Only a 1-decimal (~11 km) coordinate is sent; no photos or vectors. */
export async function fetchWeather(lat, lon) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(1)}&longitude=${lon.toFixed(1)}`
    + "&current=temperature_2m,relative_humidity_2m,precipitation,rain";
  const cur = (await (await fetch(url)).json()).current;
  return { temp_c: cur.temperature_2m, humidity: cur.relative_humidity_2m,
           rain: (cur.rain || cur.precipitation || 0) > 0.1 };
}
