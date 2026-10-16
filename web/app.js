const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
let vibe = "Work";
let meta = { vibes: [], seasons: [] };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ago = (iso) => {
  if (!iso) return "never";
  const d = Math.round((Date.now() - new Date(iso)) / 864e5);
  return d <= 0 ? "today" : d === 1 ? "yesterday" : `${d} days ago`;
};

// ---------------------------------------------------------------- tabs
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
  ({ wardrobe: loadWardrobe, collections: loadCollections, today: () => {} })[t.dataset.view]?.();
}));

// ---------------------------------------------------------------- today
$$(".pill").forEach((p) => p.addEventListener("click", () => {
  vibe = p.dataset.vibe;
  $$(".pill").forEach((x) => x.classList.toggle("active", x === p));
  recommend();
}));
$("#go").addEventListener("click", recommend);
$("#auto-weather").addEventListener("click", async () => {
  const w = await api("/api/weather");
  if (!w.available) return alert(w.hint);
  $("#temp").value = Math.round(w.temp_c);
  $("#humidity").value = Math.round(w.humidity);
  $("#rain").checked = w.rain;
  recommend();
});

function card(r, withWear = true) {
  const b = r.breakdown || {};
  return `<article class="card" data-id="${r.outfit_id ?? r.id}">
    <img src="${r.crop_url}" alt="" loading="lazy">
    <div class="body">
      <h4>${esc(r.primary_category)}</h4>
      <div class="tags"><span>${esc(r.vibe_tag)}</span><span>${esc(r.season_tag)}</span><span>${esc(r.fabric_tag)}</span></div>
      <p class="muted">Worn ${r.wear_count}× &middot; last ${r.days_since_worn == null ? "never" : Math.round(r.days_since_worn) + "d ago"}</p>
      ${(r.reasons || []).map((x) => `<p class="ok">✓ ${esc(x)}</p>`).join("")}
      ${(r.warnings || []).map((x) => `<p class="warn">⚠ ${esc(x)}</p>`).join("")}
      ${r.score != null ? `<p class="muted small">score ${r.score.toFixed(3)} = sim ${b.similarity} × recency ${b.recency} × fit ${b.context_fit} × radar ${b.venue_radar}</p>` : ""}
      ${withWear ? `<button class="wear primary" data-id="${r.outfit_id}">Wearing this today</button>` : ""}
    </div></article>`;
}

async function recommend() {
  const q = new URLSearchParams({ vibe, k: 3 });
  if ($("#temp").value) q.set("temp_c", $("#temp").value);
  if ($("#humidity").value) q.set("humidity", $("#humidity").value);
  if ($("#rain").checked) q.set("rain", "true");
  if ($("#venue").value) q.set("venue", $("#venue").value);
  $("#recs").innerHTML = `<p class="muted">Thinking locally…</p>`;
  try {
    const res = await api(`/api/recommend?${q}`);
    $("#context").textContent = `Query: “${res.context}”`;
    $("#recs").innerHTML = res.results.length ? res.results.map((r) => card(r)).join("")
      : `<p class="muted">No outfits yet — import some photos first.</p>`;
  } catch (e) { $("#recs").innerHTML = `<p class="warn">${esc(e.message)}</p>`; }
}

$("#recs").addEventListener("click", async (e) => {
  const btn = e.target.closest("button.wear");
  if (btn) {
    await api(`/api/outfits/${btn.dataset.id}/wear`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ venue_cluster: $("#venue").value || null }),
    });
    btn.textContent = "Logged ✓"; btn.disabled = true;
    return;
  }
  const c = e.target.closest(".card"); if (c) showDetail(c.dataset.id);
});

// ---------------------------------------------------------------- wardrobe
let searchTimer;
$("#search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadWardrobe, 350); });
$("#f-vibe").addEventListener("change", loadWardrobe);
$("#f-season").addEventListener("change", loadWardrobe);

async function loadWardrobe() {
  const q = new URLSearchParams();
  if ($("#search").value.trim()) q.set("q", $("#search").value.trim());
  if ($("#f-vibe").value) q.set("vibe", $("#f-vibe").value);
  if ($("#f-season").value) q.set("season", $("#f-season").value);
  const items = await api(`/api/outfits?${q}`);
  $("#grid").innerHTML = items.length ? items.map((o) => `
    <figure class="tile" data-id="${o.id}"><img src="${o.crop_url}" loading="lazy" alt="">
      <figcaption>${esc(o.primary_category)}<br><span class="muted">${o.wear_count}× · ${ago(o.last_worn_timestamp)}</span></figcaption>
    </figure>`).join("") : `<p class="muted">Nothing here yet.</p>`;
}
$("#grid").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(t.dataset.id); });

async function showDetail(id) {
  const o = await api(`/api/outfits/${id}`);
  const opt = (list, cur) => list.map((x) => `<option ${x === cur ? "selected" : ""}>${esc(x)}</option>`).join("");
  const top = (o.labels.category || []).map(([l, s]) => `${esc(l)} (${s.toFixed(2)})`).join(", ");
  $("#detail-body").innerHTML = `
    <img class="hero" src="${o.crop_url}" alt="">
    <label>Name <input id="d-name" value="${esc(o.primary_category)}"></label>
    <label>Vibe <select id="d-vibe">${opt(meta.vibes, o.vibe_tag)}</select></label>
    <label>Season <select id="d-season">${opt(meta.seasons, o.season_tag)}</select></label>
    <p class="muted small">Zero-shot: ${top}</p>
    <p class="muted small">Fabric ${esc(o.fabric_tag)} · Colour ${esc(o.color_tag)} · ${o.photos.length} photo(s)</p>
    <h4>Wear history</h4>
    <ul>${o.wears.map((w) => `<li>${esc(w.worn_at.replace("T", " "))} ${w.venue ? "· " + esc(w.venue) : ""} <span class="muted">${esc(w.source)}</span></li>`).join("")}</ul>
    <button type="button" id="d-save" class="primary">Save tags</button>
    <button type="button" id="d-del" class="ghost danger">Remove outfit</button>`;
  $("#d-save").onclick = async () => {
    await api(`/api/outfits/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ primary_category: $("#d-name").value, vibe_tag: $("#d-vibe").value, season_tag: $("#d-season").value }) });
    $("#detail").close(); loadWardrobe();
  };
  $("#d-del").onclick = async () => {
    if (!confirm("Remove this outfit from the wardrobe? Photos stay in your gallery.")) return;
    await api(`/api/outfits/${id}`, { method: "DELETE" }); $("#detail").close(); loadWardrobe();
  };
  $("#detail").showModal();
}

// ---------------------------------------------------------------- collections
async function loadCollections() {
  $("#cols").innerHTML = `<p class="muted">Clustering…</p>`;
  const [cols, items] = await Promise.all([api("/api/collections"), api("/api/outfits")]);
  const byId = Object.fromEntries(items.map((o) => [o.id, o]));
  $("#cols").innerHTML = cols.length ? cols.map((c) => `
    <h3>${esc(c.name)} <span class="muted small">${c.outfit_ids.length} · ${c.kind}</span></h3>
    <div class="row">${c.outfit_ids.map((i) => byId[i]).filter(Boolean).map((o) =>
      `<figure class="tile sm" data-id="${o.id}"><img src="${o.crop_url}" loading="lazy" alt=""><figcaption>${esc(o.primary_category)}</figcaption></figure>`).join("")}</div>`).join("")
    : `<p class="muted">Collections appear once you have a few outfits.</p>`;
}
$("#cols").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(t.dataset.id); });

// ---------------------------------------------------------------- import
const log = (x) => { $("#log").textContent = typeof x === "string" ? x : JSON.stringify(x, null, 2); };
$("#selfie").addEventListener("change", async (e) => {
  const fd = new FormData(); fd.append("file", e.target.files[0]);
  await api("/api/reference-selfie", { method: "POST", body: fd });
  $("#selfie-state").textContent = "Saved ✓"; health();
});
$("#photos").addEventListener("change", async (e) => {
  const fd = new FormData(); [...e.target.files].forEach((f) => fd.append("files", f));
  log(`Uploading and analysing ${e.target.files.length} photo(s) on-device…`);
  try { log(summary(await api("/api/upload", { method: "POST", body: fd }))); } catch (err) { log(err.message); }
  health();
});
$("#scan").addEventListener("click", async () => {
  log("Scanning data/raw_gallery… (first run loads the model)");
  try { log(summary(await api("/api/ingest", { method: "POST" }))); } catch (err) { log(err.message); }
  health();
});
const summary = (r) => `Processed ${r.processed}: ${JSON.stringify(r.counts)}\n\n` +
  r.results.map((x) => `${x.file}: ${x.status}${x.outfit_id ? " → #" + x.outfit_id + " " + (x.primary_category || "") : ""}${x.similarity ? " (sim " + x.similarity.toFixed(3) + ")" : ""}${x.detail ? " [" + x.detail + "]" : ""}`).join("\n");

async function loadVenues(list) {
  const venues = list || await api("/api/venues");
  const cur = $("#venue").value;
  $("#venue").innerHTML = `<option value="">Anywhere</option>` +
    venues.map((v) => `<option value="${esc(v.cluster)}">${esc(v.label)} (${v.wears})</option>`).join("");
  $("#venue").value = cur;
}
$("#rename-venue").addEventListener("click", async () => {
  const c = $("#venue").value;
  if (!c) return alert("Pick a venue first.");
  const label = prompt("Name this place (e.g. Office, Home, Parents' place):");
  if (label) loadVenues(await api(`/api/venues/${encodeURIComponent(c)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label }) }));
});

// ---------------------------------------------------------------- boot
async function health() {
  const h = await api("/api/health");
  meta = h;
  $("#status").textContent = `${h.outfits} outfits · local · ${h.reference_selfie ? "face anchored" : "no selfie"}`;
  $("#selfie-state").textContent = h.reference_selfie ? "Reference selfie set ✓" : "";
  return h;
}
(async () => {
  const h = await health();
  $("#f-vibe").innerHTML += h.vibes.map((v) => `<option>${v}</option>`).join("");
  $("#f-season").innerHTML += h.seasons.map((v) => `<option>${v}</option>`).join("");
  await loadVenues();
  const day = new Date().getDay();
  const def = day === 0 || day === 6 ? "Casual" : "Work";
  $$(".pill").forEach((p) => p.classList.toggle("active", p.dataset.vibe === def)); vibe = def;
  if (h.outfits) recommend();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
})();
