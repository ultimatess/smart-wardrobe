import * as api from "./app-api.js";

window.wardrobe = api; // console access for debugging / e2e tests

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (id) => `<svg><use href="#i-${id}"/></svg>`;
const ago = (iso) => {
  if (!iso) return "never";
  const d = Math.round((Date.now() - new Date(iso)) / 864e5);
  return d <= 0 ? "today" : d === 1 ? "yesterday" : d < 30 ? `${d}d ago` : d < 365 ? `${Math.round(d / 30)}mo ago` : `${Math.round(d / 365)}y ago`;
};
const VIBE_EMOJI = { Work: "🏢", Casual: "☕", Festive: "✨", Travel: "✈️" };

// ---------------------------------------------------------------- toasts + sheets
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = `${icon(kind === "warn" ? "alert" : "check")}<span>${esc(msg)}</span>`;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 3200);
}
const sheet = $("#sheet");
const openSheet = (html) => { $("#sheet-body").innerHTML = html; sheet.showModal(); };
const closeSheet = () => sheet.close();
$(".sheet-close").addEventListener("click", closeSheet);
sheet.addEventListener("click", (e) => { if (e.target === sheet) closeSheet(); }); // tap backdrop

function ask({ title, text, ok = "Continue", danger = false, input = null }) {
  return new Promise((resolve) => {
    openSheet(`<div style="max-width:440px;margin:0 auto">
      <h2 style="margin:16px 0 8px">${esc(title)}</h2><p class="muted" style="margin:0 0 16px">${esc(text)}</p>
      ${input != null ? `<div class="form"><input id="ask-input" value="${esc(input)}" placeholder="Name"></div>` : ""}
      <div class="sheet-actions"><button class="btn ghost" id="ask-no">Cancel</button>
      <button class="btn ${danger ? "ghost danger" : "primary"}" id="ask-ok">${esc(ok)}</button></div></div>`);
    const done = (v) => { sheet.removeEventListener("close", onClose); closeSheet(); resolve(v); };
    const onClose = () => resolve(null);
    sheet.addEventListener("close", onClose, { once: true });
    $("#ask-no").onclick = () => done(null);
    $("#ask-ok").onclick = () => done(input != null ? $("#ask-input").value : true);
    $("#ask-input")?.focus();
  });
}

// ---------------------------------------------------------------- model progress
const files = {};
api.onModelProgress((p) => {
  if (p.status !== "progress" || !p.total) return;
  files[p.file] = [p.loaded, p.total];
  const [l, t] = Object.values(files).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
  $("#model-card").hidden = false;
  $("#model-text").textContent = `Loading the on-device AI model · ${(l / 1e6).toFixed(0)} of ${(t / 1e6).toFixed(0)} MB (first visit only)`;
  $("#model-bar").style.width = `${(100 * l) / t}%`;
});
api.onStatus((m) => {
  $("#model-card").hidden = !m;
  if (m) { $("#model-text").textContent = m; if (!m.startsWith("Loading Embedding")) $("#model-bar").style.width = "100%"; }
});

// ---------------------------------------------------------------- navigation
function go(view) {
  $$(".nav-btn, .tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  window.scrollTo({ top: 0 });
  ({ wardrobe: loadWardrobe, collections: loadCollections })[view]?.();
}
$$(".nav-btn, .tab-btn").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));

// ---------------------------------------------------------------- today
const ctxState = JSON.parse(localStorage.getItem("ctx") || "{}");
let vibe = (() => { const d = new Date().getDay(); return d === 0 || d === 6 ? "Casual" : "Work"; })();

function setGreeting() {
  const now = new Date(), h = now.getHours();
  $("#greeting").textContent = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
  $("#today-date").textContent = now.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
}
function setVibe(v) {
  vibe = v;
  $$("#vibes button").forEach((b) => b.classList.toggle("active", b.dataset.vibe === v));
}
$$("#vibes button").forEach((b) => b.addEventListener("click", () => { setVibe(b.dataset.vibe); recommend(); }));

function readCtx() {
  const num = (s) => ($(s).value === "" ? null : Number($(s).value));
  return { temp_c: num("#temp"), humidity: num("#humidity"), rain: $("#rain").checked, venue: $("#venue").value || null };
}
function ctxSummary() {
  const c = readCtx();
  const parts = [];
  if (c.temp_c != null) parts.push(`${c.temp_c}°C`);
  if (c.humidity != null) parts.push(`${c.humidity}%`);
  if (c.rain) parts.push("Rain");
  if (c.venue) parts.push($("#venue").selectedOptions[0]?.dataset.label || "Venue");
  $("#ctx-summary").textContent = parts.length ? parts.join(" · ") : "Set weather";
  localStorage.setItem("ctx", JSON.stringify({ temp_c: c.temp_c, humidity: c.humidity, rain: c.rain, venue: c.venue }));
}
$("#ctx-toggle").addEventListener("click", () => {
  const open = $("#ctx-panel").hidden;
  $("#ctx-panel").hidden = !open;
  $("#ctx-toggle").setAttribute("aria-expanded", String(open));
});
$$("[data-step]").forEach((b) => b.addEventListener("click", () => {
  const [id, d] = b.dataset.step.split(":");
  const el = $(`#${id}`);
  const base = el.value === "" ? (id === "temp" ? 28 : 60) : Number(el.value);
  el.value = Math.max(id === "humidity" ? 0 : -20, Math.min(id === "humidity" ? 100 : 50, base + Number(d)));
  ctxSummary();
}));
["#temp", "#humidity", "#rain", "#venue"].forEach((s) => $(s).addEventListener("change", ctxSummary));
$("#go").addEventListener("click", () => { $("#ctx-panel").hidden = true; $("#ctx-toggle").setAttribute("aria-expanded", "false"); recommend(); });

$("#auto-weather").addEventListener("click", async () => {
  if (localStorage.getItem("weatherOk") !== "1") {
    const ok = await ask({ title: "Use your local weather?", ok: "Allow",
      text: "Your approximate location (rounded to about 11 km) is sent to open-meteo.com to fetch the forecast. Photos and outfits never leave this device." });
    if (!ok) return;
    localStorage.setItem("weatherOk", "1");
  }
  try {
    const w = await api.weather();
    $("#temp").value = Math.round(w.temp_c);
    $("#humidity").value = Math.round(w.humidity);
    $("#rain").checked = w.rain;
    ctxSummary();
    toast(`Weather: ${Math.round(w.temp_c)}°C, ${Math.round(w.humidity)}%${w.rain ? ", rain" : ""}`);
  } catch (e) { toast(e.message, "warn"); }
});

// "Match" = how well the look fits today's vibe/weather/venue. Freshness is shown separately as a warning.
const matchPct = (r) => { const b = r.breakdown; return Math.round(100 * Math.max(0, Math.min(1, (b.similarity * b.context_fit * b.venue_radar) / 1.2))); };

function recCard(r, i) {
  const b = r.breakdown, hero = i === 0;
  const d = r.days_since_worn;
  const last = d == null ? "Not worn yet" : `Worn ${r.wear_count}× · last ${d < 1 ? "today" : d < 2 ? "yesterday" : `${Math.round(d)} days ago`}`;
  return `<article class="rec ${hero ? "hero" : ""}" data-id="${r.outfit_id}">
    <div class="media"><img src="${r.crop_url}" alt="${esc(r.primary_category)}" loading="lazy"><span class="rank">${hero ? "Top pick" : `#${i + 1}`}</span></div>
    <div class="body">
      <div class="rec-top"><div><h3>${esc(r.primary_category)}</h3><p class="muted small" style="margin:4px 0 0">${last}</p></div>
        <div class="ring" style="--p:${matchPct(r)}" title="Match">${matchPct(r)}</div></div>
      <div class="meta">
        ${r.reasons.map((x) => `<span class="tag ok">${icon("check")}${esc(x)}</span>`).join("")}
        ${r.warnings.map((x) => `<span class="tag warn">${icon("alert")}${esc(x)}</span>`).join("")}
        ${hero ? `<span class="tag">${esc(r.fabric_tag)}</span>` : ""}
      </div>
      ${hero ? `<details class="why"><summary>Why this pick?</summary><dl>
        <dt>Style + context similarity</dt><dd>${b.similarity}</dd><dt>Freshness (anti-repeat)</dt><dd>× ${b.recency}</dd>
        <dt>Vibe / weather fit</dt><dd>× ${b.context_fit}</dd><dt>Venue radar</dt><dd>× ${b.venue_radar}</dd></dl></details>` : ""}
      <button class="btn ${hero ? "primary" : "ghost"} wear" data-id="${r.outfit_id}">${icon("check")}Wearing this</button>
    </div></article>`;
}

const emptyState = (title, text, cta = true) => `<div class="empty">${icon("sparkle")}<h3>${esc(title)}</h3><p class="muted">${esc(text)}</p>
  ${cta ? `<button class="btn primary" data-goto="import">${icon("plus")}Add outfits</button>` : ""}</div>`;
document.addEventListener("click", (e) => { const g = e.target.closest("[data-goto]"); if (g) go(g.dataset.goto); });

async function recommend() {
  $("#recs").innerHTML = `<div class="skeleton" style="min-height:420px"></div><div class="skeleton"></div><div class="skeleton"></div>`;
  try {
    const res = await api.recommendDay({ vibe, ...readCtx() });
    $("#context").textContent = res.results.length ? `Matched against: “${res.context}”` : "";
    $("#recs").innerHTML = res.results.length ? res.results.map(recCard).join("")
      : emptyState("Your wardrobe is empty", "Add a few outfit photos and we'll suggest what to wear, matched to the weather and your plans.");
  } catch (e) { $("#recs").innerHTML = emptyState("Something went wrong", e.message, false); }
}
$("#recs").addEventListener("click", async (e) => {
  const btn = e.target.closest("button.wear");
  if (btn) {
    e.stopPropagation();
    btn.disabled = true;
    await api.logWear(Number(btn.dataset.id), { venue: $("#venue").value || null });
    btn.innerHTML = `${icon("check")}Logged for today`;
    toast("Logged — we'll rotate it out for a while");
    return;
  }
  if (e.target.closest("details")) return;
  const c = e.target.closest(".rec");
  if (c) showDetail(Number(c.dataset.id));
});

// ---------------------------------------------------------------- wardrobe
const filt = { vibe: "", season: "" };
function renderFilters() {
  const chip = (g, v, label) => `<button class="filter ${filt[g] === v ? "active" : ""}" data-g="${g}" data-v="${esc(v)}">${label}</button>`;
  $("#filters").innerHTML = chip("vibe", "", "All")
    + api.VIBES.map((v) => chip("vibe", v, `${VIBE_EMOJI[v]} ${v}`)).join("")
    + api.SEASONS.map((s) => chip("season", s, s)).join("");
}
$("#filters").addEventListener("click", (e) => {
  const b = e.target.closest(".filter");
  if (!b) return;
  const { g, v } = b.dataset;
  if (v === "") { filt.vibe = ""; filt.season = ""; } else filt[g] = filt[g] === v ? "" : v;
  renderFilters();
  loadWardrobe();
});
let searchTimer;
$("#search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadWardrobe, 350); });

const tile = (o, match) => `<figure class="tile" data-id="${o.id}"><img src="${o.crop_url}" alt="" loading="lazy">
  ${match != null ? `<span class="match">${Math.round(match * 100)}%</span>` : ""}<span class="count">${o.wear_count}×</span>
  <figcaption>${esc(o.primary_category)}<span>${esc(o.vibe_tag)} · ${ago(o.last_worn_timestamp)}</span></figcaption></figure>`;

async function loadWardrobe() {
  const q = $("#search").value.trim();
  const items = await api.listOutfits({ q, vibe: filt.vibe, season: filt.season });
  $("#count-badge").textContent = `${items.length} look${items.length === 1 ? "" : "s"}`;
  $("#grid").innerHTML = items.length ? items.map((o) => tile(o, q ? o.match : null)).join("")
    : `<div style="grid-column:1/-1">${emptyState(q || filt.vibe || filt.season ? "No matches" : "No outfits yet",
      q || filt.vibe || filt.season ? "Try a different search or filter." : "Add photos of outfits you've worn.", !(q || filt.vibe || filt.season))}</div>`;
}
$("#grid").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(Number(t.dataset.id)); });

async function showDetail(id) {
  const o = await api.getOutfit(id);
  const opt = (list, cur) => list.map((x) => `<option ${x === cur ? "selected" : ""}>${esc(x)}</option>`).join("");
  const top = (o.labels.category || []).slice(0, 3).map(([l, s]) => `${esc(l)} ${(s * 100).toFixed(0)}`).join(" · ");
  openSheet(`<div class="detail">
    <div class="detail-media"><img src="${o.crop_url}" alt=""></div>
    <div>
      <h2>${esc(o.primary_category)}</h2>
      <div class="meta"><span class="tag">${VIBE_EMOJI[o.vibe_tag] || ""} ${esc(o.vibe_tag)}</span><span class="tag">${esc(o.season_tag)}</span>
        <span class="tag">${esc(o.color_tag)}</span><span class="tag">Worn ${o.wear_count}×</span></div>
      <div class="form">
        <label>Name<input id="d-name" value="${esc(o.primary_category)}"></label>
        <div class="two"><label>Vibe<select id="d-vibe">${opt(api.VIBES, o.vibe_tag)}</select></label>
        <label>Season<select id="d-season">${opt(api.SEASONS, o.season_tag)}</select></label></div>
        <label>Fabric<input id="d-fabric" value="${esc(o.fabric_tag)}"></label>
      </div>
      <p class="labels">AI guesses: ${top}</p>
      <h3>Wear history</h3>
      <ul class="timeline">${o.wears.map((w) => `<li><b>${new Date(w.worn_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}</b>
        <span>${w.venue ? esc(w.venue) + " · " : ""}${w.source === "manual" ? "logged" : "photo"}</span></li>`).join("")}</ul>
      <div class="sheet-actions"><button class="btn ghost danger" id="d-del">${icon("trash")}Remove</button><button class="btn primary" id="d-save">Save</button></div>
    </div></div>`);
  $("#d-save").onclick = async () => {
    try {
      await api.patchOutfit(id, { primary_category: $("#d-name").value, vibe_tag: $("#d-vibe").value,
        season_tag: $("#d-season").value, fabric_tag: $("#d-fabric").value });
      closeSheet(); toast("Saved"); refreshCurrent();
    } catch (e) { toast(e.message, "warn"); }
  };
  $("#d-del").onclick = async () => {
    closeSheet();
    if (!(await ask({ title: "Remove this outfit?", text: "It's removed from your wardrobe. Your original photos are not affected.", ok: "Remove", danger: true }))) return;
    await api.removeOutfit(id);
    toast("Outfit removed"); refreshStatus(); refreshCurrent();
  };
}
function refreshCurrent() {
  const v = $(".view.active").id.replace("view-", "");
  ({ today: recommend, wardrobe: loadWardrobe, collections: loadCollections })[v]?.();
}

// ---------------------------------------------------------------- collections
async function loadCollections() {
  $("#cols").innerHTML = `<div class="skeleton"></div>`;
  const [cols, items] = await Promise.all([api.collections(), api.listOutfits()]);
  const byId = Object.fromEntries(items.map((o) => [o.id, o]));
  $("#cols").innerHTML = cols.length ? cols.map((c) => `
    <div class="rail-head"><h3>${esc(c.name)}</h3><span>${c.outfit_ids.length} look${c.outfit_ids.length === 1 ? "" : "s"}${c.kind === "cluster" ? " · AI cluster" : ""}</span></div>
    <div class="rail">${c.outfit_ids.map((i) => byId[i]).filter(Boolean).map((o) => tile(o)).join("")}</div>`).join("")
    : emptyState("Collections appear automatically", "Once you have a few outfits we group them into Monsoon Office Wear, Ethnic & Festive, Seldom Worn Gems and more.");
}
$("#cols").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(Number(t.dataset.id)); });

// ---------------------------------------------------------------- import
$("#selfie").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("#selfie-label").textContent = "Checking…";
  try {
    await api.setSelfie(f);
    $("#selfie-thumb").src = URL.createObjectURL(f);
    $("#selfie-thumb").hidden = false;
    toast("Selfie saved — we'll find you in group photos");
  } catch (err) { toast(err.message, "warn"); }
  e.target.value = "";
  refreshStatus();
});

const STATUS = { new_outfit: "New", merged: "Repeat", skipped_no_user: "Skipped", error: "Error" };
async function ingest(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
  if (!files.length) return;
  const thumbs = new Map(files.map((f) => [f, URL.createObjectURL(f)]));
  $("#ingest").hidden = false;
  $("#ingest-title").textContent = "Analysing on this device…";
  $("#ingest-count").textContent = `0 / ${files.length}`;
  $("#ingest-bar").style.width = "0";
  $("#ingest-stats").textContent = "";
  $("#ingest-list").innerHTML = "";
  $("#dropzone").classList.add("busy");
  try {
    const res = await api.ingestFiles(files, ({ i, n, file, result: r }) => {
      $("#ingest-bar").style.width = `${(100 * i) / n}%`;
      $("#ingest-count").textContent = `${i} / ${n}`;
      const sub = r.status === "skipped_no_user" ? "You're not in this photo"
        : r.status === "error" ? r.detail : `${r.primary_category}${r.status === "merged" ? ` · same look (${(r.similarity * 100).toFixed(0)}%)` : ""}`;
      $("#ingest-list").insertAdjacentHTML("afterbegin", `<li><img src="${thumbs.get(file)}" alt="">
        <div class="txt"><b>${esc(file.name)}</b><span>${esc(sub)}</span></div><span class="pill-status ${r.status}">${STATUS[r.status] || r.status}</span></li>`);
    });
    const c = res.counts;
    $("#ingest-title").textContent = res.processed ? "Done" : "Nothing new — these photos were already added";
    const avg = (a) => (a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : "–");
    $("#ingest-stats").textContent = `${c.new_outfit || 0} new · ${c.merged || 0} repeats · ${c.skipped_no_user || 0} skipped`
      + `${c.error ? ` · ${c.error} errors` : ""} · ~${avg(res.timings.embed)} ms per photo`;
    if (c.new_outfit) toast(`${c.new_outfit} new outfit${c.new_outfit > 1 ? "s" : ""} added`);
  } catch (err) {
    $("#ingest-title").textContent = "Couldn't analyse photos";
    $("#ingest-stats").textContent = `${err.name}: ${err.message}`;
    toast(err.message, "warn");
  }
  $("#dropzone").classList.remove("busy");
  refreshStatus();
  await loadVenues();
  recommend(); // Today reflects the new outfits when the user switches back
}
$("#photos").addEventListener("change", (e) => { ingest(e.target.files); e.target.value = ""; });
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("over")));
dz.addEventListener("drop", (e) => { e.preventDefault(); ingest(e.dataTransfer.files); });

$("#export").addEventListener("click", async () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(await api.exportBackup());
  a.download = `smart-wardrobe-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  toast("Backup exported");
});
$("#import-btn").addEventListener("click", () => $("#import-backup").click());
$("#import-backup").addEventListener("change", async (e) => {
  try { await api.importBackup(e.target.files[0]); toast("Backup restored"); } catch (err) { toast(err.message, "warn"); }
  e.target.value = "";
  refreshStatus(); loadVenues();
});
$("#wipe").addEventListener("click", async () => {
  if (!(await ask({ title: "Delete everything?", text: "Every outfit, wear log and your selfie will be removed from this browser. This cannot be undone.", ok: "Delete all", danger: true }))) return;
  await api.wipeAll();
  localStorage.removeItem("ctx");
  location.reload();
});

// ---------------------------------------------------------------- venues
async function loadVenues(list) {
  const vs = list || (await api.venues());
  const cur = $("#venue").value || ctxState.venue || "";
  $("#venue").innerHTML = `<option value="">Anywhere</option>`
    + vs.map((v) => `<option value="${esc(v.cluster)}" data-label="${esc(v.label)}">${esc(v.label)} · ${v.wears}</option>`).join("");
  $("#venue").value = vs.some((v) => v.cluster === cur) ? cur : "";
  ctxSummary();
}
$("#rename-venue").addEventListener("click", async () => {
  const c = $("#venue").value;
  if (!c) return toast("Pick a venue first", "warn");
  const label = await ask({ title: "Name this place", text: "E.g. Office, Home, Parents' place. Used by the anti-repeat radar.", ok: "Save",
    input: $("#venue").selectedOptions[0]?.dataset.label || "" });
  if (label) { await loadVenues(await api.nameVenue(c, label)); toast("Venue renamed"); }
});

// ---------------------------------------------------------------- boot
async function refreshStatus() {
  const h = await api.health();
  $("#status").textContent = `${h.outfits} outfit${h.outfits === 1 ? "" : "s"} · ${h.reference_selfie ? "face anchored" : "no selfie yet"}`;
  if (h.reference_selfie) $("#selfie-label").textContent = "Replace selfie";
  return h;
}

(async () => {
  setGreeting();
  setVibe(vibe);
  renderFilters();
  if (ctxState.temp_c != null) $("#temp").value = ctxState.temp_c;
  if (ctxState.humidity != null) $("#humidity").value = ctxState.humidity;
  $("#rain").checked = !!ctxState.rain;
  navigator.storage?.persist?.(); // ask the browser not to evict the wardrobe
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
  const h = await refreshStatus();
  await loadVenues();
  if (!h.outfits) {
    $("#recs").innerHTML = emptyState("Welcome to your private wardrobe",
      "Add a selfie and some outfit photos. Everything is analysed on this device. Nothing is uploaded.");
    return;
  }
  await api.ready();
  recommend();
})();
