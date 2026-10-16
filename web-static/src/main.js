import * as api from "./app-api.js";

window.wardrobe = api; // console access for debugging / e2e tests

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
let vibe = "Work";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ago = (iso) => {
  if (!iso) return "never";
  const d = Math.round((Date.now() - new Date(iso)) / 864e5);
  return d <= 0 ? "today" : d === 1 ? "yesterday" : `${d} days ago`;
};
const banner = (html) => { $("#banner").innerHTML = html; };

// ---------------------------------------------------------------- model status
const files = {};
api.onModelProgress((p) => {
  if (p.status === "progress" && p.total) {
    files[p.file] = [p.loaded, p.total];
    const [l, t] = Object.values(files).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
    banner(`Loading model ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB (downloaded on first visit only, then cached)
      <div class="bar"><i style="width:${(100 * l / t).toFixed(1)}%"></i></div>`);
  } else if (p.status === "fallback") {
    console.warn("WebGPU failed, using WASM:", p.message);
  }
});
api.onStatus((m) => banner(m ? esc(m) : ""));

// ---------------------------------------------------------------- tabs
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
  ({ wardrobe: loadWardrobe, collections: loadCollections })[t.dataset.view]?.();
}));

// ---------------------------------------------------------------- today
$$(".pill").forEach((p) => p.addEventListener("click", () => {
  vibe = p.dataset.vibe;
  $$(".pill").forEach((x) => x.classList.toggle("active", x === p));
  recommend();
}));
$("#go").addEventListener("click", recommend);
$("#auto-weather").addEventListener("click", async () => {
  if (localStorage.getItem("weatherOk") !== "1") {
    if (!confirm("Auto weather sends your approximate location (rounded to ~11 km) to open-meteo.com. "
      + "Nothing else leaves this device. Allow?")) return;
    localStorage.setItem("weatherOk", "1");
  }
  try {
    const w = await api.weather();
    $("#temp").value = Math.round(w.temp_c);
    $("#humidity").value = Math.round(w.humidity);
    $("#rain").checked = w.rain;
    recommend();
  } catch (e) { alert(e.message); }
});

function card(r) {
  const b = r.breakdown;
  return `<article class="card" data-id="${r.outfit_id}">
    <img src="${r.crop_url}" alt="">
    <div class="body">
      <h4>${esc(r.primary_category)}</h4>
      <div class="tags"><span>${esc(r.vibe_tag)}</span><span>${esc(r.season_tag)}</span><span>${esc(r.fabric_tag)}</span></div>
      <p class="muted">Worn ${r.wear_count}× · last ${r.days_since_worn == null ? "never" : Math.round(r.days_since_worn) + "d ago"}</p>
      ${r.reasons.map((x) => `<p class="ok">✓ ${esc(x)}</p>`).join("")}
      ${r.warnings.map((x) => `<p class="warn">⚠ ${esc(x)}</p>`).join("")}
      <p class="muted small">score ${r.score.toFixed(3)} = sim ${b.similarity} × recency ${b.recency} × fit ${b.context_fit} × radar ${b.venue_radar}</p>
      <button class="wear primary" data-id="${r.outfit_id}">Wearing this today</button>
    </div></article>`;
}

async function recommend() {
  const num = (s) => ($(s).value === "" ? null : Number($(s).value));
  $("#recs").innerHTML = `<p class="muted">Thinking on-device…</p>`;
  try {
    const res = await api.recommendDay({ vibe, temp_c: num("#temp"), humidity: num("#humidity"),
      rain: $("#rain").checked, venue: $("#venue").value || null });
    $("#context").textContent = `Query: “${res.context}”`;
    $("#recs").innerHTML = res.results.length ? res.results.map(card).join("")
      : `<p class="muted">No outfits yet — add some photos in the Import tab.</p>`;
  } catch (e) { $("#recs").innerHTML = `<p class="warn">${esc(e.message)}</p>`; }
}

$("#recs").addEventListener("click", async (e) => {
  const btn = e.target.closest("button.wear");
  if (btn) {
    await api.logWear(Number(btn.dataset.id), { venue: $("#venue").value || null });
    btn.textContent = "Logged ✓";
    btn.disabled = true;
    return;
  }
  const c = e.target.closest(".card");
  if (c) showDetail(Number(c.dataset.id));
});

// ---------------------------------------------------------------- wardrobe
let searchTimer;
$("#search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadWardrobe, 350); });
$("#f-vibe").addEventListener("change", loadWardrobe);
$("#f-season").addEventListener("change", loadWardrobe);

async function loadWardrobe() {
  const items = await api.listOutfits({ q: $("#search").value.trim(), vibe: $("#f-vibe").value, season: $("#f-season").value });
  $("#grid").innerHTML = items.length ? items.map((o) => `
    <figure class="tile" data-id="${o.id}"><img src="${o.crop_url}" alt="">
      <figcaption>${esc(o.primary_category)}<br><span class="muted">${o.wear_count}× · ${ago(o.last_worn_timestamp)}</span></figcaption>
    </figure>`).join("") : `<p class="muted">Nothing here yet.</p>`;
}
$("#grid").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(Number(t.dataset.id)); });

async function showDetail(id) {
  const o = await api.getOutfit(id);
  const opt = (list, cur) => list.map((x) => `<option ${x === cur ? "selected" : ""}>${esc(x)}</option>`).join("");
  const top = (o.labels.category || []).slice(0, 3).map(([l, s]) => `${esc(l)} (${s.toFixed(2)})`).join(", ");
  $("#detail-body").innerHTML = `
    <img class="hero" src="${o.crop_url}" alt="">
    <label>Name <input id="d-name" value="${esc(o.primary_category)}"></label>
    <label>Vibe <select id="d-vibe">${opt(api.VIBES, o.vibe_tag)}</select></label>
    <label>Season <select id="d-season">${opt(api.SEASONS, o.season_tag)}</select></label>
    <label>Fabric <input id="d-fabric" value="${esc(o.fabric_tag)}"></label>
    <p class="muted small">Zero-shot: ${top}</p>
    <p class="muted small">Colour ${esc(o.color_tag)} · ${o.photos.length} photo(s)</p>
    <h4>Wear history</h4>
    <ul>${o.wears.map((w) => `<li>${esc(w.worn_at.replace("T", " "))} ${w.venue ? "· " + esc(w.venue) : ""} <span class="muted">${esc(w.source)}</span></li>`).join("")}</ul>
    <button type="button" id="d-save" class="primary">Save tags</button>
    <button type="button" id="d-del" class="ghost danger">Remove outfit</button>`;
  $("#d-save").onclick = async () => {
    await api.patchOutfit(id, { primary_category: $("#d-name").value, vibe_tag: $("#d-vibe").value,
      season_tag: $("#d-season").value, fabric_tag: $("#d-fabric").value });
    $("#detail").close();
    loadWardrobe();
  };
  $("#d-del").onclick = async () => {
    if (!confirm("Remove this outfit? Your original photos are not affected.")) return;
    await api.removeOutfit(id);
    $("#detail").close();
    loadWardrobe();
    refreshStatus();
  };
  $("#detail").showModal();
}

// ---------------------------------------------------------------- collections
async function loadCollections() {
  $("#cols").innerHTML = `<p class="muted">Clustering…</p>`;
  const [cols, items] = await Promise.all([api.collections(), api.listOutfits()]);
  const byId = Object.fromEntries(items.map((o) => [o.id, o]));
  $("#cols").innerHTML = cols.length ? cols.map((c) => `
    <h3>${esc(c.name)} <span class="muted small">${c.outfit_ids.length} · ${c.kind}</span></h3>
    <div class="row">${c.outfit_ids.map((i) => byId[i]).filter(Boolean).map((o) =>
      `<figure class="tile sm" data-id="${o.id}"><img src="${o.crop_url}" alt=""><figcaption>${esc(o.primary_category)}</figcaption></figure>`).join("")}</div>`).join("")
    : `<p class="muted">Collections appear once you have a few outfits.</p>`;
}
$("#cols").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) showDetail(Number(t.dataset.id)); });

// ---------------------------------------------------------------- import
const log = (s) => { $("#log").textContent = s; };
$("#selfie").addEventListener("change", async (e) => {
  try {
    await api.setSelfie(e.target.files[0]);
    $("#selfie-state").textContent = "Saved ✓";
  } catch (err) { $("#selfie-state").textContent = err.message; }
  refreshStatus();
});
$("#photos").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  const lines = [];
  log(`Preparing ${files.length} photo(s)…`);
  try {
    const res = await api.ingestFiles(files, ({ i, n, file, result: r }) => {
      $("#ingest-bar").style.width = `${(100 * i) / n}%`;
      lines.push(`${file.name}: ${r.status}${r.outfit_id ? " → #" + r.outfit_id + " " + (r.primary_category || "") : ""}`
        + `${r.similarity != null && r.status !== "skipped_no_user" ? " (sim " + r.similarity.toFixed(3) + ")" : ""}${r.detail ? " [" + r.detail + "]" : ""}`);
      log(`Analysing on-device… ${i}/${n}\n\n${lines.join("\n")}`);
    });
    const avg = (a) => (a.length ? (a.reduce((x, y) => x + y, 0) / a.length).toFixed(0) : "–");
    log(`Processed ${res.processed}: ${JSON.stringify(res.counts)}\n`
      + `mean ms: locate ${avg(res.timings.locate)}, embed ${avg(res.timings.embed)}, classify ${avg(res.timings.classify)}\n\n${lines.join("\n")}`);
  } catch (err) { log(err.stack || err.message); }
  e.target.value = "";
  refreshStatus();
  loadVenues();
});
$("#export").addEventListener("click", async () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(await api.exportBackup());
  a.download = `smart-wardrobe-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
});
$("#import-btn").addEventListener("click", () => $("#import-backup").click());
$("#import-backup").addEventListener("change", async (e) => {
  try { await api.importBackup(e.target.files[0]); log("Backup restored ✓"); } catch (err) { log(err.message); }
  refreshStatus();
  loadVenues();
});
$("#wipe").addEventListener("click", async () => {
  if (!confirm("Delete every outfit, wear log and your selfie from this browser? This cannot be undone.")) return;
  await api.wipeAll();
  location.reload();
});

// ---------------------------------------------------------------- venues
async function loadVenues(list) {
  const vs = list || (await api.venues());
  const cur = $("#venue").value;
  $("#venue").innerHTML = `<option value="">Anywhere</option>`
    + vs.map((v) => `<option value="${esc(v.cluster)}">${esc(v.label)} (${v.wears})</option>`).join("");
  $("#venue").value = cur;
}
$("#rename-venue").addEventListener("click", async () => {
  const c = $("#venue").value;
  if (!c) return alert("Pick a venue first.");
  const label = prompt("Name this place (e.g. Office, Home, Parents' place):");
  if (label) loadVenues(await api.nameVenue(c, label));
});

// ---------------------------------------------------------------- boot
async function refreshStatus() {
  const h = await api.health();
  $("#status").textContent = `${h.outfits} outfits · on-device${h.device ? " (" + h.device + ")" : ""} · ${h.reference_selfie ? "face anchored" : "no selfie"}`;
  $("#selfie-state").textContent ||= h.reference_selfie ? "Reference selfie set ✓" : "";
  return h;
}

(async () => {
  $("#f-vibe").innerHTML += api.VIBES.map((v) => `<option>${v}</option>`).join("");
  $("#f-season").innerHTML += api.SEASONS.map((v) => `<option>${v}</option>`).join("");
  const day = new Date().getDay();
  vibe = day === 0 || day === 6 ? "Casual" : "Work";
  $$(".pill").forEach((p) => p.classList.toggle("active", p.dataset.vibe === vibe));
  navigator.storage?.persist?.(); // ask the browser not to evict the wardrobe
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
  const h = await refreshStatus();
  await loadVenues();
  if (!h.outfits) {
    banner("Welcome! Your photos stay on this device. Go to <b>Import</b> to add a selfie and some outfit photos.");
    $$(".tab").find((t) => t.dataset.view === "import").click();
    return;
  }
  await api.ready();
  refreshStatus();
  recommend();
})();
