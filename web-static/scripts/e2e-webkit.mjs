// iPhone (WebKit) smoke test: every iOS browser, including Chrome, runs on WebKit.
//   npm run build && npx vite preview --host 127.0.0.1 --port 4173 & node scripts/e2e-webkit.mjs [url]
import { webkit, devices } from "playwright";
const browser = await webkit.launch();
const ctx = await browser.newContext({ ...devices["iPhone 15"] });
const p = await ctx.newPage();
const logs = [];
p.on("console", (m) => logs.push(`[${m.type()}] ${m.text().slice(0, 400)}`));
p.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
p.on("crash", () => logs.push("[CRASH]"));
p.on("requestfailed", (r) => logs.push(`[reqfail] ${r.url().slice(0, 120)} ${r.failure()?.errorText}`));
p.on("response", (r) => r.status() >= 400 && logs.push(`[http ${r.status()}] ${r.url()}`));
const t0 = Date.now();
await p.goto(process.argv[2] || "http://127.0.0.1:4173/", { waitUntil: "domcontentloaded" });
console.log("dom ready", Date.now() - t0);
await p.waitForTimeout(3000);
await p.setInputFiles("#photos", [process.env.PHOTO || "/tmp/e2e_photos/portrait.jpg"]);
let last = "";
for (let i = 0; i < 120; i++) {
  await p.waitForTimeout(5000);
  const s = await p.evaluate(() => (document.querySelector("#model-card").hidden ? "" : document.querySelector("#model-text").textContent.slice(0, 100)) + " || " + document.querySelector("#ingest-title").textContent + " " + document.querySelector("#ingest-stats").textContent + " " + document.querySelector("#ingest-list").innerText.slice(0, 200)).catch((e) => "EVAL FAIL " + e.message);
  if (s !== last) { console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s: ${s}`); last = s; }
  if (/Done|Couldn|Nothing|EVAL FAIL/.test(s)) break;
}
console.log("CONSOLE:\n" + logs.filter((l) => !/XNNPACK|gl_context|Graph successfully|feedback|NORM_RECT/.test(l)).slice(-25).join("\n"));
await browser.close();
const ok = /Done/.test(last);
console.log(ok ? "WEBKIT E2E: PASS" : "WEBKIT E2E: FAIL");
process.exit(ok ? 0 : 1);
