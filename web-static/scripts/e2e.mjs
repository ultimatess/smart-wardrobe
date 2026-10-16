// Headless-Chrome smoke test of the built site: ingest sample photos, recommend, persist across reload.
//   npm run build && node scripts/e2e.mjs <dir-with-photos> [--chrome PATH]
import { spawn } from "node:child_process";
import { readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import puppeteer from "puppeteer-core";

const dir = resolve(process.argv[2] || "test/photos");
const chrome = process.argv.includes("--chrome") ? process.argv[process.argv.indexOf("--chrome") + 1]
  : "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const selfie = process.argv.includes("--selfie") ? resolve(process.argv[process.argv.indexOf("--selfie") + 1]) : null;
const photos = readdirSync(dir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).map((f) => join(dir, f));

const server = spawn("npx", ["vite", "preview", "--host", "127.0.0.1", "--port", "4173", "--strictPort"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 4000));
const browser = await puppeteer.launch({ executablePath: chrome, headless: "new",
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--no-first-run"] });
let failed = false;
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 900 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("http://127.0.0.1:4173/", { waitUntil: "load" });
  console.log("webgpu:", await page.evaluate(async () => !!(navigator.gpu && (await navigator.gpu.requestAdapter()))));

  const t0 = Date.now();
  await page.click('.nav-btn[data-view="import"]');
  if (selfie) {
    await (await page.$("#selfie")).uploadFile(selfie);
    await page.waitForFunction(() => !document.querySelector("#selfie-thumb").hidden || /toast warn/.test(document.querySelector("#toasts").innerHTML), { timeout: 20 * 60e3 });
    const ok = await page.$eval("#selfie-thumb", (e) => !e.hidden);
    console.log("--- selfie:", ok ? "saved" : await page.$eval("#toasts", (e) => e.textContent));
    if (!ok) failed = true;
  }
  await (await page.$("#photos")).uploadFile(...photos);
  await page.waitForFunction(() => /^(Done|Couldn|Nothing)/.test(document.querySelector("#ingest-title").textContent), { timeout: 20 * 60e3 });
  const title = await page.$eval("#ingest-title", (e) => e.textContent);
  const rows = await page.$$eval("#ingest-list li", (li) => li.map((x) => x.innerText.replace(/\n/g, " | ")));
  console.log(`--- ingest (${((Date.now() - t0) / 1000).toFixed(0)} s incl. first-visit model download): ${title}\n`
    + `${await page.$eval("#ingest-stats", (e) => e.textContent)}\n${rows.join("\n")}`);
  await page.screenshot({ path: "/tmp/web_static_import.png" });
  if (title !== "Done") failed = true;

  await page.click('.nav-btn[data-view="today"]');
  await page.click('#vibes button[data-vibe="Work"]');
  await page.waitForSelector("#recs .rec", { timeout: 120e3 });
  const recs = await page.$$eval("#recs .rec h3", (h) => h.map((x) => x.textContent));
  console.log("--- top picks:", recs);
  await page.screenshot({ path: "/tmp/web_static_today.png" });

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => /\d+ outfits?/.test(document.querySelector("#status").textContent));
  console.log("--- after reload:", await page.$eval("#status", (e) => e.textContent));
  await page.waitForSelector("#recs .rec", { timeout: 120e3 });
  await page.click('.nav-btn[data-view="wardrobe"]');
  await page.waitForSelector("#grid .tile");
  await page.screenshot({ path: "/tmp/web_static_wardrobe.png" });
  await page.click("#grid .tile");
  await page.waitForSelector("#sheet[open] .detail");
  await new Promise((r) => setTimeout(r, 400));
  await page.screenshot({ path: "/tmp/web_static_detail.png" });
  await page.click(".sheet-close");
  // mobile layout
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.click('.tab-btn[data-view="today"]');
  await page.waitForSelector("#recs .rec", { timeout: 120e3 });
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: "/tmp/web_static_mobile_today.png" });
  await page.click('.tab-btn[data-view="wardrobe"]');
  await page.waitForSelector("#grid .tile");
  await page.screenshot({ path: "/tmp/web_static_mobile_wardrobe.png" });
  await page.click("#grid .tile");
  await page.waitForSelector("#sheet[open]");
  await new Promise((r) => setTimeout(r, 400));
  await page.screenshot({ path: "/tmp/web_static_mobile_detail.png" });
  if (errors.length) { console.log("--- page errors:\n" + errors.join("\n")); }
} catch (e) {
  failed = true;
  console.error("E2E FAILED:", e.message);
} finally {
  await browser.close();
  server.kill();
}
console.log(failed ? "E2E: FAIL" : "E2E: PASS");
process.exit(failed ? 1 : 0);
