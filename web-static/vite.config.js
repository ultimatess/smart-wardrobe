import { cpSync } from "node:fs";
import { defineConfig } from "vite";

// Relative base so the build works at https://<user>.github.io/<repo>/ (or any sub-path).
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  plugins: [{
    name: "copy-mediapipe-wasm",
    buildStart() {
      cpSync("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe", { recursive: true });
    },
  }],
});
