// Main-thread client for embed.worker.js
let worker, seq = 0;
const pending = new Map();
const listeners = new Set();

function call(op, args = [], transfer = []) {
  worker ||= start();
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args }, transfer);
  });
}
function start() {
  const w = new Worker(new URL("./embed.worker.js", import.meta.url), { type: "module" });
  w.onmessage = ({ data }) => {
    if (data.progress) return listeners.forEach((f) => f(data.progress));
    const p = pending.get(data.id);
    pending.delete(data.id);
    data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
  };
  return w;
}

export const onModelProgress = (f) => listeners.add(f);
export const initModel = () => call("init");
export const embedTexts = (texts, task = "classification") => call("texts", [texts, task]);
export const embedImageData = (imageData) =>
  call("image", [imageData.data, imageData.width, imageData.height]);
