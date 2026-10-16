// IndexedDB persistence. Everything stays in this browser profile.
const DB_NAME = "smart-wardrobe";
const VERSION = 1;
let dbp;

function open() {
  return (dbp ||= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore("outfits", { keyPath: "id", autoIncrement: true });
      db.createObjectStore("wears", { keyPath: "id", autoIncrement: true }).createIndex("outfit_id", "outfit_id");
      db.createObjectStore("photos", { keyPath: "key" });
      db.createObjectStore("regions", { keyPath: "id", autoIncrement: true }).createIndex("outfit_id", "outfit_id");
      db.createObjectStore("venues", { keyPath: "cluster" });
      db.createObjectStore("meta", { keyPath: "key" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

const wrap = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });

export async function all(store) {
  const db = await open();
  return wrap(db.transaction(store).objectStore(store).getAll());
}
export async function get(store, key) {
  const db = await open();
  return wrap(db.transaction(store).objectStore(store).get(key));
}
export async function put(store, value) {
  const db = await open();
  return wrap(db.transaction(store, "readwrite").objectStore(store).put(value));
}
export async function del(store, key) {
  const db = await open();
  return wrap(db.transaction(store, "readwrite").objectStore(store).delete(key));
}
/** Run fn(tx) in one readwrite transaction over several stores; resolves on commit. */
export async function tx(stores, fn) {
  const db = await open();
  const t = db.transaction(stores, "readwrite");
  const done = new Promise((res, rej) => { t.oncomplete = res; t.onerror = () => rej(t.error); t.onabort = () => rej(t.error); });
  const result = await fn(t);
  await done;
  return result;
}
export const req = wrap;

export async function deleteOutfit(id) {
  await tx(["outfits", "wears", "regions", "photos"], async (t) => {
    t.objectStore("outfits").delete(id);
    for (const s of ["wears", "regions"]) {
      const keys = await wrap(t.objectStore(s).index("outfit_id").getAllKeys(id));
      keys.forEach((k) => t.objectStore(s).delete(k));
    }
    const photos = await wrap(t.objectStore("photos").getAll());
    photos.filter((p) => p.outfit_id === id).forEach((p) => t.objectStore("photos").put({ ...p, outfit_id: null, status: "deleted" }));
  });
}

export async function wipe() {
  (await open()).close();
  dbp = null;
  await wrap(indexedDB.deleteDatabase(DB_NAME));
}
