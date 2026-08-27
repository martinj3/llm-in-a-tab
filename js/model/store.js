// IndexedDB persistence for downloaded model bytes.
//
// One object store, "models", keyed by model id. Each record holds the raw
// bytes for a given model plus enough metadata (content-length) to verify a
// cache hit without re-downloading. Phase 2 will add a second field for the
// quantized tensors; this file stays generic on purpose.

const DB_NAME = "llm-in-a-tab";
const DB_VERSION = 1;
const STORE_NAME = "models";

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode) {
  return db.transaction(STORE_NAME, mode).objectStore(STORE_NAME);
}

// record shape: { id, contentLength, bytes: ArrayBuffer, downloadedAt }
export async function getModelRecord(modelId) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = tx(db, "readonly").get(modelId);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function putModelRecord(record) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = tx(db, "readwrite").put(record);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function deleteModelRecord(modelId) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = tx(db, "readwrite").delete(modelId);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}
