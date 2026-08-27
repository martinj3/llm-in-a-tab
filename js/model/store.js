// IndexedDB persistence for quantized model tensors.
//
// Two object stores:
//   "tensors"   one record per tensor, keyed by "modelId:dtype:tensorName".
//               Splitting by tensor (instead of one whole-model blob) keeps
//               individual records small and lets Phase 2's per-tensor
//               loader write results as they arrive.
//   "manifests" one record per (modelId, dtype), listing every tensor name
//               written and the source file's content-length at download
//               time. Its presence with complete:true is what lets a
//               reload skip the network entirely.

const DB_NAME = "llm-in-a-tab";
const DB_VERSION = 2;
const TENSOR_STORE = "tensors";
const MANIFEST_STORE = "manifests";

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (db.objectStoreNames.contains("models")) {
        db.deleteObjectStore("models"); // superseded by per-tensor storage
      }
      if (!db.objectStoreNames.contains(TENSOR_STORE)) {
        db.createObjectStore(TENSOR_STORE);
      }
      if (!db.objectStoreNames.contains(MANIFEST_STORE)) {
        db.createObjectStore(MANIFEST_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function manifestKey(modelId, dtype) {
  return `${modelId}:${dtype}`;
}

function tensorKey(modelId, dtype, name) {
  return `${modelId}:${dtype}:${name}`;
}

export async function getManifest(modelId, dtype) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(MANIFEST_STORE, "readonly").objectStore(MANIFEST_STORE).get(manifestKey(modelId, dtype));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function putManifest(modelId, dtype, manifest) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(MANIFEST_STORE, "readwrite").objectStore(MANIFEST_STORE).put(manifest, manifestKey(modelId, dtype));
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// record shape: { kind: 'i8', shape, qweight: Int8Array, scales: Float32Array }
//            or { kind: 'f32', shape, f32: Float32Array }
export async function putTensor(modelId, dtype, name, record) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(TENSOR_STORE, "readwrite").objectStore(TENSOR_STORE).put(record, tensorKey(modelId, dtype, name));
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function getTensor(modelId, dtype, name) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(TENSOR_STORE, "readonly").objectStore(TENSOR_STORE).get(tensorKey(modelId, dtype, name));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// Reads every named tensor back into a Map<name, record> -- the shape
// transformer.js expects. One readonly transaction with all the gets
// queued up front, rather than getTensor() in a loop: each await there
// would let the transaction auto-close and force a fresh one per tensor,
// which for ~270 tensors is measurably slower for no reason.
//
// onProgress(done, total, name) fires as records arrive, since reading
// ~135MB back out of IndexedDB is slow enough to want a progress bar.
export async function getAllTensors(modelId, dtype, names, onProgress) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(TENSOR_STORE, "readonly");
    const objectStore = tx.objectStore(TENSOR_STORE);
    const tensors = new Map();
    let done = 0;
    for (const name of names) {
      const req = objectStore.get(tensorKey(modelId, dtype, name));
      req.onsuccess = () => {
        if (req.result === undefined) {
          reject(new Error(`Tensor missing from cache: ${name}`));
          return;
        }
        tensors.set(name, req.result);
        done++;
        if (onProgress) onProgress(done, names.length, name);
      };
    }
    tx.oncomplete = () => resolve(tensors);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
