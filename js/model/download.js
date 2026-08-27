// Fetching model files with progress, using nothing but the platform fetch
// API and ReadableStream. No library does this better (see plan.md 1.6).

export async function headContentLength(url) {
  const res = await fetch(url, { method: "HEAD" });
  if (!res.ok) throw new Error(`HEAD ${url} failed: ${res.status}`);
  const len = res.headers.get("content-length");
  return len ? parseInt(len, 10) : null;
}

export async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json();
}

// Downloads the full response body, reporting progress as it streams in.
// Returns { buffer, byteLength }. This buffers the whole file in memory --
// fine for Phase 1. Phase 2 replaces this with per-tensor ranged fetches so
// large models don't peak at 2x their size on mobile (plan.md Phase 2).
export async function downloadWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);

  const totalHeader = res.headers.get("content-length");
  const total = totalHeader ? parseInt(totalHeader, 10) : NaN;

  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (onProgress) onProgress(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return { buffer: buffer.buffer, byteLength: loaded };
}
