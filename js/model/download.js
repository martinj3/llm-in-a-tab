// Fetching model files, using nothing but the platform fetch API. No
// library does this better (see plan.md 1.6).

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

// Fetches byte range [start, end) -- end exclusive, HTTP Range is inclusive
// on both sides so we subtract 1. This is the primitive Phase 2 builds on:
// pull one tensor's bytes at a time instead of buffering the whole file
// (plan.md Phase 2 -- required for mobile, not an optimization).
export async function rangedFetch(url, start, end) {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (res.status !== 206) {
    throw new Error(
      `Expected 206 Partial Content for ranged fetch of ${url} [${start}, ${end}), got ${res.status}`
    );
  }
  return res.arrayBuffer();
}
