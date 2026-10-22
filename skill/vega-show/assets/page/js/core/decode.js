// Weight tensors as the service sends them: base64 little-endian float32 (`f32`), or int8 with a
// scale (`int8`, `scale`), with `shape: [out, in]`. Each is decoded once and shared by every
// drawing (the WeakMap is this module's only state, allowed by the page lint).

const DECODED = new WeakMap();

// Bytes of a base64 string (atob in the browser; Buffer under node, for the unit tests).
function bytesOf(b64) {
  if (typeof atob !== "function") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// {out, in, w: Float32Array (row-major, out × in)} of one weight record.
export function decode(w) {
  if (DECODED.has(w)) return DECODED.get(w);
  const bytes = bytesOf(w.f32 || w.int8);
  let f;
  if (w.f32) f = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);   // full precision
  else { f = new Float32Array(bytes.length); for (let i = 0; i < bytes.length; i++) f[i] = ((bytes[i] << 24) >> 24) * w.scale; }
  const m = { out: w.shape[0], in: w.shape[1], w: f };
  DECODED.set(w, m);
  return m;
}
