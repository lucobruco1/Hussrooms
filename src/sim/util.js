/** Small shared helpers. No dependencies, usable in Node and the browser. */

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function dist(a, b) {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

export function dist2(a, b) {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return dx * dx + dz * dz;
}

export function softmax(logits, out) {
  const n = logits.length;
  const o = out || new Float32Array(n);
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > m) m = logits[i];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - m);
    o[i] = e;
    s += e;
  }
  for (let i = 0; i < n; i++) o[i] /= s;
  return o;
}

export function sampleFrom(probs, rng = Math.random) {
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

export function argmax(arr) {
  let bi = 0;
  let bv = -Infinity;
  for (let i = 0; i < arr.length; i++)
    if (arr[i] > bv) {
      bv = arr[i];
      bi = i;
    }
  return bi;
}

export function mean(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

/** Deterministic 4-decimal string, for the tiny JSON checkpoints. */
export function q(v, d = 4) {
  const f = Math.pow(10, d);
  return Math.round(v * f) / f;
}

export function randn(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Base64 <-> float arrays, so checkpoints stay text and diffable-ish.
 *  Params live in Float64 for training accuracy but are stored as Float32 to
 *  halve checkpoint size (the cast is lossless enough for 1e-7 weights). */
export function f64ToBase64(arr) {
  const f32 = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) f32[i] = arr[i];
  return f32ToBase64(f32);
}

export function base64ToF64(b64) {
  const f32 = base64ToF32(b64);
  const out = new Float64Array(f32.length);
  for (let i = 0; i < f32.length; i++) out[i] = f32[i];
  return out;
}

export function f32ToBase64(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let s = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  if (typeof btoa === 'function') return btoa(s);
  return Buffer.from(s, 'binary').toString('base64');
}

export function base64ToF32(b64) {
  let bin;
  if (typeof atob === 'function') {
    const s = atob(b64);
    bin = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bin[i] = s.charCodeAt(i);
  } else {
    bin = new Uint8Array(Buffer.from(b64, 'base64'));
  }
  return new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);
}
