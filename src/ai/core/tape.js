/**
 * A very small reverse-mode autodiff engine over Float32Array tensors.
 *
 * Why hand-rolled? Hussrooms has to train a MuZero-family agent inside a
 * browser tab with no GPU, no PyTorch, no ONNX runtime and no build step.
 * Everything is batched 2-D ([batch, features]) plus a couple of 1-D helpers,
 * which is all EfficientZero's networks need - convolutions are unnecessary
 * here because the representation MLP eats a flattened egocentric patch.
 *
 * Correctness is checked by tools/test-grad.mjs (finite differences).
 */

const activeTapes = [];

/**
 * Inference fast path. With gradient recording disabled the ops skip building
 * their backward closures entirely, which roughly halves the cost of the
 * hundreds of tiny forward passes MCTS makes every second.
 */
let gradEnabled = true;

export function setGradEnabled(v) {
  gradEnabled = !!v;
}

export function isGradEnabled() {
  return gradEnabled;
}

function needsGrad(a, b) {
  if (!gradEnabled) return false;
  return !!((a && a.req) || (b && b.req));
}

export function tapeActive() {
  return activeTapes.length > 0;
}

export function beginTape() {
  const tape = [];
  activeTapes.push(tape);
  return tape;
}

export function endTape() {
  return activeTapes.pop();
}

export class T {
  constructor(data, shape, op = 'leaf') {
    this.data = data;
    this.shape = shape;
    this.op = op;
    this.grad = null;
    this._back = null;
    this.parents = null;
    this.req = false;
    if (tapeActive()) activeTapes[activeTapes.length - 1].push(this);
  }
  get rows() {
    return this.shape[0] ?? 1;
  }
  get cols() {
    return this.shape.length > 1 ? this.shape[1] : this.shape[0];
  }
  item() {
    return this.data[0];
  }
}

export function param(data, shape) {
  const t = new T(data, shape, 'param');
  t.req = true;
  return t;
}

export function zeros(shape) {
  let n = 1;
  for (const s of shape) n *= s;
  return new Float64Array(n);
}

function gain(t, size) {
  if (!t.req) return null;
  if (!t.grad) t.grad = new Float64Array(size);
  return t.grad;
}

export function addGrad(t, g, size) {
  if (!t.req) return;
  if (!t.grad) t.grad = new Float64Array(size);
  if (size === undefined) size = t.data.length;
  const d = t.grad;
  for (let i = 0; i < size; i++) d[i] += g[i];
}

export function backward(loss, tape) {
  const tr = tape || activeTapes.length;
  const nodes = Array.isArray(tr) ? tr : activeTapes[activeTapes.length - 1];
  loss.grad = new Float64Array(loss.data.length).fill(1);
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (!n._back || !n.grad) continue;
    n._back(n.grad);
    n._back = null; // free the closure, keep memory flat
  }
}

export function clearGrads(nodes) {
  for (const n of nodes) n.grad = null;
}

// --------------------------------------------------------------------- ops

/** [N,K] x [K,M] -> [N,M] */
export function matmul(a, b) {
  const N = a.shape[0];
  const K = a.shape[1];
  const M = b.shape[1];
  const out = new Float64Array(N * M);
  const A = a.data;
  const B = b.data;
  for (let n = 0; n < N; n++) {
    const aOff = n * K;
    const oOff = n * M;
    for (let k = 0; k < K; k++) {
      const av = A[aOff + k];
      if (av === 0) continue;
      const bOff = k * M;
      for (let m = 0; m < M; m++) out[oOff + m] += av * B[bOff + m];
    }
  }
  const t = new T(out, [N, M], 'matmul');
  t.req = needsGrad(a, b);
  if (t.req) {
    t._back = (g) => {
      if (a.req) {
        const ga = new Float64Array(N * K);
        for (let n = 0; n < N; n++) {
          const aOff = n * K;
          const oOff = n * M;
          for (let m = 0; m < M; m++) {
            const gv = g[oOff + m];
            if (gv === 0) continue;
            for (let k = 0; k < K; k++) ga[aOff + k] += gv * B[k * M + m];
          }
        }
        addGrad(a, ga, N * K);
      }
      if (b.req) {
        const gb = new Float64Array(K * M);
        for (let n = 0; n < N; n++) {
          const aOff = n * K;
          const oOff = n * M;
          for (let k = 0; k < K; k++) {
            const av = A[aOff + k];
            if (av === 0) continue;
            const bOff = k * M;
            for (let m = 0; m < M; m++) gb[bOff + m] += av * g[oOff + m];
          }
        }
        addGrad(b, gb, K * M);
      }
    };
  }
  return t;
}

/** Broadcast add: [N,M] + [M] (bias) or [N,M] + [N,M]. */
export function add(a, b) {
  const isBias = b.shape.length === 1;
  const N = a.shape[0];
  const M = a.shape[1];
  const out = new Float64Array(N * M);
  for (let n = 0; n < N; n++) {
    for (let m = 0; m < M; m++) {
      out[n * M + m] = a.data[n * M + m] + (isBias ? b.data[m] : b.data[n * M + m]);
    }
  }
  const t = new T(out, [N, M], 'add');
  t.req = needsGrad(a, b);
  if (t.req) {
    t._back = (g) => {
      if (a.req) addGrad(a, g, N * M);
      if (b.req) {
        if (isBias) {
          const gb = b.grad || (b.grad = new Float64Array(M));
          for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) gb[m] += g[n * M + m];
        } else addGrad(b, g, N * M);
      }
    };
  }
  return t;
}

export function scale(a, s) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * s;
  const t = new T(out, a.shape.slice(), 'scale');
  t.req = needsGrad(a);
  if (t.req)
    t._back = (g) => {
      const ga = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) ga[i] = g[i] * s;
      addGrad(a, ga, g.length);
    };
  return t;
}

export function addInto(a, b, wa = 1, wb = 1) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * wa + b.data[i] * wb;
  const t = new T(out, a.shape.slice(), 'addInto');
  t.req = needsGrad(a, b);
  if (t.req)
    t._back = (g) => {
      if (a.req) {
        const ga = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) ga[i] = g[i] * wa;
        addGrad(a, ga, g.length);
      }
      if (b.req) {
        const gb = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) gb[i] = g[i] * wb;
        addGrad(b, gb, g.length);
      }
    };
  return t;
}

export function relu(a) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] > 0 ? a.data[i] : 0;
  const t = new T(out, a.shape.slice(), 'relu');
  t.req = needsGrad(a);
  if (t.req)
    t._back = (g) => {
      const ga = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) ga[i] = a.data[i] > 0 ? g[i] : 0;
      addGrad(a, ga, g.length);
    };
  return t;
}

export function tanh(a) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.tanh(a.data[i]);
  const t = new T(out, a.shape.slice(), 'tanh');
  t.req = needsGrad(a);
  if (t.req)
    t._back = (g) => {
      const ga = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) ga[i] = g[i] * (1 - out[i] * out[i]);
      addGrad(a, ga, g.length);
    };
  return t;
}

/** Layer norm over the last dimension, no learnable gain (keeps things small). */
export function layerNorm(a, eps = 1e-5) {
  const N = a.shape[0];
  const D = a.shape[1];
  const out = new Float64Array(N * D);
  const means = new Float64Array(N);
  const invs = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    let m = 0;
    for (let d = 0; d < D; d++) m += a.data[n * D + d];
    m /= D;
    let v = 0;
    for (let d = 0; d < D; d++) {
      const x = a.data[n * D + d] - m;
      v += x * x;
    }
    v /= D;
    const inv = 1 / Math.sqrt(v + eps);
    means[n] = m;
    invs[n] = inv;
    for (let d = 0; d < D; d++) out[n * D + d] = (a.data[n * D + d] - m) * inv;
  }
  const t = new T(out, [N, D], 'layernorm');
  t.req = needsGrad(a);
  if (t.req)
    t._back = (g) => {
      const ga = new Float64Array(N * D);
      for (let n = 0; n < N; n++) {
        const inv = invs[n];
        let sumG = 0;
        let sumGy = 0;
        for (let d = 0; d < D; d++) {
          sumG += g[n * D + d];
          sumGy += g[n * D + d] * out[n * D + d];
        }
        for (let d = 0; d < D; d++) {
          const y = out[n * D + d];
          ga[n * D + d] = (inv / D) * (D * g[n * D + d] - sumG - y * sumGy);
        }
      }
      addGrad(a, ga, N * D);
    };
  return t;
}

/** Concatenate along the last dimension. */
export function concat(nodes) {
  const N = nodes[0].shape[0];
  let D = 0;
  for (const n of nodes) D += n.shape.length > 1 ? n.shape[1] : 1;
  const out = new Float64Array(N * D);
  let off = 0;
  const widths = [];
  for (const n of nodes) {
    const w = n.shape.length > 1 ? n.shape[1] : 1;
    widths.push(w);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < w; j++) out[i * D + off + j] = n.data[i * w + j];
    }
    off += w;
  }
  const t = new T(out, [N, D], 'concat');
  t.req = gradEnabled && nodes.some((n) => n.req);
  if (t.req)
    t._back = (g) => {
      let o = 0;
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const w = widths[i];
        if (n.req) {
          const gn = new Float64Array(N * w);
          for (let r = 0; r < N; r++)
            for (let j = 0; j < w; j++) gn[r * w + j] += g[r * D + o + j];
          addGrad(n, gn, N * w);
        }
        o += w;
      }
    };
  return t;
}

export function slice(a, start, end) {
  const N = a.shape[0];
  const D = a.shape[1];
  const W = end - start;
  const out = new Float64Array(N * W);
  for (let n = 0; n < N; n++) for (let j = 0; j < W; j++) out[n * W + j] = a.data[n * D + start + j];
  const t = new T(out, [N, W], 'slice');
  t.req = needsGrad(a);
  if (t.req)
    t._back = (g) => {
      const ga = new Float64Array(N * D);
      for (let n = 0; n < N; n++) for (let j = 0; j < W; j++) ga[n * D + start + j] += g[n * W + j];
      addGrad(a, ga, N * D);
    };
  return t;
}

export function detach(a) {
  const t = new T(a.data, a.shape.slice(), 'detach');
  t.req = false;
  return t;
}

/** Symmetric log-space categorical head helper: logits -> 2*B+1 bin values. */
// --------------------------------------------------------------- losses

/** Soft cross entropy against a target distribution. Returns a scalar node. */
export function softmaxCrossEntropy(logits, targetProbs, weight = 1) {
  const N = logits.shape[0];
  const K = logits.shape[1];
  const probs = new Float64Array(N * K);
  let loss = 0;
  for (let n = 0; n < N; n++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) m = Math.max(m, logits.data[n * K + k]);
    let s = 0;
    for (let k = 0; k < K; k++) {
      const e = Math.exp(logits.data[n * K + k] - m);
      probs[n * K + k] = e;
      s += e;
    }
    for (let k = 0; k < K; k++) {
      probs[n * K + k] /= s;
      const p = probs[n * K + k];
      const q = targetProbs[n * K + k];
      if (q > 1e-9) loss -= q * Math.log(Math.max(p, 1e-9));
    }
  }
  loss = (loss / N) * weight;
  const t = new T(new Float64Array([loss]), [1, 1], 'softmaxCE');
  t.req = needsGrad(logits);
  if (t.req)
    t._back = (g) => {
      const gl = new Float64Array(N * K);
      const gs = (g[0] * weight) / N;
      for (let i = 0; i < N * K; i++) gl[i] = (probs[i] - targetProbs[i]) * gs;
      addGrad(logits, gl, N * K);
    };
  return t;
}

/** Smooth-L1 (Huber) regression. EfficientZero uses MSE; Huber is friendlier
 *  to the occasional huge Monte-Carlo value spike we get in a horror level. */
export function huber(pred, target, weight = 1, delta = 1.0) {
  const n = pred.data.length;
  let loss = 0;
  const coeff = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = pred.data[i] - target.data[i];
    const ad = Math.abs(d);
    if (ad <= delta) {
      loss += 0.5 * d * d;
      coeff[i] = d;
    } else {
      loss += delta * (ad - 0.5 * delta);
      coeff[i] = delta * Math.sign(d);
    }
  }
  loss = (loss / n) * weight;
  const t = new T(new Float64Array([loss]), [1, 1], 'huber');
  t.req = needsGrad(pred);
  if (t.req)
    t._back = (g) => {
      const gp = new Float64Array(n);
      const gs = (g[0] * weight) / n;
      for (let i = 0; i < n; i++) gp[i] = coeff[i] * gs;
      addGrad(pred, gp, n);
    };
  return t;
}

/** -cos(pred, stopGrad(target)) averaged over the batch: the SimSiam-style
 *  self-supervised latent consistency loss from EfficientZero. */
export function negCosine(pred, target) {
  const N = pred.shape[0];
  const D = pred.shape[1];
  const cos = new Float64Array(N);
  const predNorm = new Float64Array(N);
  const tgtNorm = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    let a = 0;
    let b = 0;
    let ab = 0;
    for (let d = 0; d < D; d++) {
      const x = pred.data[n * D + d];
      const y = target.data[n * D + d];
      a += x * x;
      b += y * y;
      ab += x * y;
    }
    const na = Math.sqrt(a) + 1e-8;
    const nb = Math.sqrt(b) + 1e-8;
    predNorm[n] = na;
    tgtNorm[n] = nb;
    cos[n] = ab / (na * nb);
  }
  let loss = 0;
  for (let n = 0; n < N; n++) loss -= cos[n];
  loss /= N;
  const t = new T(new Float64Array([loss]), [1, 1], 'negCos');
  t.req = needsGrad(pred);
  if (t.req)
    t._back = (g) => {
      const gp = new Float64Array(N * D);
      const gs = g[0] / N;
      for (let n = 0; n < N; n++) {
        const na = predNorm[n];
        const nb = tgtNorm[n];
        const c = cos[n];
        for (let d = 0; d < D; d++) {
          const x = pred.data[n * D + d];
          const y = target.data[n * D + d];
          // d(-cos)/dx
          gp[n * D + d] = -gs * ((y / (na * nb)) - (c * x) / (na * na));
        }
      }
      addGrad(pred, gp, N * D);
    };
  return t;
}

/** L2 weight regularisation over a list of params. */
export function l2(params, weight) {
  let loss = 0;
  const total = params.reduce((s, p) => s + p.data.length, 0);
  for (const p of params) for (let i = 0; i < p.data.length; i++) loss += p.data[i] * p.data[i];
  loss = (0.5 * weight * loss) / total;
  const t = new T(new Float64Array([loss]), [1, 1], 'l2');
  t.req = true;
  t._back = (g) => {
    const gs = (g[0] * weight) / total;
    for (const p of params) {
      if (!p.req) continue;
      const gp = p.grad || (p.grad = new Float64Array(p.data.length));
      for (let i = 0; i < p.data.length; i++) gp[i] += gs * p.data[i];
    }
  };
  return t;
}

/** Extract one row of a [N,D] tensor as a [1,D] tensor (used for per-sample
 *  self-supervised losses inside a batch). */
export function sliceRows(node, row) {
  const D = node.shape[1];
  const out = new Float64Array(D);
  out.set(node.data.subarray(row * D, (row + 1) * D));
  const t = new T(out, [1, D], 'sliceRows');
  t.req = needsGrad(node);
  if (t.req)
    t._back = (g) => {
      const gn = new Float64Array(node.data.length);
      for (let j = 0; j < D; j++) gn[row * D + j] = g[j];
      addGrad(node, gn, node.data.length);
    };
  return t;
}

/** Weighted soft cross entropy (per-row weights, e.g. terminal masking). */
export function softmaxCrossEntropyWeighted(logits, targetProbs, weights, weight = 1) {
  const N = logits.shape[0];
  const K = logits.shape[1];
  const probs = new Float64Array(N * K);
  let loss = 0;
  let wSum = 0;
  for (let n = 0; n < N; n++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) m = Math.max(m, logits.data[n * K + k]);
    let s = 0;
    for (let k = 0; k < K; k++) {
      const e = Math.exp(logits.data[n * K + k] - m);
      probs[n * K + k] = e;
      s += e;
    }
    const w = weights[n];
    if (w <= 0) continue;
    wSum += w;
    for (let k = 0; k < K; k++) {
      probs[n * K + k] /= s;
      const q = targetProbs[n * K + k];
      if (q > 1e-9) loss -= w * q * Math.log(Math.max(probs[n * K + k], 1e-9));
    }
  }
  if (wSum <= 0) {
    const z = new T(new Float64Array([0]), [1, 1], 'wce-zero');
    return z;
  }
  const L = (loss / wSum) * weight;
  const t = new T(new Float64Array([L]), [1, 1], 'weightedCE');
  t.req = needsGrad(logits);
  if (t.req)
    t._back = (g) => {
      const gs = (g[0] * weight) / wSum;
      const gp = new Float64Array(N * K);
      for (let n = 0; n < N; n++) {
        const w = weights[n];
        if (w <= 0) continue;
        for (let k = 0; k < K; k++) gp[n * K + k] = w * (probs[n * K + k] - targetProbs[n * K + k]) * gs;
      }
      addGrad(logits, gp, N * K);
    };
  return t;
}

/** Weighted Huber / smooth-L1 with per-row weights (terminal masking again). */
export function huberWeighted(pred, target, weights, weight = 1, delta = 1.0) {
  const n = pred.data.length;
  const coeff = new Float64Array(n);
  let loss = 0;
  let wSum = 0;
  for (let i = 0; i < n; i++) {
    const w = weights[i] ?? 1;
    const d = pred.data[i] - target.data[i];
    const ad = Math.abs(d);
    if (ad <= delta) {
      loss += w * 0.5 * d * d;
      coeff[i] = w * d;
    } else {
      loss += w * delta * (ad - 0.5 * delta);
      coeff[i] = w * delta * Math.sign(d);
    }
    wSum += w;
  }
  if (wSum <= 0) {
    const z = new T(new Float64Array([0]), [1, 1], 'huber-zero');
    return z;
  }
  const L = (loss / wSum) * weight;
  const t = new T(new Float64Array([L]), [1, 1], 'huberW');
  t.req = needsGrad(pred);
  if (t.req)
    t._back = (g) => {
      const gp = new Float64Array(n);
      const gs = (g[0] * weight) / wSum;
      for (let i = 0; i < n; i++) gp[i] = coeff[i] * gs;
      addGrad(pred, gp, n);
    };
  return t;
}

/** Combine two scalar losses into one node (X = a + b). */
export function addLoss(a, b, wa = 1, wb = 1) {
  const t = new T(new Float64Array([a.data[0] * wa + b.data[0] * wb]), [1, 1], 'addLoss');
  t.req = needsGrad(a, b);
  t._back = (g) => {
    if (a.req) addGrad(a, new Float64Array([g[0] * wa]), 1);
    if (b.req) addGrad(b, new Float64Array([g[0] * wb]), 1);
  };
  return t;
}

/**
 * Convert a scalar-shaped tensor loss into a plain number, then run backward.
 * Returns the numeric loss value.
 */
export function finish(loss, tape) {
  const v = loss.data[0];
  backward(loss, tape);
  return v;
}
