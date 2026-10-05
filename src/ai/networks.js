/**
 * The MuZero/EfficientZero trio of networks, scaled down to browser-CPU size.
 *
 *   representation  o_t                -> h_t          (latent state)
 *   dynamics        (h_t, a_t)         -> h_{t+1}, r_t   (learned model)
 *   prediction      h_t                -> p_t, v_t      (policy prior, value)
 *
 * plus a small projection head used only by the self-supervised latent
 * consistency loss. The paper's ResNet-v2 torso is replaced by a two-layer MLP
 * over the flattened egocentric patch - "such a tiny network can acquire good
 * performance in the limited setting" is literally an appendix-A.1 quote from
 * the EfficientZero paper, and it is doing a lot of work here.
 */

import {
  T,
  add,
  addInto,
  beginTape,
  concat,
  endTape,
  finish,
  huber,
  l2,
  layerNorm,
  matmul,
  negCosine,
  param,
  relu,
  scale,
  softmaxCrossEntropy,
  tanh,
} from './core/tape.js';
import { f64ToBase64, base64ToF64 } from '../sim/util.js';
import { EZ } from '../sim/config.js';

export function heInit(arr, fanIn, rng, scaleFactor = 1) {
  const limit = Math.sqrt(3 / fanIn) * scaleFactor;
  for (let i = 0; i < arr.length; i++) arr[i] = (rng() * 2 - 1) * limit;
  return arr;
}

export function makeRng(seed) {
  let a = seed >>> 0 || 1;
  return function () {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Linear {
  constructor(inDim, outDim, rng, name = 'lin') {
    this.name = name;
    this.inDim = inDim;
    this.outDim = outDim;
    this.W = param(heInit(new Float64Array(inDim * outDim), inDim, rng), [inDim, outDim]);
    // tiny positive bias so ReLU units are alive at initialisation
    const b = new Float64Array(outDim);
    for (let i = 0; i < outDim; i++) b[i] = 0.02;
    this.b = param(b, [outDim]);
  }
  forward(x) {
    return add(matmul(x, this.W), this.b);
  }
  params(out = []) {
    out.push(this.W, this.b);
    return out;
  }
  serialize(obj) {
    obj[this.name + '.W'] = f64ToBase64(this.W.data);
    obj[this.name + '.b'] = f64ToBase64(this.b.data);
    obj[this.name + '.shape'] = [this.inDim, this.outDim];
  }
  load(obj) {
    const w = obj[this.name + '.W'];
    const b = obj[this.name + '.b'];
    if (!w || !b) return false;
    const W = base64ToF64(w);
    const B = base64ToF64(b);
    if (W.length !== this.W.data.length || B.length !== this.b.data.length) return false;
    this.W.data.set(W);
    this.b.data.set(B);
    return true;
  }
}

export class MLP {
  constructor(dims, rng, name = 'mlp', opts = {}) {
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      this.layers.push(new Linear(dims[i], dims[i + 1], rng, `${name}.${i}`));
    }
    this.act = opts.act || 'relu';
    this.norm = opts.norm !== false;
    this.name = name;
  }
  forward(x) {
    let h = x;
    for (let i = 0; i < this.layers.length; i++) {
      h = this.layers[i].forward(h);
      const last = i === this.layers.length - 1;
      if (!last) h = this.act === 'tanh' ? tanh(h) : relu(h);
    }
    return h;
  }
  params(out = []) {
    for (const l of this.layers) l.params(out);
    return out;
  }
  serialize(obj) {
    for (const l of this.layers) l.serialize(obj);
  }
  load(obj) {
    let ok = true;
    for (const l of this.layers) ok = l.load(obj) && ok;
    return ok;
  }
}

export class Adam {
  constructor(params, lr = EZ.lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.b1 = beta1;
    this.b2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = params.map((p) => new Float64Array(p.data.length));
    this.v = params.map((p) => new Float64Array(p.data.length));
  }

  setLr(lr) {
    this.lr = lr;
  }

  /** Global-norm clipped step. Returns {gnorm, step}. */
  step(clip = EZ.gradClip, wd = EZ.weightDecay) {
    this.t++;
    const params = this.params;
    let sq = 0;
    for (let pi = 0; pi < params.length; pi++) {
      const g = params[pi].grad;
      if (!g) continue;
      for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    }
    const gnorm = Math.sqrt(sq);
    const clipScale = gnorm > clip ? clip / (gnorm + 1e-9) : 1;
    const bc1 = 1 - Math.pow(this.b1, this.t);
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi];
      const g = p.grad;
      if (!g) continue;
      const m = this.m[pi];
      const v = this.v[pi];
      for (let i = 0; i < g.length; i++) {
        const gv = g[i] * clipScale + wd * p.data[i];
        m[i] = this.b1 * m[i] + (1 - this.b1) * gv;
        v[i] = this.b2 * v[i] + (1 - this.b2) * gv * gv;
        const mh = m[i] / bc1;
        const vh = v[i] / bc2;
        p.data[i] -= (this.lr * mh) / (Math.sqrt(vh) + this.eps);
      }
      p.grad = null;
    }
    return { gnorm, scale: clipScale };
  }

  serialize(obj) {
    obj.adam = {
      t: this.t,
      lr: this.lr,
    };
  }
}

// Reward classification support. Nine bins rather than MuZero's five: our
// rewards span three orders of magnitude (a step costs -0.001, a kill is +3),
// and coarse bins let a slightly-biased reward head inject a systematic +-2
// per imagined step, which MCTS then compounds into a value collapse.
export const SUPPORT = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
export const REWARD_BINS = SUPPORT.length;

export class Networks {
  constructor(cfg = {}) {
    this.cfg = { ...EZ, ...cfg };
    const rng = makeRng(this.cfg.seed || 1234);
    this.rng = rng;
    const obsDim = this.cfg.obsFrameDim + this.cfg.obsScalarDim;
    const L = this.cfg.latentDim;
    const H = this.cfg.hiddenDim;
    const A = this.cfg.actionDim;

    this.rep = new MLP([obsDim, H, L], rng, 'rep', {});
    this.repNorm = new Linear(L, L, rng, 'repNorm'); // cheap scale/shift stand-in
    this.dyn = new MLP([L + A, H, L], rng, 'dyn', {});
    this.dynReward = new MLP([L + A, H, REWARD_BINS], rng, 'dynR', { norm: false });
    this.predPolicy = new MLP([L, H, A], rng, 'predP', { norm: false });
    this.predValue = new MLP([L, H, 1], rng, 'predV', { norm: false });
    // consistency projection (SimSiam-style)
    this.proj = new MLP([L, H, this.cfg.projDim], rng, 'proj', {});
    this.consistValue = new MLP([L, H, 1], rng, 'consV', { norm: false }); // unused head kept for parity

    this.allParams = [];
    for (const net of [this.rep, this.repNorm, this.dyn, this.dynReward, this.predPolicy, this.predValue, this.proj]) {
      net.params(this.allParams);
    }
    this.optimizer = new Adam(this.allParams, this.cfg.lr);
    this.updates = 0;
    /**
     * Value rescaling (EfficientZero sec. "value rescaling", MuZero-style).
     * The value head is a bounded tanh; if the raw return drifts (it does - a
     * kill is worth +3 while a step costs -0.001) a fixed output range lets the
     * head saturate at its bound, and because MCTS bootstraps from the head the
     * saturation feeds itself. Early builds of Hussrooms did exactly that and
     * every agent's value locked to -8. So the head predicts a *normalised*
     * value and the affine decode tracks the running return statistics.
     */
    this.valueMean = 0;
    this.valueSpread = cfg.valueScale;
  }

  /** Map the tanh output of the value head back into raw return units. */
  decodeValue(tanhOut) {
    return this.valueMean + tanhOut * this.valueSpread;
  }

  // ------------------------------------------------------------- inference
  represent(frame, scalars) {
    // frame: Float32Array(obsFrameDim), scalars: Float32Array(obsScalarDim)
    this._buf = this._buf || new Float64Array(this.cfg.obsFrameDim + this.cfg.obsScalarDim);
    this._buf.set(frame, 0);
    this._buf.set(scalars, this.cfg.obsFrameDim);
    const inNode = this._obs(this._buf);
    const h = this.rep.forward(inNode);
    const hn = layerNorm(h);
    const gated = addInto(hn, this.repNorm.forward(hn), 1, 1);
    return scale(gated, 0.7071);
  }

  representBatch(frameMat, scalarMat) {
    // frameMat: (N * obsFrameDim), scalarMat: (N * obsScalarDim), any float array
    const N = scalarMat.length / this.cfg.obsScalarDim;
    const F = this.cfg.obsFrameDim;
    const D = F + this.cfg.obsScalarDim;
    const buf = new Float64Array(N * D);
    for (let n = 0; n < N; n++) {
      buf.set(frameMat.subarray(n * F, (n + 1) * F), n * D);
      buf.set(scalarMat.subarray(n * this.cfg.obsScalarDim, (n + 1) * this.cfg.obsScalarDim), n * D + F);
    }
    const inNode = new T(buf, [N, D], 'obs');
    const h = this.rep.forward(inNode);
    const hn = layerNorm(h);
    const gated = addInto(hn, this.repNorm.forward(hn), 1, 1);
    return scale(gated, 0.7071);
  }

  predict(h) {
    const logits = this.predPolicy.forward(h);
    const v = tanh(this.predValue.forward(h));
    return { logits, value: scale(v, this.valueSpread), valueNorm: v };
  }

  dynamics(h, actionOneHot) {
    const x = concat([h, actionOneHot]);
    const hNextRaw = this.dyn.forward(x);
    const hNextNorm = layerNorm(hNextRaw);
    const hNext = scale(addInto(h, hNextNorm, 1, 0.5), 0.95); // residual-ish
    const rLogits = this.dynReward.forward(x);
    return { hNext, rLogits };
  }

  _obs(vec) {
    return new T(vec, [1, vec.length], 'obs');
  }

  /** Teacher/stub-style action prior used only by tools/pretrain.mjs. */
  params() {
    return this.allParams;
  }

  // ------------------------------------------------------------ persistence
  serialize() {
    const obj = {
      format: 'hussrooms-ez-net-v1',
      valueMean: this.valueMean,
      valueSpread: this.valueSpread,
      cfg: {
        latentDim: this.cfg.latentDim,
        hiddenDim: this.cfg.hiddenDim,
        obsFrameDim: this.cfg.obsFrameDim,
        obsScalarDim: this.cfg.obsScalarDim,
        actionDim: this.cfg.actionDim,
        projDim: this.cfg.projDim,
      },
      updates: this.updates,
      tensors: {},
    };
    for (const net of [this.rep, this.repNorm, this.dyn, this.dynReward, this.predPolicy, this.predValue, this.proj]) {
      net.serialize(obj.tensors);
    }
    return obj;
  }

  load(obj) {
    if (!obj || obj.format !== 'hussrooms-ez-net-v1') return false;
    const c = obj.cfg;
    if (
      c.latentDim !== this.cfg.latentDim ||
      c.hiddenDim !== this.cfg.hiddenDim ||
      c.obsFrameDim !== this.cfg.obsFrameDim ||
      c.obsScalarDim !== this.cfg.obsScalarDim ||
      c.actionDim !== this.cfg.actionDim
    ) {
      return false;
    }
    let ok = true;
    for (const net of [this.rep, this.repNorm, this.dyn, this.dynReward, this.predPolicy, this.predValue, this.proj]) {
      ok = net.load(obj.tensors) && ok;
    }
    this.updates = obj.updates | 0;
    if (Number.isFinite(obj.valueMean)) this.valueMean = obj.valueMean;
    if (Number.isFinite(obj.valueSpread)) this.valueSpread = obj.valueSpread;
    return ok;
  }

}

/**
 * Tiny imitation target for warm starts: softmax over "which action turns my
 * facing toward the given local bearing". Used by the offline pretrainer and by
 * the (default-off) in-browser warm start toggle.
 */
export function heuristicPolicy(bearingRight, bearingFwd, temperature = 2.0, actionTable) {
  const out = new Float64Array(actionTable.length);
  let sum = 0;
  for (let i = 0; i < actionTable.length; i++) {
    const a = actionTable[i];
    const len = Math.hypot(a.dx, a.dz);
    if (len < 1e-6) {
      out[i] = Math.exp(0);
    } else {
      const align = (a.dx / len) * bearingRight + (a.dz / len) * bearingFwd;
      out[i] = Math.exp((align * temperature) / 1);
    }
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export const _tapeFns = { beginTape, endTape, finish, huber, l2, softmaxCrossEntropy, negCosine };
