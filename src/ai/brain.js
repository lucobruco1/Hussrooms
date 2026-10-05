/**
 * EZBrain - one EfficientZero agent.
 *
 * Implements the three ingredients the EfficientZero paper identifies as
 * essential for sample-efficient model-based visual RL (arXiv:2111.00210,
 * "Mastering Atari Games with Limited Data"):
 *
 *   1. a self-supervised environment model - dynamics predicts the next latent,
 *      trained with a SimSiam-style negative-cosine consistency loss against
 *      the representation of the *real* next observation;
 *   2. mitigation of model compounding error - end-to-end "value prefix"
 *      training: the value predicted at every unrolled step is trained against
 *      a multi-step bootstrapped target;
 *   3. off-policy correction - MuZero-style reanalysis: stored MCTS value and
 *      policy targets are refreshed using the current network.
 *
 * Plus the MuZero machinery: PUCT search over the learned model, prioritised
 * replay with importance-sampling weights, target networks, categorical reward
 * and value heads.
 *
 * Honest caveats vs. the paper: the networks are ~1000x smaller, the batch is
 * 24 rather than 256, reanalysis is budgeted to a couple of windows per update,
 * and there is no GPU. Same algorithm, a scale a browser tab can afford.
 *
 * NOTE ON AUTONOMY: there is no behaviour code anywhere in this file. The brain
 * cannot see "the player" as a concept - it sees a latent, a reward stream and
 * its own search. Everything it comes to do, it learns from the reward signal
 * produced by World.computeShaping / kill().
 */

import { LEARN, ezCfg } from '../sim/config.js';
import { clamp, mean } from '../sim/util.js';
import { MCTS } from './mcts.js';
import { Networks, SUPPORT } from './networks.js';
import {
  T,
  addLoss,
  beginTape,
  detach,
  endTape,
  huberWeighted,
  l2,
  negCosine,
  scale,
  setGradEnabled,
  sliceRows,
  softmaxCrossEntropy,
  softmaxCrossEntropyWeighted,
} from './core/tape.js';

const BINS = SUPPORT.length;

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function makeRng(seed) {
  let a = seed >>> 0 || 1;
  return function () {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Replay {
  constructor(capacity) {
    this.capacity = capacity;
    this.buf = new Array(capacity).fill(null);
    this.head = 0; // next write position
    this.count = 0;
    this.totalWritten = 0;
  }

  push(entry) {
    this.buf[this.head] = entry;
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
    this.totalWritten++;
  }

  /** Chronological index: 0 = oldest retained entry, size-1 = newest. */
  at(i) {
    const start = this.totalWritten > this.capacity ? this.head : 0;
    return this.buf[(start + i) % this.capacity];
  }

  get size() {
    return this.count;
  }
}

export class EZBrain {
  /**
   * @param opts {id, label, obsDims:{pooled, scalars}, seed, cfg?}
   */
  constructor(opts) {
    this.id = opts.id;
    this.label = opts.label || opts.id;
    this.cfg = { ...ezCfg(opts.profile || 'live', opts.cfg || {}) };
    this.cfg.obsFrameDim = opts.obsDims.pooled;
    this.cfg.obsScalarDim = opts.obsDims.scalars;
    this.cfg.actionDim = opts.actionDim || 11; // survivors; Clark passes 13
    this.cfg.seed = opts.seed ?? 42;
    this.net = new Networks(this.cfg);
    this.targetNet = new Networks(this.cfg);
    this.targetNet.load(this.net.serialize());
    this.mcts = new MCTS(this.net, this.cfg);
    this.targetMcts = new MCTS(this.targetNet, this.cfg);
    this.replay = new Replay(this.cfg.replayCapacity);
    this.rng = makeRng((opts.seed ?? 42) ^ 0x5bf03635);
    this.decisions = 0;
    this.updates = 0;
    this.history = [];
    this.emaValue = 1;
    this.lossEma = 1;
    this.lastLosses = { total: 0, value: 0, reward: 0, policy: 0, consistency: 0 };
    this.lastPriority = 1;
    this.trainMsUsed = 0;
    this.lastAction = 0;
    this.policyEntropy = Math.log(this.cfg.actionDim);
    this.savedBy = null; // 'pretrain' | 'in-browser' bookkeeping label
    this.warmupActionsRemaining = this.cfg.warmupActions;
    this.skippedUpdates = 0; // non-finite gradients rejected by the guard in trainOnce()
    this.lastTrace = null; // last search tree, for the lab panel (rendering only)
  }

  get obsFrameDim() {
    return this.cfg.obsFrameDim;
  }

  // ------------------------------------------------------------------ acting
  /**
   * @param pooled  Float32Array egocentric patch (pooled)
   * @param scalars Float32Array scalars
   * @param opts    {explore:boolean}
   * @returns {action, probs, value, warmup}
   */
  act(pooled, scalars, opts = {}) {
    const explore = opts.explore !== false;
    const A = this.cfg.actionDim;
    this.decisions++;
    if (this.decisions <= this.cfg.warmupActions) {
      return this._actRandom(A);
    }
    setGradEnabled(false); // inference is a no-op for autodiff
    try {
      return this._actSearch(pooled, scalars, explore, opts);
    } finally {
      setGradEnabled(true);
    }
  }

  _actRandom(A) {
    // Pure exploration phase. Recorded as a smoothed one-hot policy target so
    // the policy head has something to imitate before real search
    // distributions exist; MuZero itself would have no policy target here.
    const action = Math.floor(this.rng() * A);
    const probs = new Float64Array(A).fill(0.12 / Math.max(1, A - 1));
    probs[action] = 0.88;
    this.lastAction = action;
    this._lastSearch = { probs, value: 0, warmup: true };
    return { action, probs, value: 0, warmup: true };
  }

  _actSearch(pooled, scalars, explore, opts = {}) {
    const A = this.cfg.actionDim;
    const h0 = this.net.represent(pooled, scalars);
    const temperature = this.decisionTemperature();
    const res = this.mcts.search(h0, {
      explore,
      dirichlet: explore ? this.cfg.rootNoise : 0,
      sims: this.cfg.sims,
    });
    const { probs, value, visits } = res;
    // Policy-target smoothing: the imitation target is a slightly flattened
    // search distribution, so the policy head can never fully saturate on one
    // action and lose the ability to explore its way out of a bad basin.
    if (explore && this.cfg.policySmoothing > 0) {
      const lam = this.cfg.policySmoothing;
      for (let a = 0; a < probs.length; a++) probs[a] = (1 - lam) * probs[a] + lam / probs.length;
    }
    let action = MCTS.pick(probs, opts.temperature ?? temperature, this.rng);
    // Exploration floor: never let the collection policy become deterministic.
    // Off-policy RL in a world where the only way to learn a reward exists is to
    // stumble into it needs a permanent source of surprise.
    if (explore && this.rng() < this.cfg.exploreEps) action = Math.floor(this.rng() * A);
    if (explore && visits.every((v) => v === 0)) action = Math.floor(this.rng() * A);
    this.lastAction = action;
    this._lastSearch = { probs, value, visits };
    this.policyEntropy = entropy(probs);
    this.lastTrace = this._buildTrace(res.root, action, visits);
    return { action, probs, value };
  }

  /**
   * Renderer-only view of the search that just happened: the root's children
   * with their visit counts/priors/Q, plus the principal variation annotated
   * with the model's predicted reward and value. Nothing in training reads it.
   */
  _buildTrace(root, action, visits) {
    if (!root || !root.children) return null;
    const children = [];
    for (let a = 0; a < this.cfg.actionDim; a++) {
      const c = root.children[a];
      if (!c) continue;
      children.push({ a, n: c.visits, prior: root.priors ? root.priors[a] : 0, q: c.visits ? c.value : 0 });
    }
    const rollout = [{ reward: 0, value: root.value }];
    let node = root;
    for (let d = 0; d < this.cfg.maxDepth; d++) {
      if (!node.children) break;
      let best = -1;
      let bestN = 0;
      for (let a = 0; a < this.cfg.actionDim; a++) {
        const c = node.children[a];
        if (c && c.visits > bestN) {
          bestN = c.visits;
          best = a;
        }
      }
      if (best < 0) break;
      node = node.children[best];
      rollout.push({ reward: node.reward, value: node.value, action: best });
    }
    return { children, rollout, action, visits: Array.from(visits) };
  }

  decisionTemperature() {
    const t = this.decisions / this.cfg.tempSteps;
    return t >= 1 ? this.cfg.temperatureFinal : this.cfg.temperature;
  }

  /** Append a finished transition to the replay buffer. */
  observe(entry) {
    this.replay.push({
      frame: entry.frame.slice(),
      scalars: entry.scalars.slice(),
      action: entry.action,
      reward: entry.reward,
      policy: entry.policy ? Float64Array.from(entry.policy) : null,
      value: entry.value ?? 0,
      done: !!entry.done,
      priority: this.lastPriority,
      sampled: 0,
    });
  }

  // ---------------------------------------------------------------- training
  /**
   * Run gradient updates until the time budget is exhausted. Called from inside
   * the render loop, so it must never block for long.
   */
  update(budgetMs = LEARN.trainBudgetMs, opsBudget = LEARN.opsPerFrame) {
    if (!LEARN.learn) return 0;
    const t0 = nowMs();
    let ops = 0;
    const minSize = this.cfg.unroll + this.cfg.batch + 4;
    while (ops < opsBudget && this.replay.size >= minSize) {
      this.trainOnce();
      ops++;
      if (nowMs() - t0 >= budgetMs) break;
    }
    this.trainMsUsed = nowMs() - t0;
    return ops;
  }

  trainOnce() {
    const cfg = this.cfg;
    const K = cfg.unroll;
    const N = cfg.batch;
    const A = cfg.actionDim;
    const F = cfg.obsFrameDim;
    const S = cfg.obsScalarDim;
    const replay = this.replay;
    const usable = replay.size - K - 1;
    if (usable < 2) return;
    const gamma = cfg.discount;

    // ------------------------------------------------------------ sampling
    const idx = new Int32Array(N);
    let maxP = 0;
    for (let i = 0; i < N; i++) {
      idx[i] = Math.floor(this.rng() * usable);
      maxP = Math.max(maxP, replay.at(idx[i]).priority);
    }
    const isw = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const p = replay.at(idx[i]).priority / (maxP + 1e-9);
      isw[i] = Math.pow(p + 1e-6, -cfg.priorityBeta);
    }
    const maxW = Math.max(...isw);
    for (let i = 0; i < N; i++) isw[i] /= maxW;

    // ------------------------------------------------------- gather windows
    const frames = new Float32Array(N * F);
    const scalars = new Float32Array(N * S);
    const actIdx = new Int32Array(N * K);
    const rewardTargets = new Float64Array(K * N * BINS);
    const valueTargets = new Float64Array(N);
    const prefixTargets = new Float64Array(K * N);
    const stepPolicyTargets = new Float64Array(K * N * A); // MCTS policy at t+k
    const stepMask = new Float64Array(K * N).fill(1); // 1 = train this step
    const consK = [];
    const consRow = [];
    const consFrame = [];
    const consScalar = [];
    const consistStride = Math.max(2, cfg.consistEveryK);

    for (let i = 0; i < N; i++) {
      const base = idx[i];
      const e0 = replay.at(base);
      frames.set(e0.frame, i * F);
      scalars.set(e0.scalars, i * S);
      // --- walk the window forward
      let termStep = -1;
      for (let k = 0; k < K; k++) {
        const ek = replay.at(base + k);
        actIdx[i * K + k] = ek.action;
        const rScaled = clamp(ek.reward / cfg.rewardScale, -1, 1);
        softOneHot(rewardTargets, ((k * N + i) * BINS) | 0, rScaled);
        // policy target for this unrolled step (search distribution at t+k)
        const pol = ek.policy;
        const off = (k * N + i) * A;
        if (pol) for (let a = 0; a < A; a++) stepPolicyTargets[off + a] = pol[a];
        else stepPolicyTargets[off + ek.action] = 1;
        if (ek.done && termStep < 0) termStep = k;
      }
      if (termStep >= 0) {
        for (let k = termStep + 1; k < K; k++) stepMask[k * N + i] = 0;
      }
      // --- value targets z_k = sum_{j<k} g^j u_j + g^k nu_k
      let acc = 0;
      let pw = 1;
      for (let k = 0; k <= K; k++) {
        if (k > 0) {
          acc += pw * replay.at(base + k - 1).reward;
          pw *= gamma;
        }
        const beyond = termStep >= 0 && k > termStep;
        const z = acc + (beyond ? 0 : pw * replay.at(base + k).value);
        if (k === 0) valueTargets[i] = z;
        else prefixTargets[(k - 1) * N + i] = z;
      }
      // --- self-supervised consistency pairs (real o_{t+k} vs predicted h_k)
      for (let k = 2; k <= K; k += consistStride) {
        const ek = replay.at(base + k);
        consK.push(k - 1);
        consRow.push(i);
        consFrame.push(ek.frame);
        consScalar.push(ek.scalars);
      }
    }
    // --- advantage weights for the auxiliary actor term: exp(A / beta) where
    //     A = (bootstrapped K-step return) - (value search reported at the time)
    //     clamped so one lucky bite cannot dominate a batch.
    const advW = new Float64Array(N);
    {
      const b = Math.max(1e-3, cfg.advBeta);
      for (let i = 0; i < N; i++) {
        const v0 = replay.at(idx[i]).value ?? 0;
        const adv = (valueTargets[i] - v0) / b;
        advW[i] = Math.min(cfg.advClamp, Math.exp(clamp(adv, -6, 6)));
      }
    }

    // --- running return statistics for value rescaling
    let zMean = 0;
    for (let i = 0; i < N; i++) zMean += valueTargets[i];
    zMean /= N;
    let zVar = 0;
    for (let i = 0; i < N; i++) zVar += (valueTargets[i] - zMean) ** 2;
    zVar /= N;
    const zStd = Math.sqrt(zVar);
    // Damped, clamped adaptation. An unclamped tracker happily follows a
    // collapsing value function down to -100, which hides the collapse instead
    // of preventing it.
    const meanTarget = clamp(zMean, -40, 40);
    const spreadTarget = clamp(3 * zStd, 1.0, 40);
    this.net.valueMean = clamp(this.net.valueMean * 0.995 + meanTarget * 0.005, -40, 40);
    this.net.valueSpread = clamp(this.net.valueSpread * 0.995 + spreadTarget * 0.005, 1, 40);
    this.targetNet.valueMean = this.net.valueMean;
    this.targetNet.valueSpread = this.net.valueSpread;
    const vMean = this.net.valueMean;
    const vSpread = this.net.valueSpread;

    // -------------------------------------------------------------- forward
    const tape = beginTape();
    const h0 = this.net.representBatch(frames, scalars);
    const rootPred = this.net.predict(h0);
    const hs = [h0];
    const ps = [rootPred.logits];
    const vs = [rootPred.value];
    const vNorm = [rootPred.valueNorm];
    const rs = [];
    let h = h0;
    for (let k = 0; k < K; k++) {
      const oh = new Float64Array(N * A);
      for (let i = 0; i < N; i++) oh[i * A + actIdx[i * K + k]] = 1;
      const { hNext, rLogits } = this.net.dynamics(h, new T(oh, [N, A], 'onehot'));
      h = hNext;
      const pred = this.net.predict(h);
      hs.push(h);
      ps.push(pred.logits);
      vs.push(pred.value);
      vNorm.push(pred.valueNorm);
      rs.push(rLogits);
    }

    // --------------------------------------------------------------- losses
    // 1. policy: imitate the search distribution at the root (k = 0), exactly
    //    MuZero's policy target.
    const rootPolicy = new Float64Array(N * A);
    for (let i = 0; i < N; i++) {
      const e = replay.at(idx[i]);
      if (e.policy) rootPolicy.set(e.policy.subarray(0, A), i * A);
      else rootPolicy[i * A + e.action] = 1;
    }
    let policyLoss = softmaxCrossEntropy(ps[0], rootPolicy, cfg.wPolicy);
    // ...and the search distribution at every imagined step, MuZero-style. This
    // is a large amount of extra policy signal per gradient step for almost no
    // compute, because those logits were already produced by the unroll.
    for (let k = 1; k <= K; k++) {
      const tgt = stepPolicyTargets.subarray((k - 1) * N * A, k * N * A);
      const w = stepMask.subarray((k - 1) * N, k * N);
      const entry = softmaxCrossEntropyWeighted(ps[k], tgt, w, (cfg.wPolicy * 0.5) / k);
      policyLoss = addLoss(policyLoss, entry);
    }

    // 1b. auxiliary actor term: push the probability of actions that beat the
    //     value estimate up, and those that fell short up less (the weights are
    //     detached numbers, so this is a plain weighted cross-entropy).
    if (cfg.wAdvantage > 0) {
      const advTargets = new Float64Array(N * A);
      for (let i = 0; i < N; i++) {
        const e = replay.at(idx[i]);
        advTargets[i * A + e.action] = 1;
      }
      const entry = softmaxCrossEntropyWeighted(ps[0], advTargets, advW, cfg.wAdvantage);
      policyLoss = addLoss(policyLoss, entry);
    }

    // 2. reward: categorical prediction at every unrolled step, masked after
    //    death.
    let rewardLoss = null;
    for (let k = 0; k < K; k++) {
      const tgt = rewardTargets.subarray(k * N * BINS, (k + 1) * N * BINS);
      const w = stepMask.subarray(k * N, (k + 1) * N);
      const entry = softmaxCrossEntropyWeighted(rs[k], tgt, w, cfg.wReward);
      rewardLoss = rewardLoss ? addLoss(rewardLoss, entry) : entry;
    }

    // 3. value: the root state is trained on the full K-step bootstrapped
    //    target; every unrolled state carries the EfficientZero value prefix.
    //    Both live in rescaled units so the bounded head can express them.
    const inv = 1 / vSpread;
    const vt = new Float64Array(N);
    // Clamp in rescaled space: a single wild bootstrap target must not be able
    // to yank the head, and the clamp is wide enough (6 sigma) that real
    // returns are untouched.
    for (let i = 0; i < N; i++) vt[i] = clamp((valueTargets[i] - vMean) * inv, -6, 6);
    const rootValueLoss = huberWeighted(vNorm[0], new T(vt, [N, 1], 'vt'), isw, cfg.wValue);
    let prefixLoss = null;
    for (let k = 0; k < K; k++) {
      const t = new Float64Array(N);
      for (let i = 0; i < N; i++) t[i] = clamp((prefixTargets[k * N + i] - vMean) * inv, -6, 6);
      const w = stepMask.subarray(k * N, (k + 1) * N);
      const entry = huberWeighted(vNorm[k + 1], new T(t, [N, 1], 'pt'), w, cfg.wValue * 0.5);
      prefixLoss = prefixLoss ? addLoss(prefixLoss, entry) : entry;
    }

    // 4. self-supervised latent consistency: pull the model's imagined latent
    //    towards the representation of the real observation that followed.
    let consLoss = null;
    if (cfg.useConsistency && consRow.length) {
      const M = consRow.length;
      const cf = new Float32Array(M * F);
      const cs = new Float32Array(M * S);
      for (let m = 0; m < M; m++) {
        cf.set(consFrame[m], m * F);
        cs.set(consScalar[m], m * S);
      }
      const tgtLatents = this.targetNet.representBatch(cf, cs); // stop-grad family
      const tgtProj = detach(this.targetNet.proj.forward(tgtLatents));
      for (let m = 0; m < M; m++) {
        const proj = this.net.proj.forward(sliceRows(hs[consK[m]], consRow[m]));
        const tgt = sliceRows(tgtProj, m);
        const entry = negCosine(proj, tgt);
        consLoss = consLoss ? addLoss(consLoss, entry) : entry;
      }
      consLoss = scale(consLoss, cfg.wConsistency);
    }

    // 5. assemble + regularise
    let total = addLoss(addLoss(policyLoss, rewardLoss), addLoss(rootValueLoss, prefixLoss));
    if (consLoss) total = addLoss(total, consLoss);
    if (cfg.wL2 > 0) total = addLoss(total, l2(this.net.params(), cfg.wL2));
    const lossValue = total.data[0];

    // ------------------------------------------------------------- backward
    const t1 = nowMs();
    total.grad = new Float64Array(1).fill(1);
    for (let i = tape.length - 1; i >= 0; i--) {
      const n = tape[i];
      if (!n._back || !n.grad) continue;
      n._back(n.grad);
      n._back = null;
    }
    endTape();
    // Safety valve: one non-finite gradient would otherwise poison every
    // parameter in a single step and the agent would be dead for the rest of
    // the session. Skip the update instead (and count it).
    let finite = Number.isFinite(lossValue);
    if (finite) {
      for (const p of this.net.params()) {
        if (p.grad && !allFinite(p.grad)) {
          finite = false;
          break;
        }
      }
    }
    if (finite) {
      this.net.optimizer.step(cfg.gradClip, cfg.weightDecay);
    } else {
      this.skippedUpdates = (this.skippedUpdates || 0) + 1;
      for (const p of this.net.params()) p.grad = null;
    }
    this.trainMs = nowMs() - t1;

    // ------------------------------------------- priorities + bookkeeping
    let errSum = 0;
    for (let i = 0; i < N; i++) {
      const err = Math.abs(vs[0].data[i] - valueTargets[i]);
      errSum += err;
      const e = replay.at(idx[i]);
      e.priority = Math.pow(err + 1e-4, cfg.priorityAlpha);
      e.sampled++;
    }
    this.lastPriority = Math.pow(errSum / N + 1e-4, cfg.priorityAlpha);
    this.updates++;
    this.lossEma = this.lossEma * 0.95 + lossValue * 0.05;
    this.emaValue = this.emaValue * 0.99 + Math.abs(valueTargets[0]) * 0.01;
    this.lastLosses = {
      // the consistency term is a sum of negative cosines, so it is reported
      // separately; `total` keeps the raw optimised value
      total: lossValue,
      value: rootValueLoss.data[0] + (prefixLoss ? prefixLoss.data[0] : 0),
      reward: rewardLoss ? rewardLoss.data[0] : 0,
      policy: policyLoss.data[0],
      consistency: consLoss ? consLoss.data[0] : 0,
    };
    if (this.updates % cfg.targetSyncEvery === 0) this.syncTarget();
    if (cfg.useReanalyze && this.updates % 3 === 0) this.reanalyze(idx, 2);
    if (this.updates % 25 === 0) {
      this.history.push({ u: this.updates, loss: this.lossEma, v: this.emaValue, entropy: this.policyEntropy });
      if (this.history.length > 500) this.history.shift();
    }
  }

  syncTarget() {
    const src = this.net.params();
    const dst = this.targetNet.params();
    for (let i = 0; i < src.length; i++) dst[i].data.set(src[i].data);
  }

  /**
   * Off-policy correction. EfficientZero reanalyses ~99% of stored data with
   * the current network; we spend a slice of the update budget refreshing the
   * value/policy targets of a couple of sampled windows, writing the refreshed
   * numbers back into the buffer so old transitions stop being anchored to the
   * (much worse) network that produced them.
   */
  reanalyze(sampledIdx, count = 2) {
    setGradEnabled(false);
    try {
      this._reanalyze(sampledIdx, count);
    } finally {
      setGradEnabled(true);
    }
  }

  _reanalyze(sampledIdx, count = 2) {
    for (let c = 0; c < count; c++) {
      const i = sampledIdx[(c * 11 + 3) % sampledIdx.length];
      const e = this.replay.at(i);
      if (!e) continue;
      const h = this.targetNet.represent(e.frame, e.scalars);
      const { probs, value } = this.targetMcts.search(h, {
        explore: false,
        sims: Math.max(4, Math.floor(this.cfg.sims * 0.6)),
      });
      e.policy = Float64Array.from(probs);
      e.value = value;
      e.reanalyzed = (e.reanalyzed || 0) + 1;
    }
  }

  /** Value estimate for a single observation (UI / debugging). */
  valueOf(pooled, scalars) {
    setGradEnabled(false);
    try {
      const h = this.net.represent(pooled, scalars);
      const { value } = this.net.predict(h);
      return value.data[0];
    } finally {
      setGradEnabled(true);
    }
  }

  /** Policy prior for a single observation (UI / debugging). */
  policyOf(pooled, scalars) {
    setGradEnabled(false);
    try {
      return this._policyOf(pooled, scalars);
    } finally {
      setGradEnabled(true);
    }
  }

  _policyOf(pooled, scalars) {
    const h = this.net.represent(pooled, scalars);
    const { logits } = this.net.predict(h);
    const m = Math.max(...logits.data);
    const exps = Array.from(logits.data, (v) => Math.exp(v - m));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / s);
  }

  // ------------------------------------------------------------ persistence
  serialize() {
    return {
      format: 'hussrooms-brain-v1',
      id: this.id,
      label: this.label,
      updates: this.updates,
      decisions: this.decisions,
      net: this.net.serialize(),
      history: this.history.slice(-200),
    };
  }

  load(obj) {
    if (!obj || obj.format !== 'hussrooms-brain-v1') return false;
    const ok = this.net.load(obj.net);
    if (ok) {
      this.syncTarget();
      this.updates = obj.updates | 0;
      this.decisions = obj.decisions | 0;
      this.history = obj.history || [];
    }
    return ok;
  }
}

// ------------------------------------------------------------------ helpers

function softOneHot(out, off, value) {
  let lo = 0;
  for (let i = 0; i < SUPPORT.length - 1; i++) if (value >= SUPPORT[i]) lo = i;
  const hi = Math.min(SUPPORT.length - 1, lo + 1);
  out.fill(0, off, off + SUPPORT.length);
  if (hi === lo || SUPPORT[hi] === SUPPORT[lo]) {
    out[off + lo] = 1;
    return;
  }
  const t = (value - SUPPORT[lo]) / (SUPPORT[hi] - SUPPORT[lo]);
  out[off + lo] += 1 - t;
  out[off + hi] += t;
}

function allFinite(arr) {
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) return false;
  return true;
}

function entropy(probs) {
  let h = 0;
  for (const p of probs) if (p > 1e-9) h -= p * Math.log(p);
  return h;
}

export { SUPPORT };
