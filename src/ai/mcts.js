/**
 * Monte-Carlo tree search over the *learned* model - MuZero/EfficientZero
 * flavour. There is no copy of the real simulator inside the planner: nodes
 * hold latents produced by the dynamics network, priors and values come from
 * the prediction network, and returns accumulate the model's predicted rewards
 * along the searched path.
 *
 * One edge is expanded per simulation (the standard MuZero scheme), so a
 * 12-simulation search costs ~12 dynamics + ~13 prediction forwards - cheap
 * enough to run four survivor brains plus one monster brain at 12 Hz on a CPU.
 *
 * Selection is PUCT with MuZero's exploration schedule
 *   U(s,a) = P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
 * with the log((1+N+c2)/c2) rescaling folded into the constant.
 */

import { ACTIONS, EZ } from '../sim/config.js';
import { softmax, sampleFrom } from '../sim/util.js';
import { SUPPORT as REWARD_SUPPORT } from './networks.js';

export class MCTSNode {
  constructor(latent, parent = null, actionIdx = -1) {
    this.h = latent; // latent tensor (Float64 tape node)
    this.parent = parent;
    this.actionIdx = actionIdx;
    this.children = null; // Array(actionDim) of MCTSNode | null
    this.priors = null; // Float64Array(actionDim)
    this.reward = 0; // predicted reward on the edge into this node (raw units)
    this.value = 0; // mean backed-up value
    this.valueSum = 0;
    this.visits = 0;
    this.evaluated = false; // prediction net has run on this latent
  }
}

/**
 * Symmetric categorical head expectation -> scalar.
 *
 * The bin count MUST match the reward head's output width; getting this wrong
 * (a 9-bin head evaluated against a 5-bin support) silently produced NaN
 * rewards, which NaN'd the whole value function within a few updates.
 */
export function categoricalToScalar(logits, bins = REWARD_SUPPORT) {
  const probs = softmax(logits);
  if (bins.length !== probs.length) {
    throw new Error(`categoricalToScalar: ${probs.length} bins in the head vs ${bins.length} in the support`);
  }
  let v = 0;
  for (let i = 0; i < probs.length; i++) v += probs[i] * bins[i];
  return v;
}

export class MCTS {
  constructor(net, cfg = {}) {
    this.net = net;
    this.cfg = { ...EZ, ...cfg };
    this.actionDim = this.cfg.actionDim || ACTIONS.length;
    this.simsRun = 0;
  }

  /**
   * @param h0 latent for the current observation (encoded by the caller)
   * @param opts {explore, dirichlet, sims, temperature, rng}
   * @returns {{probs: Float64Array, value: number, visits: Float64Array, root: MCTSNode}}
   */
  search(h0, opts = {}) {
    const cfg = this.cfg;
    const sims = opts.sims ?? cfg.sims;
    const root = new MCTSNode(h0);
    this.evaluate(root);
    for (let i = 0; i < sims; i++) {
      const leaf = this.simulate(root, 0);
      this.backup(leaf);
      this.simsRun++;
    }
    const A = this.actionDim;
    const visits = new Float64Array(A);
    let total = 0;
    for (let a = 0; a < A; a++) {
      const c = root.children[a];
      visits[a] = c ? c.visits : 0;
      total += visits[a];
    }
    const probs = new Float64Array(A);
    if (total === 0) probs.fill(1 / A);
    else for (let a = 0; a < A; a++) probs[a] = visits[a] / total;
    if (opts.explore && opts.dirichlet) {
      const noise = dirichlet(A, 0.3);
      for (let a = 0; a < A; a++) probs[a] = (1 - opts.dirichlet) * probs[a] + opts.dirichlet * noise[a];
    }
    return { probs, value: root.value, visits, root };
  }

  evaluate(node) {
    const { logits, value } = this.net.predict(node.h);
    const probs = softmax(logits.data);
    node.priors = probs;
    node.value = value.data[0];
    node.children = new Array(this.actionDim).fill(null);
    node.evaluated = true;
    return node;
  }

  simulate(node, depth) {
    if (!node.evaluated) {
      this.evaluate(node);
      return node;
    }
    if (depth >= this.cfg.maxDepth) return node;
    const a = this.selectAction(node);
    let child = node.children[a];
    if (!child) {
      const oneHot = { data: new Float64Array(this.actionDim), shape: [1, this.actionDim], req: false, op: 'onehot' };
      oneHot.data[a] = 1;
      const { hNext, rLogits } = this.net.dynamics(node.h, oneHot);
      const reward = categoricalToScalar(rLogits.data) * this.cfg.rewardScale;
      child = new MCTSNode(hNext, node, a);
      child.reward = reward;
      node.children[a] = child;
      this.evaluate(child);
      return child;
    }
    return this.simulate(child, depth + 1);
  }

  /**
   * MuZero's backpropagate(): each node accumulates the *return from that
   * node*, i.e. (predicted reward on the edge into it) + discount * (return
   * from its child). The root simply takes the returned value.
   */
  backup(leaf) {
    const cfg = this.cfg;
    let value = leaf.value;
    for (let n = leaf; n; n = n.parent) {
      const add = n.parent === null ? value : n.reward + cfg.discount * value;
      n.visits++;
      n.valueSum += add;
      n.value = n.valueSum / n.visits;
      value = add;
    }
  }

  /**
   * PUCT selection. Q values are min-max normalised across the node's visited
   * children (MuZero's _normalize_value) so the exploration bonus and the value
   * term always live on the same [0,1] scale, whatever the value head learned.
   */
  selectAction(node) {
    const cfg = this.cfg;
    const sqrtN = Math.sqrt(Math.max(1, node.visits));
    const c1 = cfg.c1 * Math.log((node.visits + cfg.c2 + 1) / cfg.c2);
    let lo = Infinity;
    let hi = -Infinity;
    for (let a = 0; a < this.actionDim; a++) {
      const c = node.children[a];
      if (c && c.visits > 0) {
        if (c.value < lo) lo = c.value;
        if (c.value > hi) hi = c.value;
      }
    }
    const span = hi - lo;
    let best = -1;
    let bestScore = -Infinity;
    for (let a = 0; a < this.actionDim; a++) {
      const child = node.children[a];
      const prior = node.priors[a];
      let q = 0.5;
      if (child && child.visits > 0 && span > 1e-6) q = (child.value - lo) / span;
      else if (child && child.visits > 0) q = 0.5;
      const u = (c1 * prior * sqrtN) / (1 + (child ? child.visits : 0));
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        best = a;
      }
    }
    return best;
  }

  /** Visit-count distribution -> action, with temperature. */
  static pick(probs, temperature, rng = Math.random) {
    let p;
    if (temperature !== 1) {
      p = new Float64Array(probs.length);
      let s = 0;
      for (let i = 0; i < probs.length; i++) {
        p[i] = Math.pow(Math.max(probs[i], 1e-9), 1 / temperature);
        s += p[i];
      }
      for (let i = 0; i < p.length; i++) p[i] /= s;
    } else {
      p = probs;
    }
    return sampleFrom(p, rng);
  }
}

export function dirichlet(n, alpha = 0.3) {
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    out[i] = gammaSample(alpha, 1);
    sum += out[i];
  }
  if (sum <= 0) out.fill(1 / n);
  else for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

function gammaSample(alpha, beta) {
  if (alpha < 1) {
    const u = Math.random();
    return gammaSample(1 + alpha, beta) * Math.pow(u, 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x;
    let v;
    do {
      x = gauss();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return (d * v) / beta;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return (d * v) / beta;
  }
}

function gauss() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
