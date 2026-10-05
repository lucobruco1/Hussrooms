/**
 * Gradient checker for the hand-rolled autodiff tape.
 * Run: node tools/test-grad.mjs
 *
 * Compares analytic gradients against central finite differences for every op
 * the EfficientZero networks use. If this passes, backprop through the
 * representation / dynamics / prediction stack is sound.
 *
 * NB: the setup closures below deliberately create all inputs and parameters
 * ONCE, and `build()` only re-runs the forward graph - otherwise finite
 * differences would be differentiating a different random network each time.
 */

import {
  T,
  add,
  addInto,
  addLoss,
  backward,
  beginTape,
  concat,
  endTape,
  huber,
  l2,
  layerNorm,
  matmul,
  negCosine,
  param,
  relu,
  scale,
  slice,
  softmaxCrossEntropy,
  tanh,
} from '../src/ai/core/tape.js';
import { makeRng, MLP } from '../src/ai/networks.js';

const rng = makeRng(99);
let failures = 0;
let checks = 0;

const mk = (len, s = 1) => {
  const a = new Float64Array(len);
  for (let i = 0; i < len; i++) a[i] = (rng() * 2 - 1) * s;
  return a;
};

function check(name, setup) {
  const { params, build } = setup();
  const tape = beginTape();
  const loss = build();
  backward(loss, tape);
  endTape();
  // --- finite differences on the very same parameter objects -------------
  const eps = 1e-3;
  let worst = 0;
  let worstDesc = '';
  for (const p of params) {
    const analytic = p.grad || new Float32Array(p.data.length);
    for (let i = 0; i < p.data.length; i += 1) {
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = build().data[0];
      p.data[i] = orig - eps;
      const lm = build().data[0];
      p.data[i] = orig;
      const num = (lp - lm) / (2 * eps);
      // tolerance: absolute floor for FD noise + relative band for real error
      const denom = Math.max(1e-6, Math.abs(analytic[i]) + Math.abs(num));
      const err = Math.abs(analytic[i] - num) / denom;
      if (err > worst) {
        worst = err;
        worstDesc = `${p.name || 'p'}[${i}] analytic=${analytic[i].toExponential(3)} numeric=${num.toExponential(3)}`;
      }
    }
  }
  checks++;
  const ok = worst < 2e-2;
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(34)} max rel err ${worst.toExponential(2)}  ${ok ? '' : worstDesc}`);
}

check('linear+relu chain', () => {
  const W = param(mk(12 * 7), [12, 7], 'W');
  const b = param(mk(7, 0.1), [7], 'b');
  const W2 = param(mk(7 * 4), [7, 4], 'W2');
  const b2 = param(mk(4, 0.1), [4], 'b2');
  const x = new T(mk(3 * 12), [3, 12], 'x');
  const target = new T(mk(3 * 4, 0.5), [3, 4], 't');
  return {
    params: [W, b, W2, b2],
    build: () => huber(add(matmul(relu(add(matmul(x, W), b)), W2), b2), target),
  };
});

check('layerNorm + scale + addInto', () => {
  const A = param(mk(4 * 9), [4, 9], 'A');
  const B = param(mk(4 * 9), [4, 9], 'B');
  const x = new T(mk(4 * 9), [4, 9], 'x');
  const target = new T(mk(4 * 9, 0.3), [4, 9], 't');
  return {
    params: [A, B],
    build: () => {
      const h = layerNorm(addInto(relu(matmul(x, A)), matmul(x, B), 1, 1));
      return huber(scale(h, 0.7), target);
    },
  };
});

check('softmax cross entropy', () => {
  const W = param(mk(5 * 11), [5, 11], 'W');
  const b = param(mk(11, 0.1), [11], 'b');
  const x = new T(mk(6 * 5), [6, 5], 'x');
  const target = new Float64Array(6 * 11);
  for (let i = 0; i < 6; i++) target[i * 11 + (i % 11)] = 1;
  return {
    params: [W, b],
    build: () => softmaxCrossEntropy(add(matmul(x, W), b), target, 1.0),
  };
});

check('concat + slice + tanh', () => {
  const W = param(mk(10 * 6), [10, 6], 'W');
  const x = new T(mk(3 * 10), [3, 10], 'x');
  const target = new T(mk(3 * 6, 0.3), [3, 6], 't');
  return {
    params: [W],
    build: () => {
      const y = tanh(matmul(x, W));
      const c = concat([y, x]);
      return huber(slice(c, 2, 8), target);
    },
  };
});

check('negative cosine consistency', () => {
  const W = param(mk(7 * 7), [7, 7], 'W');
  const x = new T(mk(4 * 7), [4, 7], 'x');
  const tgt = new T(mk(4 * 7), [4, 7], 'tgt'); // detached target branch
  return {
    params: [W],
    build: () => negCosine(matmul(x, W), tgt),
  };
});

check('l2 regulariser + combined losses', () => {
  const W = param(mk(9 * 4), [9, 4], 'W');
  const x = new T(mk(2 * 9), [2, 9], 'x');
  const target = new T(mk(2 * 4, 0.2), [2, 4], 't');
  return {
    params: [W],
    build: () => addLoss(huber(matmul(x, W), target), l2([W], 0.01)),
  };
});

check('full MLP stack (as used by nets)', () => {
  const net = new MLP([16, 12, 8], rng, 'test');
  const x = new T(mk(5 * 16), [5, 16], 'x');
  const target = new T(mk(5 * 8, 0.3), [5, 8], 't');
  return {
    params: net.params(),
    build: () => huber(net.forward(x), target),
  };
});

console.log(`\n${checks - failures}/${checks} gradient checks passed`);
process.exit(failures ? 1 : 0);
