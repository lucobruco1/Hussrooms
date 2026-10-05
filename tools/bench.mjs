/**
 * Micro-benchmark: where does the CPU budget actually go?
 *   node tools/bench.mjs
 */

import { World } from '../src/sim/world.js';
import { Session } from '../src/ai/session.js';
import { obsDims, poolObs } from '../src/sim/features.js';

function time(label, n, fn) {
  fn(); // warmup
  const t0 = performance.now();
  for (let i = 0; i < n; i++) fn();
  const dt = (performance.now() - t0) / n;
  console.log(`${label.padEnd(38)} ${dt.toFixed(3)} ms/op   (${(1000 / dt).toFixed(0)}/s)`);
  return dt;
}

const world = new World({ seed: 5 });
for (let i = 0; i < 60; i++) world.step();

const dims = obsDims();
const e = world.entities.find((x) => x.kind === 'bot');
const obs = world.observe(e);
const pooled = new Float32Array(dims.pooled);
poolObs(obs.frame, pooled);

time('world.step (6 entities)', 3000, () => world.step());
time('features.encode (1 agent)', 3000, () => world.observe(e));
time('poolObs', 20000, () => poolObs(obs.frame, pooled));

const session = new Session({ seed: 7 });
// collect some experience + finish warmup so MCTS is actually used
for (let i = 0; i < 400; i++) session.step();
time('full session.step (sim + decisions)', 300, () => {
  session.world.tick++;
  session.runDecisions();
  session.world.step();
});

const brain = session.botBrain;
brain.decisions = 10 ** 6; // force MCTS path
time('brain.act (8 sims MCTS)', 200, () => brain.act(pooled, obs.scalars, { explore: true }));

const net = brain.net;
time('net.represent (1 obs)', 2000, () => net.represent(pooled, obs.scalars));
const h = net.represent(pooled, obs.scalars);
time('net.predict (1 latent)', 4000, () => net.predict(h));
const oh = { data: new Float64Array(11).fill(1 / 11), shape: [1, 11], req: false, op: 'oh' };
time('net.dynamics (1 latent)', 4000, () => net.dynamics(h, oh));

const N = brain.cfg.batch;
const F = dims.pooled;
const S = dims.scalars;
const f = new Float32Array(N * F);
const s = new Float32Array(N * S);
time(`net.representBatch (N=${N})`, 200, () => net.representBatch(f, s));

for (let i = 0; i < 2000; i++) {
  brain.observe({
    frame: pooled,
    scalars: obs.scalars,
    action: i % 11,
    reward: Math.random() * 0.05 - 0.02,
    policy: null,
    value: 0,
  });
}
time('trainOnce (1 gradient step)', 30, () => brain.trainOnce());

console.log('\ncfg:', JSON.stringify({ K: brain.cfg.unroll, batch: N, sims: brain.cfg.sims, latent: brain.cfg.latentDim, hidden: brain.cfg.hiddenDim }));
