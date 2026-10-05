/**
 * Behaviour diagnostics: what is each agent actually doing?
 *   node tools/diagnose.mjs [simSeconds] [--learn] [--checkpoint path.json]
 */
import { Session } from '../src/ai/session.js';
import { ACTIONS } from '../src/sim/config.js';
import { readFileSync } from 'node:fs';

const argv = process.argv.slice(2);
const seconds = Number(argv[0]) || 60;
const learn = argv.includes('--learn');
const cpIdx = argv.indexOf('--checkpoint');
const session = new Session({ seed: 20260529, profile: 'turbo', learn, autoPlayer: true });
if (cpIdx >= 0) {
  const ok = session.loadCheckpoint(JSON.parse(readFileSync(argv[cpIdx + 1], 'utf8')));
  console.log('checkpoint loaded:', ok);
}
if (!learn && !argv.includes('--keep-explore')) {
  // force the greedy/MCTS path (no random warmup) for evaluation
  session.botBrain.cfg.warmupActions = 0;
  session.clarkBrain.cfg.warmupActions = 0;
  session.botBrain.cfg.exploreEps = 0.02;
  session.clarkBrain.cfg.exploreEps = 0.02;
}

const start = session.world.entities.map((e) => ({ id: e.id, kind: e.kind, x: e.x, z: e.z, dist: 0, novelty: 0, wall: 0 }));
const t0 = Date.now();
const steps = Math.round(seconds * 60);
for (let i = 0; i < steps; i++) session.frame();

for (const e of session.world.entities) {
  const s = start.find((x) => x.kind === e.kind && x.id === e.id) || {};
  void s;
}
console.log(`ran ${seconds}s of sim in ${((Date.now() - t0) / 1000).toFixed(1)}s wall`);
const sum = session.summary();
const tel = session.world.telemetry;
console.log(`kills=${sum.clarkKills} bites=${sum.bites} roars=${sum.roars} lunges=${sum.lunges} botDeaths=${sum.botDeaths}`);
console.log(
  `contact: clarkSees=${tel.clarkSees} ticks (${((tel.clarkSees / Math.max(1, tel.steps)) * 100).toFixed(2)}%)  ` +
    `clarkClosest=${Number.isFinite(tel.clarkClosest) ? tel.clarkClosest.toFixed(1) : '-'}m  botSees=${tel.botSees}  ` +
    `firstKillAt=${sum.firstKillAt ?? '-'}  updates=${sum.updates}`,
);
console.log('loss components clark:', JSON.stringify(session.clarkBrain.lastLosses));
console.log('loss components bots :', JSON.stringify(session.botBrain.lastLosses));
console.log('\nagent              travelled  novelty  lastDistToPrey  value   alive');
for (const e of session.world.entities) {
  const brain = session.brains.get(e.id);
  let nearest = Infinity;
  for (const o of session.world.entities) {
    if (o === e || !o.alive) continue;
    if (o.faction === e.faction) continue;
    nearest = Math.min(nearest, Math.hypot(o.x - e.x, o.z - e.z));
  }
  console.log(
    `${(e.kind + '#' + e.id).padEnd(18)} ${e.stats.distance.toFixed(0).padStart(6)}m  ${String(e.stats.novelty).padStart(6)}  ` +
      `${(nearest === Infinity ? '-' : nearest.toFixed(1)).padStart(10)}  ${(brain && brain._lastSearch ? brain._lastSearch.value : 0).toFixed(2).padStart(6)}   ${e.alive}`,
  );
}
console.log('\nclark replay:', JSON.stringify(session.brainStats('clark')));
console.log('bots  replay:', JSON.stringify(session.brainStats('bots')));
console.log('\nclark action mix:', session.actionHistogram('clark').map((a, i) => `${ACTIONS[i].name}:${a.p.toFixed(2)}`).join(' '));
console.log('bots  action mix:', session.actionHistogram('bots').map((a, i) => `${ACTIONS[i].name}:${a.p.toFixed(2)}`).join(' '));
