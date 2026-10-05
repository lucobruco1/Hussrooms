/**
 * Headless smoke test / short self-play run.
 *
 *   node tools/smoke.mjs [seconds] [--fast]
 *
 * Runs the exact simulation + EfficientZero loop the browser uses, with the
 * renderer absent. Prints per-agent learning telemetry and per-second timing so
 * you can tell instantly whether the agent is alive, learning, and affordable.
 */

import { Session } from '../src/ai/session.js';
import { LEARN } from '../src/sim/config.js';

const args = process.argv.slice(2);
const seconds = Number(args[0]) || 20;
const fast = args.includes('--fast');

const t0 = Date.now();
const session = new Session({ seed: 20260529, learn: true });
console.log(
  `map: ${session.world.grid.gx}x${session.world.grid.gz} cells, ${session.world.grid.props.length} props, ` +
    `${session.dims.pooled}+${session.dims.scalars} obs dims per agent`,
);
console.log(`clark spawn: ${session.world.clark.x.toFixed(1)},${session.world.clark.z.toFixed(1)}  bots: ${session.world.entities
  .filter((e) => e.kind === 'bot')
  .map((b) => `${b.x.toFixed(0)},${b.z.toFixed(0)}`)
  .join(' ')}`);

const steps = Math.round(seconds * 60);
let lastReport = 0;
const tSim0 = Date.now();
for (let i = 0; i < steps; i++) {
  session.frame();
  const simT = session.world.t;
  if (simT - lastReport >= 5) {
    lastReport = simT;
    const s = session.summary();
    const wall = (Date.now() - tSim0) / 1000;
    console.log(
      `t=${s.t.toFixed(0).padStart(4)}s  clarkKills=${String(s.clarkKills).padStart(3)}  bites=${String(s.bites).padStart(
        4,
      )}  roars=${String(s.roars).padStart(3)}  botsAlive=${s.botsAlive}/4  |  ` +
        `updates ${s.botUpdates}/${s.clarkUpdates}  loss ${s.botLoss.toFixed(3)}/${s.clarkLoss.toFixed(
          3,
        )}  replay ${s.botReplay}/${s.clarkReplay}  V ${s.botValue.toFixed(2)}/${s.clarkValue.toFixed(2)}  ` +
        `x${(s.t / wall).toFixed(1)} realtime`,
    );
  }
}
const s = session.summary();
console.log('\n--- final ---');
console.log(
  JSON.stringify(
    {
      simTime: +s.t.toFixed(1),
      wallTime: +((Date.now() - t0) / 1000).toFixed(1),
      simSteps: s.simSteps,
      clarkKills: s.clarkKills,
      playerDeaths: s.playerDeaths,
      botDeaths: s.botDeaths,
      bites: s.bites,
      roars: s.roars,
      lunges: s.lunges,
      clarkDeaths: session.stats.clarkDeaths,
      updates: { bots: s.botUpdates, clark: s.clarkUpdates },
      replay: { bots: s.botReplay, clark: s.clarkReplay },
      losses: { bots: +s.botLoss.toFixed(4), clark: +s.clarkLoss.toFixed(4) },
      entropy: { bots: +s.botEntropy.toFixed(3), clark: +s.clarkEntropy.toFixed(3) },
      firstKillAt: s.firstKillAt,
      trainMsPerFrame: +(s.trainMs).toFixed(2),
    },
    null,
    2,
  ),
);
console.log('\nclark action mix:', session.actionHistogram('clark').map((a) => `${a.action}:${a.p.toFixed(2)}`).join(' '));
console.log('bot   action mix:', session.actionHistogram('bots').map((a) => `${a.action}:${a.p.toFixed(2)}`).join(' '));
if (session.stats.firstKillAt != null) {
  console.log(`\nfirst kill at t=${session.stats.firstKillAt.toFixed(1)}s of simulated time`);
} else {
  console.log('\nno kill yet in this window (normal early on - see docs/LEARNING.md)');
}
void fast;
void LEARN;
