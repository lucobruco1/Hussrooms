/**
 * Headless self-play trainer.
 *
 *   node tools/train.mjs --minutes 10 --profile turbo [--out public/brain]
 *
 * Runs the same Session the browser runs, but with no renderer, so the wall
 * clock goes into simulation and gradient steps instead of pixels. Periodically
 * reports the learning curve; writes a checkpoint at the end (loadable from the
 * in-game Lab panel, or shipped as the default warm start).
 */

import { Session } from '../src/ai/session.js';
import { LEARN } from '../src/sim/config.js';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? argv[i + 1] : def;
}
const minutes = Number(arg('minutes', 5));
const profile = arg('profile', 'turbo');
const seed = Number(arg('seed', 20260529));
const out = arg('out', 'public/brain');
const reportEvery = Number(arg('report', 30));
const simSpeed = Number(arg('sim-speed', LEARN.simSpeed));
const budget = Number(arg('budget', LEARN.trainBudgetMs));
const ops = Number(arg('ops', LEARN.opsPerFrame));
// The two knobs that decide the trade-off this prototype actually cares about:
// how much experience per wall second (sim-speed) versus how many gradient
// steps per frame (budget/ops). Both are logged so runs stay comparable.
LEARN.simSpeed = simSpeed;
LEARN.trainBudgetMs = budget;
LEARN.opsPerFrame = ops;

const autosaveMin = Number(arg('autosave', 10)); // wall-clock minutes between checkpoints
const session = new Session({ seed, profile, learn: true, simSpeed, autoPlayer: argv.includes('--static-player') ? false : true });
console.log(`Hussrooms trainer | profile=${profile} | seed=${seed} | budget ${minutes} min | simSpeed=${simSpeed} trainBudget=${budget}ms ops=${ops}`);
console.log(
  `map ${session.world.grid.gx}x${session.world.grid.gz}, ${session.world.grid.spawnPoints.length} spawn points, ` +
    `obs ${session.dims.pooled}+${session.dims.scalars}`,
);

const t0 = Date.now();
const deadline = t0 + minutes * 60 * 1000;
let lastReport = 0;
let frames = 0;
let lastFrameWall = t0;
let lastAutosave = 0;
let wallPerSimSec = 0;

while (Date.now() < deadline) {
  session.frame();
  frames++;
  const t = session.world.t;
  if (t - lastReport >= reportEvery) {
    const now = Date.now();
    wallPerSimSec = (now - lastFrameWall) / 1000 / Math.max(1e-6, t - lastReport);
    lastFrameWall = now;
    lastReport = t;
    const s = session.summary();
    const mins = (now - t0) / 60000;
    // top-3 action mix: with sparse-reward hunting, *what the policy does* is
    // the number that moves first, before bites show up at all.
    const mix = (hist) =>
      hist
        .slice()
        .sort((a, b) => b.p - a.p)
        .slice(0, 3)
        .map((h) => `${h.action.slice(0, 5)} ${(h.p * 100).toFixed(0)}%`)
        .join(' ');
    console.log(
      `[${mins.toFixed(1)}m wall | t=${s.t.toFixed(0)}s sim] kills=${s.clarkKills} bites=${s.bites} roars=${s.roars} ` +
        `| upd ${s.botUpdates}/${s.clarkUpdates} loss ${s.botLoss.toFixed(2)}/${s.clarkLoss.toFixed(2)} ` +
        `| V ${s.botValue.toFixed(2)}/${s.clarkValue.toFixed(2)} H ${s.botEntropy.toFixed(2)}/${s.clarkEntropy.toFixed(2)} ` +
        `| ${wallPerSimSec.toFixed(1)}s wall per sim-s (${(frames / (now - t0) * 1000).toFixed(0)} frames/s)`,
    );
    console.log(`  clark mix: ${mix(session.actionHistogram('clark'))}   |   bot mix: ${mix(session.actionHistogram('bots'))}`);
    // Autosave: a long run on a shared machine must not lose its brain to a
    // timeout, and being able to inspect an in-progress checkpoint is the
    // difference between "it learned" and "it was still learning".
    if (autosaveMin > 0 && mins - lastAutosave >= autosaveMin) {
      lastAutosave = mins;
      mkdirSync(dirname(out), { recursive: true });
      writeFileSync(`${out}-${Math.round(s.t)}s.json`, JSON.stringify(session.checkpoint()));
      console.log(`  [autosave] ${out}-${Math.round(s.t)}s.json`);
    }
  }
}

const s = session.summary();
console.log('\n--- training summary ---');
console.log(
  JSON.stringify(
    {
      wallMinutes: +((Date.now() - t0) / 60000).toFixed(2),
      simTime: +s.t.toFixed(1),
      simSteps: s.simSteps,
      clarkKills: s.clarkKills,
      bites: s.bites,
      roars: s.roars,
      lunges: s.lunges,
      botDeaths: s.botDeaths,
      playerDeaths: s.playerDeaths,
      clarkDeaths: session.stats.clarkDeaths,
      firstKillAt: s.firstKillAt,
      updates: { bots: s.botUpdates, clark: s.clarkUpdates },
      replay: { bots: s.botReplay, clark: s.clarkReplay },
      losses: { bots: +s.botLoss.toFixed(3), clark: +s.clarkLoss.toFixed(3) },
      entropy: { bots: +s.botEntropy.toFixed(3), clark: +s.clarkEntropy.toFixed(3) },
      values: { bots: +s.botValue.toFixed(3), clark: +s.clarkValue.toFixed(3) },
      frames,
    },
    null,
    2,
  ),
);
console.log('\nclark action mix:', session.actionHistogram('clark').map((a) => `${a.action}:${a.p.toFixed(2)}`).join(' '));
console.log('bot   action mix:', session.actionHistogram('bots').map((a) => `${a.action}:${a.p.toFixed(2)}`).join(' '));

if (out) {
  const cp = session.checkpoint();
  cp.profile = profile;
  cp.training = {
    wallMinutes: +((Date.now() - t0) / 60000).toFixed(2),
    simTime: +s.t.toFixed(1),
    clarkKills: s.clarkKills,
    bites: s.bites,
    firstKillAt: s.firstKillAt,
  };
  const file = `${out}-${profile}.json`;
  mkdirSync(dirname(file), { recursive: true });
  const json = JSON.stringify(cp);
  writeFileSync(file, json);
  console.log(`\ncheckpoint written: ${file} (${(json.length / 1024).toFixed(0)} KB)`);
}
