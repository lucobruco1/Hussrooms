/**
 * Headless trainer, in a Web Worker.
 *
 * Runs `Session.frame()` in bursts and posts statistics back. It is the same
 * Session the browser game and the node tools use, so a checkpoint saved here is
 * loadable by `node tools/diagnose.mjs --checkpoint` and by the game itself.
 */

import { Session } from './session.js';
import { A } from '../sim/config.js';

let session = null;
let running = false;
let frames = 0;

const NAMES = Object.fromEntries(Object.entries(A).map(([k, v]) => [v, k]));
void NAMES;

function makeSession(profile, seed) {
  session = new Session({
    seed,
    learn: true,
    profile,
    autoPlayer: true, // scripted fleeing player: prey that behaves like prey
    shareBotBrains: true,
  });
  session.forceExplore = false;
}

function stats() {
  const s = session;
  const sum = s.summary();
  const hist = (b) => {
    const counts = new Array(b.cfg.actionDim).fill(0);
    for (let i = 0; i < b.replay.size; i++) counts[b.replay.at(i).action]++;
    const total = counts.reduce((a, c) => a + c, 0) || 1;
    return counts.map((c, i) => ({ a: i, p: c / total }));
  };
  return {
    t: sum.t,
    frames,
    wall: (performance.now() - wallStart) / 1000,
    clarkKills: sum.clarkKills,
    botDeaths: sum.botDeaths,
    playerDeaths: sum.playerDeaths,
    bites: sum.bites,
    lunges: sum.lunges,
    roars: sum.roars,
    firstKillAt: sum.firstKillAt,
    clark: {
      updates: s.clarkBrain.updates,
      skipped: s.clarkBrain.skippedUpdates,
      decisions: s.clarkBrain.decisions,
      replay: s.clarkBrain.replay.size,
      loss: s.clarkBrain.lossEma,
      entropy: s.clarkBrain.policyEntropy,
      value: sum.clarkValue,
      history: s.clarkBrain.history.slice(-120),
      hist: hist(s.clarkBrain),
      actionDim: s.clarkBrain.cfg.actionDim,
    },
    bot: {
      updates: s.botBrain.updates,
      replay: s.botBrain.replay.size,
      loss: s.botBrain.lossEma,
      value: sum.botValue,
      history: s.botBrain.history.slice(-120),
      hist: hist(s.botBrain),
      actionDim: s.botBrain.cfg.actionDim,
    },
    telemetry: s.world.telemetry,
    events: s.lastEvents.slice(-40),
  };
}

let wallStart = 0;
let framesPerSec = 0;

function burst() {
  if (!running || !session) return;
  const chunk = 240; // ~4 seconds of sim per tick of the message loop
  const t0 = performance.now();
  for (let i = 0; i < chunk; i++) {
    session.frame();
    frames++;
  }
  const dt = performance.now() - t0;
  framesPerSec = framesPerSec ? framesPerSec * 0.8 + (chunk / (dt / 1000)) * 0.2 : chunk / (dt / 1000);
  self.postMessage({ type: 'stats', stats: stats(), fps: framesPerSec });
  setTimeout(burst, 0);
}

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'start') {
    if (!session) {
      wallStart = performance.now();
      makeSession(msg.profile || 'live', msg.seed ?? 20260529);
    }
    running = true;
    if (!wallStart) wallStart = performance.now();
    self.postMessage({ type: 'state', state: 'running' });
    burst();
  } else if (msg.type === 'pause') {
    running = false;
    self.postMessage({ type: 'state', state: 'paused' });
  } else if (msg.type === 'checkpoint') {
    if (!session) {
      self.postMessage({ type: 'checkpoint', json: null });
      return;
    }
    self.postMessage({ type: 'checkpoint', json: JSON.stringify(session.checkpoint()) });
  } else if (msg.type === 'load') {
    try {
      const ok = session ? session.loadCheckpoint(JSON.parse(msg.json)) : false;
      if (!session) {
        wallStart = performance.now();
        makeSession(msg.profile || 'live', msg.seed ?? 20260529);
      }
      session.loadCheckpoint(JSON.parse(msg.json));
      self.postMessage({ type: 'log', text: `checkpoint loaded (${ok})` });
    } catch (err) {
      self.postMessage({ type: 'log', text: 'load failed: ' + err.message });
    }
  }
};
