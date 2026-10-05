/**
 * Hussrooms - browser entry point.
 *
 * The browser and node run the *same* simulator, the same networks and the same
 * MCTS: `src/sim/*` and `src/ai/*` are shared, and `src/render/*` only draws.
 * Whatever Cap'n Clark learns here is learnable there, and a checkpoint saved
 * from the lab panel can be dropped into `public/brain.json` to give the
 * monster a head start (or, as shipped, to show him freshly born and harmless).
 *
 * The player in this build is the human. The monster has no built-in knowledge
 * of where the player is: he sees a 12x12m egocentric crop of the level and a
 * handful of scalars, and learns what to do with them from reward.
 */

import { Renderer } from './render/renderer.js';
import { Hud } from './ui/hud.js';
import { LabPanel } from './ui/lab.js';
import { Session } from './ai/session.js';
import { PHYS, LEARN } from './sim/config.js';

const params = new URLSearchParams(location.search);
const seed = Number(params.get('seed') || 20260529);
const learn = params.get('learn') !== '0';
const lowQuality = params.get('quality') === 'low';
const profile = params.get('profile') || 'live';
const checkpointUrl = params.get('checkpoint');

const app = document.getElementById('app');
const overlay = document.getElementById('overlay');
const loading = document.getElementById('loading');

// The browser shares its CPU with the renderer, so it gets a smaller training
// slice per frame than the node trainer uses.
LEARN.trainBudgetMs = Number(params.get('budget') || 5);
LEARN.opsPerFrame = Number(params.get('ops') || 3);

const session = new Session({
  seed,
  learn,
  profile,
  shareBotBrains: true,
  simSpeed: Number(params.get('speed') || 1),
});
session.forceExplore = params.get('explore') === '1';

const hud = new Hud(session, overlay);
const renderer = new Renderer(app, session, { lowQuality });
const lab = new LabPanel(session, session.clarkBrain, document.body);

window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Tab') {
    ev.preventDefault();
    lab.toggle();
  }
  if (ev.code === 'KeyP') session.learn = !session.learn;
});

// Optional head start: a checkpoint produced by tools/train.mjs.
if (checkpointUrl) {
  fetch(checkpointUrl)
    .then((r) => (r.ok ? r.json() : null))
    .then((obj) => {
      if (obj && session.loadCheckpoint(obj)) hud.toast('checkpoint loaded — Clark remembers');
    })
    .catch(() => {});
} else {
  fetch('public/brain.json', { method: 'HEAD' })
    .then((r) => {
      if (!r.ok) return null;
      return fetch('public/brain.json').then((x) => x.json());
    })
    .then((obj) => {
      if (obj && session.loadCheckpoint(obj)) hud.toast('pretrained brain loaded from public/brain.json');
    })
    .catch(() => {});
}

let last = performance.now();
let frames = 0;
let fps = 60;
let slowFrames = 0;
let wasAlive = true;
let degraded = false;

function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  fps = fps * 0.92 + (1 / Math.max(dt, 1e-3)) * 0.08;
  renderer.pumpInput();
  session.frame(); // fixed-timestep sim + one round of gradient updates
  renderer.sync(dt);
  renderer.render();
  frames++;

  // death / respawn feedback
  const alive = session.world.player.alive;
  if (wasAlive && !alive) hud.toast("Cap'n Clark killed you — watching him until you respawn", 3000);
  if (!wasAlive && alive) hud.toast('respawned', 1200);
  wasAlive = alive;

  // If the machine cannot keep up, give the renderer its time back rather than
  // silently dropping the simulation rate.
  if (fps < 24) slowFrames++;
  else slowFrames = Math.max(0, slowFrames - 1);
  if (!degraded && slowFrames > 180) {
    degraded = true;
    LEARN.trainBudgetMs = 2;
    LEARN.opsPerFrame = 1;
    hud.toast('running slowly — reduced the training budget per frame (P toggles learning)', 4200);
  }

  if (frames % 4 === 0) {
    hud.update(fps);
    lab.update();
  }
  requestAnimationFrame(loop);
}

loading.classList.add('hidden');
hud.toast(
  learn
    ? "Cap'n Clark is untrained. Watch him fail to hunt — then get better."
    : 'Learning is off: Clark is running on whatever he already knows.',
  5200,
);
requestAnimationFrame(loop);

// expose for tinkering in the console
window.HUSSROOMS = { session, renderer, hud, lab, PHYS, LEARN };
