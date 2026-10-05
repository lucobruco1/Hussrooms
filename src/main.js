/**
 * Hussrooms - browser entry point.
 *
 * The browser and node run the *same* simulator, the same networks and the same
 * MCTS: `src/sim/*` and `src/ai/*` are shared, and `src/render/*` only draws.
 * Whatever Cap'n Clark learns here is learnable there, and a checkpoint saved
 * from the lab panel can be handed back to node, or shipped as
 * `public/brain-pretrained.json` to show a monster who has already learned.
 *
 * In this build the player is the human. The monster has no built-in knowledge
 * of where the player is: he sees a 12x12m egocentric crop of the level and a
 * handful of scalars, and learns what to do with them from reward.
 */

import { Renderer } from './render/renderer.js';
import { Hud } from './ui/hud.js';
import { LabPanel } from './ui/lab.js';
import { Session } from './ai/session.js';
import { EZ_PROFILES, MAP, PHYS, LEARN } from './sim/config.js';

const params = new URLSearchParams(location.search);
const seed = Number(params.get('seed') || 20260529);
const learn = params.get('learn') !== '0';
const lowQuality = params.get('quality') === 'low';
const checkpointUrl = params.get('checkpoint') || 'public/brain-pretrained.json';

const app = document.getElementById('app');
const overlay = document.getElementById('overlay');
const loading = document.getElementById('loading');
const startNote = document.getElementById('start-note');

// The level layout is read by the grid generator when the Session is built.
if (params.get('layout')) MAP.layout = params.get('layout');

// The browser shares its CPU with the renderer, so it gets a smaller training
// slice per frame than the node trainer uses.
LEARN.trainBudgetMs = Number(params.get('budget') || 5);
LEARN.opsPerFrame = Number(params.get('ops') || 3);

async function fetchCheckpoint(url) {
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/**
 * A brain's architecture is baked into its weights, so a checkpoint decides
 * which profile the session has to be built with. (Loading a turbo brain into
 * a live network is rejected, not silently misinterpreted.)
 */
function profileFor(obj, fallback) {
  const c = obj?.brains?.clark?.net?.cfg;
  if (!c) return fallback;
  const t = EZ_PROFILES.turbo;
  const isTurbo = c.latentDim === t.latentDim && c.hiddenDim === t.hiddenDim && c.projDim === t.projDim;
  return isTurbo ? 'turbo' : 'live';
}

// Pre-fetch the shared pretrained brain (if one is shipped) before building the
// session, so the "Load the trained Clark" button always works.
const pretrained = await fetchCheckpoint(checkpointUrl);
const profile = params.get('profile') || profileFor(pretrained, 'live');

const session = new Session({
  seed,
  learn,
  profile,
  shareBotBrains: true,
  simSpeed: Number(params.get('speed') || 1),
});
session.forceExplore = params.get('explore') === '1';
/** the shipped brain is turbo-architecture; anything else cannot be applied */
const pretrainedUsable = !!pretrained && profileFor(pretrained, 'live') === profile;

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

let started = false;
function start() {
  if (started) return;
  started = true;
  loading.classList.add('hidden');
  requestAnimationFrame(loop);
}

const trainedBtn = document.getElementById('start-trained');
if (!pretrainedUsable) {
  trainedBtn.disabled = true;
  trainedBtn.title = 'no checkpoint at ' + checkpointUrl;
  startNote.textContent = `no trained checkpoint found at ${checkpointUrl} — train one in the lab, or start fresh`;
}

document.getElementById('start-fresh').addEventListener('click', () => {
  hud.toast(
    learn
      ? "Cap'n Clark is untrained. Watch him fail to hunt — then get better."
      : 'Learning is off: Clark is running on whatever he already knows.',
    5200,
  );
  start();
});

trainedBtn.addEventListener('click', () => {
  if (pretrainedUsable && session.loadCheckpoint(pretrained)) {
    hud.toast(
      `trained brain loaded (${session.clarkBrain.updates.toLocaleString()} gradient steps over ${Math.round(
        session.clarkBrain.decisions / 10,
      )}s of play) — this Clark has already hunted`,
      6000,
    );
  } else {
    hud.toast('no trained brain available', 3000);
  }
  start();
});

// Opening the page with ?start=1 skips the title screen (handy for links).
if (params.get('start') === '1') start();

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
  renderer.pumpInput(dt);
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

// expose for tinkering in the console
window.HUSSROOMS = { session, renderer, hud, lab, PHYS, LEARN, profile };
