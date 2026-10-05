/**
 * UI smoke test without a browser.
 *
 *   node tools/ui-smoke.mjs
 *
 * Stubs just enough DOM and 2D-canvas to load every browser module in node and
 * drive the HUD, lab panel and renderer synchronisation against a live Session
 * (including the spectator camera after a death, checkpoint round-trip and the
 * brain-reset path). It cannot validate pixels — only that the module graph
 * imports and that no code path touches a missing field or API.
 */

import { readFileSync } from 'node:fs';

class StubCtx {
  constructor() {
    this.fillStyle = '';
    this.strokeStyle = '';
    this.font = '';
    this.lineWidth = 1;
  }
  clearRect() {}
  fillRect() {}
  strokeRect() {}
  fillText() {}
  beginPath() {}
  moveTo() {}
  lineTo() {}
  stroke() {}
  fill() {}
  arc() {}
  createRadialGradient() {
    return { addColorStop() {} };
  }
  createLinearGradient() {
    return { addColorStop() {} };
  }
  getImageData(x, y, w, h) {
    return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h };
  }
  putImageData() {}
}

class StubEl {
  constructor(tag = 'div') {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.files = [];
    this.width = 0;
    this.height = 0;
    this.textContent = '';
    this.tabIndex = -1;
    this._html = '';
    this._listeners = {};
    this._q = new Map();
    const set = new Set();
    this.classList = {
      add: (c) => set.add(c),
      remove: (c) => set.delete(c),
      contains: (c) => set.has(c),
      toggle: (c, on) => (on === undefined ? (set.has(c) ? set.delete(c) : set.add(c)) : on ? set.add(c) : set.delete(c)),
      _set: set,
    };
  }
  appendChild(c) {
    this.children.push(c);
    return c;
  }
  removeChild(c) {
    this.children = this.children.filter((x) => x !== c);
  }
  remove() {}
  addEventListener(type, fn) {
    (this._listeners[type] ||= []).push(fn);
  }
  removeEventListener() {}
  querySelector(sel) {
    if (!this._q.has(sel)) this._q.set(sel, new StubEl('div'));
    return this._q.get(sel);
  }
  querySelectorAll() {
    return [];
  }
  getContext() {
    return new StubCtx();
  }
  getBoundingClientRect() {
    return { width: 1280, height: 720, left: 0, top: 0 };
  }
  requestPointerLock() {}
  set innerHTML(v) {
    this._html = String(v);
  }
  get innerHTML() {
    return this._html;
  }
}

const listeners = {};
globalThis.window = {
  innerWidth: 1280,
  innerHeight: 720,
  devicePixelRatio: 1,
  addEventListener: (t, fn) => (listeners[t] ||= []).push(fn),
  removeEventListener: () => {},
};
const byId = new Map();
globalThis.document = {
  createElement: (tag) => new StubEl(tag),
  getElementById: (id) => {
    if (!byId.has(id)) byId.set(id, new StubEl('div'));
    return byId.get(id);
  },
  addEventListener: (t, fn) => (listeners[t] ||= []).push(fn),
  removeEventListener: () => {},
  pointerLockElement: null,
  body: new StubEl('body'),
};
globalThis.location = { search: '?start=1&quality=low' }; // drives the real frame loop
globalThis.performance = globalThis.performance || { now: () => Date.now() };
// The real frame loop calls WebGLRenderer.render(), and there is no WebGL here,
// so the loop is not allowed to schedule itself: importing src/main.js still
// exercises the wiring (params, element lookups, session/renderer/hud/lab
// construction, the start-screen buttons), and the per-frame functions are
// driven by hand further down.
globalThis.requestAnimationFrame = () => 0;
globalThis.URL.createObjectURL = () => 'blob:stub';
globalThis.URL.revokeObjectURL = () => {};
globalThis.confirm = () => false;
// Serve the real shipped checkpoint through the fake fetch, so the browser
// code path that derives its architecture profile and installs it is tested.
globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.includes('brain-pretrained.json')) {
    const body = readFileSync(new URL('../public/brain-pretrained.json', import.meta.url), 'utf8');
    return { ok: true, json: async () => JSON.parse(body) };
  }
  return { ok: false, json: async () => null };
};

const THREE = await import('../src/three-shim.js');
console.log('three.js version:', THREE.REVISION);

const { Session } = await import('../src/ai/session.js');
const { WorldView } = await import('../src/render/worldview.js');
const { ClarkMesh } = await import('../src/render/clark.js');
const { SurvivorMesh, PlayerHands } = await import('../src/render/actors.js');
const { Hud } = await import('../src/ui/hud.js');
const { LabPanel } = await import('../src/ui/lab.js');
const { Renderer } = await import('../src/render/renderer.js');

// ---------------------------------------------------------------- cheap bits
const session = new Session({
  seed: 777,
  learn: true,
  profile: 'live',
  simSpeed: 1,
  clarkCfg: { warmupActions: 2, sims: 8, maxDepth: 5 },
  botCfg: { warmupActions: 2, sims: 6, maxDepth: 4 },
});
session.forceExplore = true;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x2a2410, 0.055); // the real Renderer ctor sets this
const worldView = new WorldView(session.world.grid, scene);
console.log('WorldView: children =', worldView.group.children.length);

const clark = new ClarkMesh();
const bot = new SurvivorMesh(2);
const camera = new THREE.PerspectiveCamera(78, 1.7, 0.05, 120);
const hands = new PlayerHands(camera);

// ---------------------------------------------------------- renderer internals
const renderer = Object.create(Renderer.prototype);
Object.assign(renderer, {
  session,
  world: session.world,
  container: new StubEl('div'),
  lowQuality: false,
  scene,
  camera,
  actorMeshes: new Map(),
  keys: new Set(['KeyW', 'ShiftLeft']),
  pitch: 0.1,
  pointerLocked: false,
  firing: 0,
  mouseSensitivity: 0.002,
  torch: new THREE.SpotLight(),
  hands,
});
renderer.session.world.entities.forEach((e) => renderer.ensureMesh(e));
renderer.pumpInput();
console.log('pumpInput -> action', JSON.stringify(session.playerInput));

// run the sim for a few seconds and drive every visual sync path
for (let i = 0; i < 60 * 12; i++) session.step();
for (let i = 0; i < 30; i++) session.frame(); // sim + learning + rolling rates
renderer.sync(1 / 60);
for (const [id, m] of renderer.actorMeshes) {
  const e = session.world.entities.find((x) => x.id === id);
  if (e) m.update(1 / 60, e);
}
renderer.camera.updateMatrixWorld(true);
console.log('after 6s sim: t =', session.world.t.toFixed(1), 'clark pos', session.world.clark.x.toFixed(1), session.world.clark.z.toFixed(1));
console.log('camera matrix ok:', Number.isFinite(renderer.camera.matrixWorld.elements[12]));

// kill the player and re-sync: spectator camera path
session.world.player.alive = false;
session.world.player.respawnTimer = session.world.t + 999;
renderer.sync(1 / 60);
renderer.sync(1 / 60);
console.log('spectating:', renderer.spectating, 'cam y', renderer.camera.position.y.toFixed(2));
session.world.player.alive = true;
renderer.sync(1 / 60);
console.log('back to first person:', !renderer.spectating, 'eye height', renderer.camera.position.y.toFixed(2));

// ---------------------------------------------------------------- DOM layers
const overlay = new StubEl('div');
const hud = new Hud(session, overlay);
hud.update();
hud.toast('hello');
console.log('hud html lengths:', hud.el.status.innerHTML.length, hud.el.vitals.innerHTML.length, hud.el.learning.innerHTML.length);

const lab = new LabPanel(session, session.clarkBrain, new StubEl('body'));
lab.toggle(true);
lab.update();
console.log('lab trace children:', session.clarkBrain.lastTrace ? session.clarkBrain.lastTrace.children.length : 'none');
console.log('lab rollout points:', session.clarkBrain.lastTrace ? session.clarkBrain.lastTrace.rollout.length : 'none');
console.log('lab foot:', lab.footEl.textContent.slice(0, 110));
const tr = session.clarkBrain.lastTrace;
if (tr) {
  console.log('trace: action', tr.action, 'top children', tr.children.slice(0, 4).map((c) => `${c.a}:n${c.n}/q${c.q.toFixed(2)}/p${c.prior.toFixed(2)}`).join(' '));
  console.log('trace rollout', tr.rollout.map((r) => `r${r.reward.toFixed(2)}v${r.value.toFixed(2)}`).join(' '));
  console.log('trace finite:', tr.children.every((c) => Number.isFinite(c.q)) && tr.rollout.every((r) => Number.isFinite(r.value)));
} else {
  console.log('trace: NONE (bug)');
}
console.log('learning: clark updates', session.clarkBrain.updates, 'skipped', session.clarkBrain.skippedUpdates, 'lossEma', session.clarkBrain.lossEma.toFixed(3));
console.log('losses', JSON.stringify(session.clarkBrain.lastLosses, (k, v) => (typeof v === 'number' ? +v.toFixed(4) : v)));

// checkpoint round trip through the UI code path
const cp = session.checkpoint();
console.log('checkpoint bytes ~', JSON.stringify(cp).length, 'format', cp.format);
const session2 = new Session({ seed: 777, learn: true, profile: 'live' });
console.log('loadCheckpoint:', session2.loadCheckpoint(cp));
const s2 = session2.summary();
console.log('restored updates:', s2.clarkUpdates, 'replay:', s2.clarkReplay);

// reset button path
session.resetLearning();
console.log('after resetLearning, clark updates:', session.clarkBrain.updates, 'firstKillAt:', session.stats.firstKillAt);

// ------------------------------------------------- the real entry point
await import('../src/main.js');
const H = globalThis.window.HUSSROOMS;
console.log('main.js booted:', !!H, '| entry session t =', H?.session.world.t.toFixed(2), '| profile =', H?.session.profile);
// drive the real frame body (everything except the GL call)
for (let i = 0; i < 90; i++) {
  H.renderer.pumpInput(1 / 60);
  H.session.frame();
  H.renderer.sync(1 / 60);
  H.hud.update(60);
  H.lab.update();
}
const cam = H.renderer.camera.position;
console.log('main.js frame body ran 90x | t =', H.session.world.t.toFixed(1), '| camera finite:', Number.isFinite(cam.x) && Number.isFinite(cam.y));
console.log('loading overlay hidden:', byId.get('loading')?.classList.contains('hidden'));
console.log(
  'start buttons wired:',
  (byId.get('start-fresh')?._listeners.click || []).length,
  (byId.get('start-trained')?._listeners.click || []).length,
);
console.log('profile chosen for the shipped checkpoint:', H.profile, '| trained updates before click:', H.session.clarkBrain.updates);
for (const fn of byId.get('start-trained')._listeners.click || []) fn();
for (let i = 0; i < 30; i++) {
  H.renderer.pumpInput(1 / 60);
  H.session.frame();
  H.renderer.sync(1 / 60);
}
console.log(
  'after clicking "load trained Clark": updates =',
  H.session.clarkBrain.updates,
  '| value finite:',
  Number.isFinite(H.clarkBrainValue ?? H.session.summary().clarkValue),
  '| sim t =',
  H.session.world.t.toFixed(1),
);

console.log('\nDOM/render harness completed with no exceptions.');
