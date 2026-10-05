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
globalThis.document = {
  createElement: (tag) => new StubEl(tag),
  addEventListener: (t, fn) => (listeners[t] ||= []).push(fn),
  removeEventListener: () => {},
  pointerLockElement: null,
  body: new StubEl('body'),
};
globalThis.performance = globalThis.performance || { now: () => Date.now() };
globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(performance.now()), 16);
globalThis.URL.createObjectURL = () => 'blob:stub';
globalThis.URL.revokeObjectURL = () => {};
globalThis.confirm = () => false;
globalThis.fetch = globalThis.fetch || (async () => ({ ok: false }));

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

console.log('\nDOM/render harness completed with no exceptions.');
