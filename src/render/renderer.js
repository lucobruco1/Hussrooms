/**
 * The viewer: first-person camera, the level, the actors, and input.
 *
 * This file contains no gameplay logic. It reads the simulator's state and
 * draws it, and writes player input back into the simulator. Bots and Cap'n
 * Clark keep behaving identically with the renderer switched off (that is
 * exactly what tools/train.mjs does).
 */

import * as THREE from '../three-shim.js';
import { A } from '../sim/config.js';
import { WorldView } from './worldview.js';
import { ClarkMesh } from './clark.js';
import { PlayerHands, SurvivorMesh } from './actors.js';

export class Renderer {
  constructor(container, session, opts = {}) {
    this.session = session;
    this.world = session.world;
    this.container = container;
    this.lowQuality = !!opts.lowQuality;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1608);
    this.scene.fog = new THREE.FogExp2(0x2a2410, 0.055);

    this.camera = new THREE.PerspectiveCamera(opts.fov || 78, 1, 0.05, 120);
    this.scene.add(this.camera);

    // Lighting: flat, sourceless fluorescence. The Backrooms have no shadows.
    this.scene.add(new THREE.AmbientLight(0xfff2c0, 0.85));
    const hemi = new THREE.HemisphereLight(0xfff0b8, 0x6a5a24, 0.7);
    this.scene.add(hemi);
    const torch = new THREE.SpotLight(0xfff0c8, 1.1, 22, Math.PI / 5, 0.45, 1.4);
    torch.position.set(0.2, -0.1, 0.1);
    this.camera.add(torch);
    this.torch = torch;

    this.worldView = new WorldView(this.world.grid, this.scene);

    this.actorMeshes = new Map();
    this.botOrdinal = 0;
    for (const e of this.world.entities) this.ensureMesh(e);
    this.clarkMesh = new ClarkMesh();
    this.scene.add(this.clarkMesh.group);
    this.actorMeshes.set(this.world.clark.id, this.clarkMesh);
    this.hands = new PlayerHands(this.camera);
    this.hands.group.visible = !this.lowQuality;

    // --- input state
    this.keys = new Set();
    this.pitch = 0;
    this.pointerLocked = false;
    this.firing = 0;
    this.mouseSensitivity = 0.0022;
    this.bindInput(container);

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  ensureMesh(e) {
    if (e.kind === 'player') return null; // first person: the player has no body
    if (this.actorMeshes.has(e.id)) return this.actorMeshes.get(e.id);
    if (e.kind === 'clark') {
      const m = new ClarkMesh();
      this.scene.add(m.group);
      this.actorMeshes.set(e.id, m);
      this.clarkMesh = m;
      return m;
    }
    const m = new SurvivorMesh(this.botOrdinal++);
    this.scene.add(m.group);
    this.actorMeshes.set(e.id, m);
    return m;
  }

  resize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.rendererSize = { w, h };
  }

  bindInput(el) {
    el.tabIndex = 0;
    el.addEventListener('click', () => {
      if (!this.pointerLocked) el.requestPointerLock?.();
    });
    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === el;
    });
    // Look control has to work in two worlds: a normal tab (pointer lock) and
    // an embedded preview iframe, where pointer lock may be unavailable. In the
    // second case, holding the left mouse button and dragging turns the view,
    // and the arrow keys turn it without any mouse at all.
    this.dragging = false;
    document.addEventListener('mousemove', (ev) => {
      if (!this.pointerLocked && !this.dragging) return;
      const p = this.world.player;
      if (!p || !p.alive) return;
      p.yaw -= ev.movementX * this.mouseSensitivity * (this.pointerLocked ? 1 : 1.6);
      this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch - ev.movementY * this.mouseSensitivity));
    });
    window.addEventListener('mouseup', () => {
      this.dragging = false;
    });
    window.addEventListener('blur', () => {
      this.dragging = false;
      this.keys.clear();
    });
    window.addEventListener('keydown', (ev) => {
      this.keys.add(ev.code);
      if (ev.code === 'Space') {
        this.firing = 0.12;
        this.session.playerShoot();
      }
      if (
        ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ShiftLeft', 'Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(
          ev.code,
        )
      )
        ev.preventDefault();
    });
    window.addEventListener('keyup', (ev) => this.keys.delete(ev.code));
    window.addEventListener('mousedown', (ev) => {
      if (ev.button === 0) {
        this.dragging = true;
        this.firing = 0.12;
        this.session.playerShoot();
      }
    });
  }

  /** Read the keyboard and push a discrete action into the simulator. */
  pumpInput(dt = 1 / 60) {
    const k = this.keys;
    const p = this.world.player;
    // keyboard turning: works everywhere, including iframes that refuse
    // pointer lock
    if (p && p.alive) {
      const turn = (k.has('ArrowLeft') || k.has('KeyQ') ? 1 : 0) - (k.has('ArrowRight') || k.has('KeyE') ? 1 : 0);
      if (turn) p.yaw += turn * 2.2 * dt;
      const look = (k.has('ArrowUp') ? 1 : 0) - (k.has('ArrowDown') ? 1 : 0);
      if (look) this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + look * 1.4 * dt));
    }
    const fwd = k.has('KeyW') ? 1 : k.has('KeyS') ? -1 : 0;
    const strafe = k.has('KeyD') ? 1 : k.has('KeyA') ? -1 : 0;
    let action = A.idle;
    if (fwd > 0 && strafe > 0) action = A.fwdR;
    else if (fwd > 0 && strafe < 0) action = A.fwdL;
    else if (fwd < 0 && strafe > 0) action = A.backR;
    else if (fwd < 0 && strafe < 0) action = A.backL;
    else if (fwd > 0) action = A.fwd;
    else if (fwd < 0) action = A.back;
    else if (strafe > 0) action = A.strafeR;
    else if (strafe < 0) action = A.strafeL;
    this.session.setPlayerInput({ action, sprint: k.has('ShiftLeft') || k.has('ShiftRight') });
  }

  /** Copy simulator state into the scene graph. */
  sync(dt) {
    const w = this.world;
    const t = w.t;
    for (const e of w.entities) {
      const mesh = this.ensureMesh(e);
      if (!mesh) continue;
      if (e.kind === 'player') continue; // first person: no body
      mesh.group.position.set(e.x, 0, e.z);
      mesh.group.rotation.y = e.yaw;
      mesh.update(dt, e, t);
      // Clark is a Still Life: when he is destroyed he is simply not there
      if (e.kind === 'clark') mesh.group.visible = e.alive;
    }
    // player camera (or a spectator when the player is dead)
    const p = w.player;
    const eye = 1.62;
    if (p.alive) {
      this.camera.position.set(p.x, eye, p.z);
      this.camera.rotation.set(this.pitch, p.yaw + Math.PI, 0, 'YXZ');
      this.spectating = false;
    } else {
      // Being killed by Clark should be a *reward*, not a black screen: drift
      // backwards out of the corpse's eyes and watch him. This also happens to
      // be the best way to see what he actually learned to do.
      const c = w.clark;
      this.spectating = true;
      if (c && c.alive) {
        const dx = this.camera.position.x - c.x;
        const dz = this.camera.position.z - c.z;
        const len = Math.hypot(dx, dz) || 1;
        const want = new THREE.Vector3(c.x + (dx / len) * 7, 3.1, c.z + (dz / len) * 7);
        this.camera.position.lerp(want, Math.min(1, dt * 1.5));
        this.camera.up.set(0, 1, 0);
        this.camera.lookAt(c.x, 1.9, c.z);
      } else {
        this.camera.position.y = Math.min(4.5, this.camera.position.y + dt * 2);
      }
    }
    // heavy breathing when close to Clark
    const cp = w.clark;
    if (cp && cp.alive) {
      const d = Math.hypot(cp.x - p.x, cp.z - p.z);
      const prox = Math.max(0, 1 - d / 16);
      this.torch.intensity = 1.05 + prox * 0.5;
      this.scene.fog.density = 0.05 + prox * 0.075;
    }
    this.hands.update(dt, this.firing > 0);
    this.firing = Math.max(0, this.firing - dt);
  }

  render() {
    if (!this.renderer) {
      this.renderer = new THREE.WebGLRenderer({ antialias: !this.lowQuality, powerPreference: 'high-performance' });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.lowQuality ? 1 : 1.5));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.container.appendChild(this.renderer.domElement);
    }
    const { w, h } = this.rendererSize || { w: window.innerWidth, h: window.innerHeight };
    this.renderer.setSize(w, h, false);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.renderer?.dispose();
    this.renderer?.domElement?.remove();
  }
}
