/**
 * Survivor bodies and the player's own hands.
 *
 * The four bots are visually distinct (colour + name tag) so you can tell which
 * brain is doing what while they learn. Their animation is driven from the
 * simulation state: speed drives the walk cycle, daze/stun slumps the body.
 */

import * as THREE from '../three-shim.js';
import { PHYS } from '../sim/config.js';

const SHIRTS = [0xc4553a, 0x3f6fbf, 0x4a8f4a, 0xb08a3a];
const SKIN = 0xc79a76;

export class SurvivorMesh {
  constructor(index = 0) {
    this.group = new THREE.Group();
    this.index = index;
    const shirt = SHIRTS[index % SHIRTS.length];

    this.hips = new THREE.Group();
    this.hips.position.y = 0.92;
    this.group.add(this.hips);

    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.62, 0.26), new THREE.MeshLambertMaterial({ color: shirt }));
    torso.position.y = 0.22;
    this.hips.add(torso);
    const hipsBox = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.22, 0.24), new THREE.MeshLambertMaterial({ color: 0x2e3138 }));
    this.hips.add(hipsBox);

    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 10, 8), new THREE.MeshLambertMaterial({ color: SKIN }));
    this.head.position.y = 0.68;
    this.hips.add(this.head);
    // a cap so you can tell which way they face
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.06, 8), new THREE.MeshLambertMaterial({ color: 0x22262b }));
    cap.position.y = 0.79;
    this.hips.add(cap);

    const armGeo = new THREE.BoxGeometry(0.12, 0.58, 0.12);
    const armMat = new THREE.MeshLambertMaterial({ color: shirt });
    this.armL = new THREE.Mesh(armGeo, armMat);
    this.armL.position.set(-0.29, 0.42, 0);
    this.hips.add(this.armL);
    this.armR = new THREE.Mesh(armGeo, armMat);
    this.armR.position.set(0.29, 0.42, 0);
    this.hips.add(this.armR);

    const legGeo = new THREE.BoxGeometry(0.15, 0.86, 0.15);
    const legMat = new THREE.MeshLambertMaterial({ color: 0x39393f });
    this.legL = new THREE.Mesh(legGeo, legMat);
    this.legL.position.set(-0.12, -0.5, 0);
    this.hips.add(this.legL);
    this.legR = new THREE.Mesh(legGeo, legMat);
    this.legR.position.set(0.12, -0.5, 0);
    this.hips.add(this.legR);

    this.phase = Math.random() * 6.28;
    this.deadSince = null;
  }

  update(dt, e, worldT = 0) {
    const speed = Math.hypot(e.vx, e.vz);
    const gait = Math.min(1, speed / PHYS.walkSpeed);
    this.phase += dt * (3 + gait * 8);
    const s = Math.sin(this.phase);
    this.legL.rotation.x = s * 0.7 * gait;
    this.legR.rotation.x = -s * 0.7 * gait;
    this.armL.rotation.x = -s * 0.5 * gait;
    this.armR.rotation.x = s * 0.5 * gait;
    this.hips.position.y = 0.92 + Math.abs(s) * 0.045 * gait;
    this.hips.rotation.x = e.alive ? 0 : -Math.PI / 2;
    if (!e.alive) {
      // a bot that has been killed collapses; Clark leaves them where they fall
      this.hips.position.y = 0.25;
      this.group.rotation.y += dt * 0.4;
    }
    const dazed = e.dazedUntil > worldT;
    if (dazed) this.hips.rotation.z = Math.sin(this.phase * 9) * 0.12;
    else this.hips.rotation.z = 0;
    this.group.visible = true;
  }
}

/** Very simple viewmodel: a flashlight held in the right hand. */
export class PlayerHands {
  constructor(camera) {
    const g = new THREE.Group();
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.09, 0.5), new THREE.MeshLambertMaterial({ color: SKIN }));
    arm.position.set(0.34, -0.28, -0.45);
    g.add(arm);
    const torch = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 0.24, 8), new THREE.MeshLambertMaterial({ color: 0x2b2b30 }));
    torch.rotation.x = Math.PI / 2;
    torch.position.set(0.34, -0.24, -0.72);
    g.add(torch);
    const lens = new THREE.Mesh(new THREE.CircleGeometry(0.05, 10), new THREE.MeshBasicMaterial({ color: 0xfff6cc }));
    lens.position.set(0.34, -0.24, -0.85);
    g.add(lens);
    this.group = g;
    this.kick = 0;
    camera.add(g);
  }

  update(dt, firing) {
    this.kick = Math.max(0, this.kick - dt * 6);
    if (firing) this.kick = 1;
    this.group.position.z = this.kick * 0.06;
    this.group.rotation.x = this.kick * 0.15;
  }
}
