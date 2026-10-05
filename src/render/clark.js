/**
 * Cap'n Clark - the Still Life.
 *
 * Modelled from the Backrooms film design: a towering, malformed copy of Clark
 * in the pirate costume from his furniture-store adverts ("Cap'n Clark's
 * Ottoman Empire"). Stretched-upward face, bulging misaligned eyes, a tricorn
 * hat with feathers, a stained teal shirt under a yellow vest with a purple
 * sash, one arm jutting out at the angle of the crutch he held in the advert,
 * and a leg fused into a splintered table-leg peg. In the film he is ~3.7 m.
 *
 * Everything is primitives and it is animated procedurally - no rigs, no
 * assets, and the animation is driven straight from the entity's simulation
 * state (speed, lunge, roar, bite), so what you see is what the agent did.
 */

import * as THREE from '../three-shim.js';
import { PHYS } from '../sim/config.js';

const TEAL = 0x2f8f8a;
const TEAL_DARK = 0x1d5f5c;
const VEST = 0xd8c15a;
const SASH = 0x5b2f7a;
const PANTS = 0x3b3128;
const WOOD = 0x6b4a2a;
const SKIN = 0x6b4a3a;
const HAT = 0x3a2a1e;

function box(w, h, d, color, opts = {}) {
  const mat = new THREE.MeshLambertMaterial({ color, ...opts });
  return new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
}

function cyl(rt, rb, h, color, seg = 8, opts = {}) {
  const mat = new THREE.MeshLambertMaterial({ color, ...opts });
  return new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
}

export class ClarkMesh {
  constructor() {
    this.group = new THREE.Group();
    this.group.scale.setScalar(1);
    // ---------------------------------------------------------------- legs
    // Left leg: long, thin, slightly wrong. Right leg: the peg.
    this.hip = new THREE.Group();
    this.hip.position.y = 1.85;
    this.group.add(this.hip);

    this.legL = new THREE.Group();
    this.legL.position.set(-0.22, 0, 0);
    const thighL = cyl(0.16, 0.13, 0.95, PANTS, 7);
    thighL.position.y = -0.48;
    this.legL.add(thighL);
    const shinL = cyl(0.12, 0.1, 0.95, PANTS, 7);
    shinL.position.y = -1.4;
    this.legL.add(shinL);
    const shoeL = box(0.3, 0.14, 0.5, 0x241c16);
    shoeL.position.set(0, -1.9, 0.1);
    this.legL.add(shoeL);
    this.hip.add(this.legL);

    this.legR = new THREE.Group();
    this.legR.position.set(0.22, 0, 0);
    const thighR = cyl(0.15, 0.12, 0.7, PANTS, 7);
    thighR.position.y = -0.35;
    this.legR.add(thighR);
    const peg = cyl(0.09, 0.16, 1.15, WOOD, 6);
    peg.rotation.z = 0.06;
    peg.position.y = -1.28;
    this.legR.add(peg);
    const pegFoot = cyl(0.17, 0.17, 0.08, WOOD, 6);
    pegFoot.position.y = -1.85;
    this.legR.add(pegFoot);
    this.hip.add(this.legR);

    // --------------------------------------------------------------- torso
    this.torso = new THREE.Group();
    this.torso.position.y = 1.85;
    this.group.add(this.torso);

    const belly = box(0.95, 0.8, 0.6, TEAL);
    belly.position.y = 0.42;
    this.torso.add(belly);
    const chest = box(1.05, 0.75, 0.62, TEAL);
    chest.position.y = 1.12;
    this.torso.add(chest);
    // vest panels
    const vestL = box(0.28, 1.3, 0.66, VEST);
    vestL.position.set(-0.3, 0.72, 0.01);
    this.torso.add(vestL);
    const vestR = box(0.28, 1.3, 0.66, VEST);
    vestR.position.set(0.3, 0.72, 0.01);
    this.torso.add(vestR);
    // sash
    const sash = box(1.12, 0.22, 0.68, SASH);
    sash.position.set(0, 0.62, 0.01);
    sash.rotation.z = -0.12;
    this.torso.add(sash);
    // huge collar
    const collar = box(0.95, 0.3, 0.5, TEAL_DARK);
    collar.position.y = 1.55;
    this.torso.add(collar);
    const collarFlareL = box(0.34, 0.42, 0.16, TEAL_DARK);
    collarFlareL.position.set(-0.42, 1.62, 0.24);
    collarFlareL.rotation.z = 0.25;
    this.torso.add(collarFlareL);
    const collarFlareR = box(0.34, 0.42, 0.16, TEAL_DARK);
    collarFlareR.position.set(0.42, 1.62, 0.24);
    collarFlareR.rotation.z = -0.25;
    this.torso.add(collarFlareR);

    // --------------------------------------------------------------- arms
    // Left arm hangs long; right arm juts out stiffly at the crutch angle.
    this.armL = new THREE.Group();
    this.armL.position.set(-0.56, 1.45, 0);
    this.armL.rotation.z = 0.18;
    const upperL = cyl(0.13, 0.11, 1.0, TEAL, 7);
    upperL.position.y = -0.5;
    this.armL.add(upperL);
    this.foreL = new THREE.Group();
    this.foreL.position.y = -1.0;
    const lower = cyl(0.11, 0.1, 0.95, SKIN, 7);
    lower.position.y = -0.47;
    this.foreL.add(lower);
    const handL = box(0.2, 0.34, 0.16, SKIN);
    handL.position.y = -1.02;
    this.foreL.add(handL);
    this.armL.add(this.foreL);
    this.torso.add(this.armL);

    this.armR = new THREE.Group();
    this.armR.position.set(0.56, 1.45, 0);
    this.armR.rotation.z = -0.95; // jutting out sideways, like the advert's crutch
    this.armR.rotation.x = -0.25;
    const upperR = cyl(0.13, 0.11, 1.0, TEAL, 7);
    upperR.position.y = -0.5;
    this.armR.add(upperR);
    this.foreR = new THREE.Group();
    this.foreR.position.y = -1.0;
    const lowerR = cyl(0.11, 0.1, 0.95, SKIN, 7);
    lowerR.position.y = -0.47;
    this.foreR.add(lowerR);
    const handR = box(0.22, 0.36, 0.18, SKIN);
    handR.position.y = -1.02;
    this.foreR.add(handR);
    this.armR.add(this.foreR);
    this.torso.add(this.armR);

    // ---------------------------------------------------------------- head
    this.head = new THREE.Group();
    this.head.position.y = 1.78;
    this.torso.add(this.head);

    const skull = box(0.52, 0.62, 0.5, SKIN);
    skull.position.y = 0.3;
    skull.scale.set(1, 1.25, 1); // stretched upward
    this.head.add(skull);
    const jaw = box(0.44, 0.3, 0.42, SKIN);
    jaw.position.set(0, 0.0, 0.04);
    this.jaw = new THREE.Group();
    this.jaw.add(jaw);
    this.head.add(this.jaw);
    // beard
    const beard = box(0.46, 0.26, 0.2, 0x241a14);
    beard.position.set(0, 0.06, 0.26);
    this.head.add(beard);
    // bulging, misaligned eyes
    const eyeGeo = new THREE.SphereGeometry(0.1, 10, 8);
    const eyeMat = new THREE.MeshLambertMaterial({ color: 0xf3f0e2, emissive: 0x181410 });
    const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
    eyeL.position.set(-0.15, 0.42, 0.25);
    eyeL.scale.set(1.15, 1.25, 0.8);
    this.head.add(eyeL);
    const eyeR = new THREE.Mesh(eyeGeo, eyeMat);
    eyeR.position.set(0.17, 0.5, 0.24);
    eyeR.scale.set(1.0, 1.4, 0.75);
    this.head.add(eyeR);
    const pupilGeo = new THREE.SphereGeometry(0.045, 8, 6);
    const pupilMat = new THREE.MeshBasicMaterial({ color: 0x120c0a });
    const pL = new THREE.Mesh(pupilGeo, pupilMat);
    pL.position.set(-0.14, 0.42, 0.33);
    this.head.add(pL);
    const pR = new THREE.Mesh(pupilGeo, pupilMat);
    pR.position.set(0.18, 0.5, 0.32);
    this.head.add(pR);
    // brow ridge
    const brow = box(0.56, 0.09, 0.16, 0x3a2a22);
    brow.position.set(0, 0.6, 0.22);
    brow.rotation.x = 0.2;
    this.head.add(brow);

    // tricorn hat + feathers
    const hatCrown = cyl(0.3, 0.34, 0.3, HAT, 8);
    hatCrown.position.y = 0.78;
    this.head.add(hatCrown);
    const brim = cyl(0.52, 0.52, 0.06, HAT, 10);
    brim.position.y = 0.64;
    this.head.add(brim);
    const brimUp1 = box(0.5, 0.3, 0.08, HAT);
    brimUp1.position.set(0, 0.76, -0.4);
    brimUp1.rotation.x = -0.5;
    this.head.add(brimUp1);
    const brimUp2 = box(0.5, 0.3, 0.08, HAT);
    brimUp2.position.set(-0.4, 0.76, 0.05);
    brimUp2.rotation.z = 0.5;
    this.head.add(brimUp2);
    const brimUp3 = box(0.5, 0.3, 0.08, HAT);
    brimUp3.position.set(0.4, 0.76, 0.05);
    brimUp3.rotation.z = -0.5;
    this.head.add(brimUp3);
    const feather1 = cyl(0.02, 0.05, 0.5, 0xb03a3a, 5);
    feather1.position.set(0.16, 0.95, -0.2);
    feather1.rotation.z = -0.5;
    this.head.add(feather1);
    const feather2 = cyl(0.02, 0.05, 0.44, 0xd8d0a0, 5);
    feather2.position.set(-0.16, 0.95, -0.22);
    feather2.rotation.z = 0.5;
    this.head.add(feather2);

    // stains on the shoulder (as described in the film's costume)
    const stain = box(0.3, 0.18, 0.02, 0x101010, { transparent: true, opacity: 0.55 });
    stain.position.set(0.34, 1.42, 0.33);
    this.torso.add(stain);

    // a faint red glow at chest height, purely so he reads in dim corridors
    const glow = new THREE.PointLight(0xff5533, 0.35, 6, 2);
    glow.position.set(0, 1.4, 0.4);
    this.torso.add(glow);

    this.group.traverse((o) => {
      if (o.isMesh) o.frustumCulled = true;
    });
    this.phase = 0;
    this.biteT = 0; // local animation timers - the renderer never writes sim state
    this.roarT = 0;
    this.lastBiteFlash = 0;
    this.lastRoar = 0;
  }

  /**
   * @param dt      seconds since the previous frame
   * @param e       entity from the simulator (velocity, lungeUntil, roar and bite
   *                flash timestamps are all real simulation state)
   * @param worldT  the simulation clock, passed in rather than written onto `e`
   */
  update(dt, e, worldT = 0) {
    const speed = Math.hypot(e.vx, e.vz);
    const gait = Math.min(1, speed / PHYS.clarkSpeed);
    this.phase += dt * (2.2 + gait * 5.5);
    const s = Math.sin(this.phase);
    const c = Math.cos(this.phase);

    // limp: the peg leg drags, the torso rolls
    this.legL.rotation.x = s * 0.55 * (0.25 + gait);
    this.legR.rotation.x = -s * 0.35 * (0.25 + gait);
    this.hip.position.y = 1.85 + Math.abs(s) * 0.06 * gait;
    this.torso.rotation.z = s * 0.07 * gait;
    this.torso.rotation.x = gait * 0.12;
    this.armL.rotation.x = -s * 0.5 * (0.2 + gait);
    this.foreL.rotation.x = Math.max(0, -s) * 0.5 * gait;
    // the jutting arm stays jutting - it is part of who he is
    this.armR.rotation.z = -0.95 + Math.sin(this.phase * 0.5) * 0.06;
    this.armR.rotation.x = -0.25 + s * 0.12;

    // head: constantly scanning, with a scared tilt
    this.head.rotation.y = Math.sin(this.phase * 0.33) * 0.35;
    this.head.rotation.z = 0.08 + Math.sin(this.phase * 0.21) * 0.05;
    this.head.rotation.x = 0.06 + Math.sin(this.phase * 0.17) * 0.04;

    // jaw: bite snaps and roars
    this.biteT = Math.max(0, this.biteT - dt);
    this.roarT = Math.max(0, this.roarT - dt);
    let jaw = 0.1 + Math.sin(this.phase * 0.7) * 0.05;
    if (e.biteFlashUntil > this.lastBiteFlash) {
      this.lastBiteFlash = e.biteFlashUntil;
      this.biteT = 0.35;
    }
    if (e.lastRoarAt != null && e.lastRoarAt > this.lastRoar) {
      this.lastRoar = e.lastRoarAt;
      this.roarT = 0.9;
    }
    if (this.biteT > 0) jaw = 0.55 * (this.biteT / 0.35);
    else if (this.roarT > 0) jaw = 0.8 * (this.roarT / 0.9);
    this.jaw.rotation.x = jaw;

    // lunging: whole body pitched forward, arms up
    const lunging = e.lungeUntil > worldT;
    this.torso.rotation.x = lunging ? 0.55 : gait * 0.12;
    if (lunging) {
      this.armL.rotation.x = -1.5;
      this.armR.rotation.x = -1.0;
      this.jaw.rotation.x = 0.7;
      this.hip.position.y = 1.75;
    }
  }
}
