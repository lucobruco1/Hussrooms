/**
 * Feature/observation encoder.
 *
 * Every brain - the four survivor bots and Cap'n Clark - sees exactly the same
 * kind of input. There is no privileged channel: no one is handed "the player
 * is 4.2 m away, bearing 37 degrees". They get an egocentric 13 x 13 patch of
 * the floor around themselves (walls, bodies, smell, sound, how much of this
 * they have already seen) plus a dozen scalars, and they have to work the rest
 * out from pixels and feedback, the same way EfficientZero's representation
 * network has to work out Atari from raw frames.
 *
 * The player's own HUD-less view is rendered from the same world state; the
 * difference is only that a human has eyes.
 */

import { MAP, OBS, PHYS } from './config.js';

/** Exact scalar layout - keep SCALAR_DIM in sync. */
export const SCALARS = [
  'posX', 'posZ', 'sinYaw', 'cosYaw',
  'health', 'enemyDist', 'enemyBearingSin', 'enemyBearingCos',
  'clearanceAhead', 'speed', 'sinceDamage', 'damageBearingSin', 'damageBearingCos',
];
export const SCALAR_DIM = SCALARS.length;

/** Dimensions of the representation net's input. */
export function obsDims() {
  const pooled = (OBS.crop / OBS.pool) * (OBS.crop / OBS.pool) * OBS.channels;
  return { frame: OBS.crop * OBS.crop * OBS.channels, pooled: Math.round(pooled), scalars: SCALAR_DIM };
}

/**
 * Fixed (weight-free) pooling of the egocentric patch: max for walls / bodies /
 * novelty so thin features survive, mean for the smooth scent + noise fields.
 * This is what keeps a 60 fps browser tick affordable on 2 CPU cores.
 */
export function poolObs(frame, out) {
  const C = OBS.crop;
  const ch = OBS.channels;
  const k = OBS.pool;
  const P = C / k;
  const target = out || new Float32Array(Math.round(P * P * ch));
  if (C % k !== 0) throw new Error('crop must be divisible by pool');
  for (let px = 0; px < P; px++) {
    for (let pz = 0; pz < P; pz++) {
      for (let c = 0; c < ch; c++) {
        let acc = 0;
        let mx = -Infinity;
        for (let dx = 0; dx < k; dx++) {
          for (let dz = 0; dz < k; dz++) {
            const v = frame[((px * k + dx) * C + (pz * k + dz)) * ch + c];
            acc += v;
            if (v > mx) mx = v;
          }
        }
        const mean = acc / (k * k);
        target[(px * P + pz) * ch + c] = c === 0 || c === 1 || c === 5 ? mx : mean;
      }
    }
  }
  return target;
}

export class Features {
  constructor(grid) {
    this.grid = grid;
    const n = grid.gx * grid.gz;
    this.n = n;
    this.scent = new Float32Array(n); // everyone's smell, mixed
    this.noise = new Float32Array(n); // recent movement sound
    this.visit = new Map(); // entityId -> Float32Array (discovered cells)
    this.crop = OBS.crop;
    this.half = (OBS.crop - 1) / 2;
    this.chan = OBS.channels;
    this.cell = OBS.cell;
    this.frameSize = this.crop * this.crop * this.chan;
  }

  reset() {
    this.scent.fill(0);
    this.noise.fill(0);
    this.visit.clear();
  }

  visitMap(entityId) {
    let v = this.visit.get(entityId);
    if (!v) {
      v = new Float32Array(this.n);
      this.visit.set(entityId, v);
    }
    return v;
  }

  decay() {
    const sd = OBS.scentDecay;
    const nd = OBS.noiseDecay;
    const s = this.scent;
    const nn = this.noise;
    for (let i = 0; i < this.n; i++) {
      s[i] *= sd;
      nn[i] *= nd;
    }
  }

  /** Lay down smell + noise where an entity is standing. */
  stamp(entity, speed) {
    const [cx, cz] = this.grid.cellOf(entity.x, entity.z);
    if (!this.grid.inBounds(cx, cz)) return;
    const i = this.grid.idx(cx, cz);
    const weight = entity.kind === 'clark' ? 2.2 : 1.0;
    this.scent[i] = Math.min(4, this.scent[i] + weight * 0.09);
    if (speed > 0.2) this.noise[i] = Math.min(3, this.noise[i] + Math.min(1, speed / PHYS.sprintSpeed) * 0.3);
    // NOTE: discovery (the novelty channel / novelty reward) is recorded by
    // World.computeShaping, not here, so the reward and the observation agree.
  }

  /**
   * Modelled perception: within range, inside the vision cone, line of sight
   * clear. `propsBlock` lets furniture hide people, which matters a lot for
   * the survivor bots' hide-and-seek.
   */
  canSee(a, b) {
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const d2 = dx * dx + dz * dz;
    if (d2 > PHYS.sightRange * PHYS.sightRange) return false;
    const d = Math.sqrt(d2);
    const rangeFactor = Math.max(PHYS.sightCone, 0.35 - (d / PHYS.sightRange) * 0.1);
    const fx = Math.sin(a.yaw);
    const fz = Math.cos(a.yaw);
    const dot = (dx * fx + dz * fz) / (d || 1);
    if (dot < rangeFactor) return false;
    return this.grid.lineOfSight(a.x, a.z, b.x, b.z, true);
  }

  canHear(a, b) {
    const d = Math.hypot(b.x - a.x, b.z - a.z);
    return d < PHYS.hearingRange;
  }

  clearanceAhead(entity, maxD = 3.2, steps = 10) {
    const fx = Math.sin(entity.yaw);
    const fz = Math.cos(entity.yaw);
    for (let s = 1; s <= steps; s++) {
      const t = (s / steps) * maxD;
      if (this.grid.isWall(...this.grid.cellOf(entity.x + fx * t, entity.z + fz * t))) return t - maxD / steps;
    }
    return maxD;
  }

  /** Fill `outFrame` (Float32Array) + `outScalars` for one entity. */
  encode(world, ent, outFrame, outScalars) {
    const g = this.grid;
    const C = this.crop;
    const half = this.half;
    const ch = this.chan;
    const size = C * C * ch;
    if (!outFrame || outFrame.length !== size) outFrame = new Float32Array(size);
    outFrame.fill(0);
    const visit = this.visitMap(ent.id);
    const fx = Math.sin(ent.yaw);
    const fz = Math.cos(ent.yaw);
    // right-hand basis in the ground plane
    const rx = fz;
    const rz = -fx;
    // --- patch channels -----------------------------------------------------
    for (let ax = 0; ax < C; ax++) {
      for (let az = 0; az < C; az++) {
        const lx = (ax - half) * this.cell; // forward offset
        const lz = (az - half) * this.cell; // right offset
        // rotate local frame offset into world space
        const wx = ent.x + fx * lx + rx * lz;
        const wz = ent.z + fz * lx + rz * lz;
        const cxi = Math.floor(wx / g.cs);
        const czi = Math.floor(wz / g.cs);
        const o = (ax * C + az) * ch;
        if (!g.inBounds(cxi, czi)) {
          outFrame[o] = 1; // outside the level reads as wall
          continue;
        }
        const gi = g.idx(cxi, czi);
        if (g.wall[gi]) {
          outFrame[o] = 1;
          continue;
        }
        const pi = g.propAt[gi];
        if (pi >= 0 && g.props[pi].blocksMove) outFrame[o + 0] = 0.45;
        outFrame[o + 1] = 0; // bodies, filled below
        outFrame[o + 2] = Math.min(1, this.scent[gi] / 2);
        outFrame[o + 3] = 0; // rival smell, filled below
        outFrame[o + 4] = Math.min(1, this.noise[gi] / 1.5);
        outFrame[o + 5] = visit[gi] ? 0 : 1; // unexplored
      }
    }
    const cellIndexFromWorld = (wx, wz) => {
      const dx = wx - ent.x;
      const dz = wz - ent.z;
      const lx = dx * fx + dz * fz; // forward
      const lz = dx * rx + dz * rz; // right
      const ax = Math.round(lx / this.cell) + half;
      const az = Math.round(lz / this.cell) + half;
      if (ax < 0 || az < 0 || ax >= C || az >= C) return -1;
      return (ax * C + az) * ch;
    };
    // --- bodies and rival smell --------------------------------------------
    let bestEnemy = null;
    let bestD = Infinity;
    for (const other of world.entities) {
      if (other === ent || !other.alive) continue;
      const hostile = world.hostile(ent, other);
      // scent trail: always smellable (it lingers), stronger for the other side
      const [ocx, ocz] = g.cellOf(other.x, other.z);
      if (g.inBounds(ocx, ocz)) {
        const oi = g.idx(ocx, ocz);
        const oo = cellIndexFromWorld(other.x, other.z);
        if (oo >= 0) {
          outFrame[oo + 3] = Math.min(1, this.scent[oi] / 2);
          if (hostile) outFrame[oo + 3] = Math.max(outFrame[oo + 3], 0.55);
          else outFrame[oo + 3] = Math.max(outFrame[oo + 3], 0.25);
        }
      }
      if (this.canSee(ent, other)) {
        const o = cellIndexFromWorld(other.x, other.z);
        if (o >= 0) {
          outFrame[o + 1] = hostile ? 1.0 : 0.6;
          // sharpen with a 2-cell halo so the encoder can localise it
          for (let dx = -1; dx <= 1; dx++)
            for (let dz = -1; dz <= 1; dz++) {
              if (dx === 0 && dz === 0) continue;
              const oo = cellIndexFromWorld(other.x + dx, other.z + dz);
              if (oo >= 0) outFrame[oo + 1] = Math.max(outFrame[oo + 1], hostile ? 0.35 : 0.2);
            }
        }
        if (hostile) {
          const d = Math.hypot(other.x - ent.x, other.z - ent.z);
          if (d < bestD) {
            bestD = d;
            bestEnemy = other;
          }
        }
      } else if (hostile && this.canHear(ent, other)) {
        const o = cellIndexFromWorld(other.x, other.z);
        if (o >= 0) outFrame[o + 4] = Math.min(1, outFrame[o + 4] + 0.6);
      }
    }
    // --- scalars ------------------------------------------------------------
    const s = outScalars && outScalars.length === SCALAR_DIM ? outScalars : new Float32Array(SCALAR_DIM);
    const mapW = g.gx * g.cs;
    s[0] = (ent.x / mapW) * 2 - 1;
    s[1] = (ent.z / mapW) * 2 - 1;
    s[2] = Math.sin(ent.yaw);
    s[3] = Math.cos(ent.yaw);
    s[4] = ent.health / ent.maxHealth;
    if (bestEnemy) {
      const dx = bestEnemy.x - ent.x;
      const dz = bestEnemy.z - ent.z;
      const d = Math.hypot(dx, dz);
      const ux = dx / (d || 1);
      const uz = dz / (d || 1);
      s[5] = Math.min(1, d / PHYS.sightRange);
      s[6] = (ux * rx + uz * rz); // signed right (+ = to the right)
      s[7] = (ux * fx + uz * fz); // forward
    } else {
      s[5] = 1;
      s[6] = 0;
      s[7] = 0;
    }
    s[8] = this.clearanceAhead(ent) / 3.2;
    s[9] = Math.min(1, Math.hypot(ent.vx, ent.vz) / PHYS.sprintSpeed);
    s[10] = Math.min(1, (world.t - ent.lastDamageAt) / 2);
    if (ent.lastDamageFrom != null) {
      const src = world.byId(ent.lastDamageFrom);
      if (src) {
        const dx = src.x - ent.x;
        const dz = src.z - ent.z;
        const d = Math.hypot(dx, dz) || 1;
        s[11] = (dx / d) * rx + (dz / d) * rz;
        s[12] = (dx / d) * fx + (dz / d) * fz;
      } else {
        s[11] = 0;
        s[12] = 0;
      }
    } else {
      s[11] = 0;
      s[12] = 0;
    }
    return { frame: outFrame, scalars: s };
  }

  /**
   * Omniscient teaching view - used ONLY by tools/pretrain.mjs to give the
   * from-scratch networks a warm start. The browser never calls this.
   */
  encodeOmniscient(world, ent, radiusCells = 3) {
    const g = this.grid;
    const C = this.crop;
    const half = this.half;
    const ch = this.chan;
    const frame = new Float32Array(C * C * ch);
    const visit = this.visitMap(ent.id);
    const [ecx, ecz] = g.cellOf(ent.x, ent.z);
    for (let ax = 0; ax < C; ax++) {
      for (let az = 0; az < C; az++) {
        const cx = ecx + (ax - half) * radiusCells;
        const cz = ecz + (az - half) * radiusCells;
        const o = (ax * C + az) * ch;
        if (!g.inBounds(cx, cz)) {
          frame[o] = 1;
          continue;
        }
        const gi = g.idx(cx, cz);
        if (g.wall[gi]) frame[o] = 1;
        else {
          frame[o + 2] = Math.min(1, this.scent[gi] / 2);
          frame[o + 4] = Math.min(1, this.noise[gi] / 1.5);
          frame[o + 5] = visit[gi] ? 0 : 1;
        }
      }
    }
    for (const other of world.entities) {
      if (other === ent || !other.alive) continue;
      const [ocx, ocz] = g.cellOf(other.x, other.z);
      const ax = Math.round((ocx - ecx) / radiusCells) + half;
      const az = Math.round((ocz - ecz) / radiusCells) + half;
      if (ax < 0 || az < 0 || ax >= C || az >= C) continue;
      const o = (ax * C + az) * ch;
      frame[o + 1] = world.hostile(ent, other) ? 1 : 0.5;
      frame[o + 3] = 1;
    }
    const s = new Float32Array(SCALAR_DIM);
    const mapW = g.gx * g.cs;
    s[0] = (ent.x / mapW) * 2 - 1;
    s[1] = (ent.z / mapW) * 2 - 1;
    s[2] = Math.sin(ent.yaw);
    s[3] = Math.cos(ent.yaw);
    s[4] = ent.health / ent.maxHealth;
    let best = null;
    let bestD = Infinity;
    for (const other of world.entities) {
      if (other === ent || !other.alive || !world.hostile(ent, other)) continue;
      const d = Math.hypot(other.x - ent.x, other.z - ent.z);
      if (d < bestD) {
        bestD = d;
        best = other;
      }
    }
    if (best) {
      const dx = best.x - ent.x;
      const dz = best.z - ent.z;
      const d = Math.hypot(dx, dz) || 1;
      const fx = Math.sin(ent.yaw);
      const fz = Math.cos(ent.yaw);
      const rx = fz;
      const rz = -fx;
      s[5] = Math.min(1, d / PHYS.sightRange);
      s[6] = (dx / d) * rx + (dz / d) * rz;
      s[7] = (dx / d) * fx + (dz / d) * fz;
    } else {
      s[5] = 1;
    }
    s[8] = this.clearanceAhead(ent) / 3.2;
    s[9] = Math.min(1, Math.hypot(ent.vx, ent.vz) / PHYS.sprintSpeed);
    s[10] = 1;
    s[11] = 0;
    s[12] = 0;
    return { frame, scalars: s };
  }
}

export const OBS_META = { crops: OBS.crop, channels: OBS.channels, scalars: SCALAR_DIM, map: MAP };
