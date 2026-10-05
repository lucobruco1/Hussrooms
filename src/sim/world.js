/**
 * Hussrooms world simulation.
 *
 * Deliberately headless: no three.js, no DOM. The renderer is a viewer bolted
 * on top, and tools/ can run the identical simulation in Node for fast
 * background training.
 *
 * TWO FACTIONS
 *   survivors : the player + 4 respawning bots
 *   clark     : Cap'n Clark, one StiIl Life, respawns far away when destroyed
 *
 * The world never decides what anybody *should* do. It resolves the action an
 * entity emitted, applies physics, and returns a number. Behaviour comes
 * entirely from the EfficientZero brains in src/ai.
 */

import { A, ACTIONS, MAP, PHYS, REWARDS } from './config.js';
import { Grid, collideProps, collideWalls, mulberry32 } from './grid.js';
import { dist } from './util.js';
import { Features } from './features.js';

export const FACTION = { SURVIVORS: 0, CLARK: 1 };

let nextId = 1;

export class Entity {
  constructor(kind, x, z, yaw, faction) {
    this.id = nextId++;
    this.kind = kind; // 'player' | 'bot' | 'clark'
    this.faction = faction;
    this.x = x;
    this.z = z;
    this.vx = 0;
    this.vz = 0;
    this.yaw = yaw;
    this.maxHealth = kind === 'clark' ? PHYS.clarkHealth : PHYS.playerHealth;
    this.health = this.maxHealth;
    this.alive = true;
    this.respawnTimer = 0;
    this.radius = kind === 'clark' ? PHYS.clarkRadius : PHYS.agentRadius;
    this.action = 0;
    this.actionStart = 0;
    this.pendingReward = 0;
    this.lifetimeReward = 0;
    this.lastDamageAt = -999;
    this.lastDamageFrom = null;
    this.stunUntil = 0;
    this.lungeUntil = 0;
    this.lungeCooldown = 0;
    this.biteCooldown = 0;
    this.roarCooldown = 0;
    this.dazedUntil = 0;
    this.slowUntil = 0;
    this.stamina = 1;
    this.spawnedAt = 0;
    this.stats = { kills: 0, deaths: 0, bites: 0, roars: 0, lunges: 0, distance: 0, novelty: 0 };
    this.deathAt = -999;
    this.emergeUntil = 0;
  }

  get speedNow() {
    return Math.hypot(this.vx, this.vz);
  }
}

export class World {
  constructor(opts = {}) {
    this.grid = new Grid(opts.seed ?? MAP.seed);
    this.features = new Features(this.grid);
    this.rng = mulberry32((opts.seed ?? MAP.seed) ^ 0x9e3779b9);
    this.entities = [];
    this.byIdMap = new Map();
    this.t = 0;
    this.tick = 0;
    this.rewards = { ...REWARDS, ...(opts.rewards || {}) };
    this.events = []; // drained each frame by the UI/telemetry
    this.telemetry = {
      clarkKills: 0,
      botDeaths: 0,
      playerDeaths: 0,
      bites: 0,
      roars: 0,
      lunges: 0,
      respawns: 0,
      steps: 0,
      clarkSees: 0, // ticks where Clark had anybody in his modelled perception
      clarkClosest: Infinity, // closest he ever got to a survivor
      botSees: 0,
    };
    this.detectedPrev = new Map(); // entityId -> Map(otherId -> dist) for closing rewards
    this.spawnIndex = 0;
    this._obsCache = new Map();
    this.spawnPlayer();
    for (let i = 0; i < PHYS.botCount; i++) this.spawnBot(i);
    this.spawnClark(true);
    this.entityCounts = { player: 1, bot: PHYS.botCount, clark: 1 };
  }

  // ---------------------------------------------------------------- spawning
  /**
   * Pick a spawn cell. Survivors spawn away from Clark; Clark spawns as far as
   * possible from the survivors - which is a *world* rule (he crawls out of a
   * hole somewhere far away), not a behaviour rule. What he does next is his
   * own business.
   */
  /**
   * Choose a spawn cell that is *maximally far* from the given entities (in
   * metres), with an optional hard minimum. Survivors spawn clear of Clark,
   * Clark crawls out as far from the survivors as the floor allows, and nothing
   * ever spawns on top of anything else.
   */
  pickSpawn(farFromIds, minCellsAway = 0) {
    const pts = this.grid.spawnPoints;
    const others = farFromIds.map((id) => this.byIdMap.get(id)).filter(Boolean);
    let best = null;
    let bestScore = -Infinity;
    const stride = Math.max(1, Math.floor(pts.length / 220));
    for (let i = 0; i < pts.length; i += stride) {
      const p = pts[i];
      let minD = Infinity;
      for (const o of others) minD = Math.min(minD, Math.hypot(o.x - p.x, o.z - p.z));
      if (!others.length) minD = Infinity;
      const ok = minD >= minCellsAway * this.grid.cs;
      const score = (ok ? 1000 : 0) + (Number.isFinite(minD) ? minD : 0) + this.rng() * 3;
      if (score > bestScore) {
        bestScore = score;
        best = p;
      }
    }
    return best || pts[0];
  }

  spawnPlayer() {
    const p = this.pickSpawn([], 0);
    const e = new Entity('player', p.x, p.z, this.rng() * Math.PI * 2, FACTION.SURVIVORS);
    this.entities.push(e);
    this.byIdMap.set(e.id, e);
    this.player = e;
    return e;
  }

  spawnBot(i) {
    const p = this.pickSpawn(
      this.entities.filter((e) => e.faction === FACTION.SURVIVORS).map((e) => e.id),
      2,
    );
    const e = new Entity('bot', p.x, p.z, this.rng() * Math.PI * 2, FACTION.SURVIVORS);
    e.index = i;
    this.entities.push(e);
    this.byIdMap.set(e.id, e);
    return e;
  }

  spawnClark(initial = false) {
    if (this.clark && this.clark.alive) return this.clark;
    const survivors = this.entities.filter((e) => e.faction === FACTION.SURVIVORS && e.alive);
    const p = this.pickSpawn(survivors.map((e) => e.id), 6);
    if (!this.clark) {
      const e = new Entity('clark', p.x, p.z, 0, FACTION.CLARK);
      this.entities.push(e);
      this.byIdMap.set(e.id, e);
      this.clark = e;
      e.emergeUntil = this.t + 2.2;
      return e;
    }
    const e = this.clark;
    e.x = p.x;
    e.z = p.z;
    e.yaw = Math.atan2(this.player.x - p.x, this.player.z - p.z);
    e.vx = e.vz = 0;
    e.alive = true;
    e.health = e.maxHealth;
    e.respawnTimer = 0;
    e.spawnedAt = this.t;
    e.emergeUntil = this.t + 2.2;
    e.lungeUntil = 0;
    e.biteCooldown = 0;
    e.roarCooldown = 0;
    this.telemetry.respawns++;
    this.emit({ type: 'respawn', kind: 'clark', x: e.x, z: e.z });
    if (!initial) this.features.visitMap(e.id).fill(0);
    return e;
  }

  byId(id) {
    return this.byIdMap.get(id) || null;
  }

  hostile(a, b) {
    return a.faction !== b.faction;
  }

  emit(ev) {
    ev.t = this.t;
    this.events.push(ev);
    if (this.events.length > 300) this.events.splice(0, this.events.length - 300);
  }

  drainEvents() {
    const e = this.events;
    this.events = [];
    return e;
  }

  // ------------------------------------------------------------- perception
  /** Cached observation for the current tick. */
  observe(ent) {
    const key = ent.id + ':' + this.tick;
    const hit = this._obsCache.get(key);
    if (hit) return hit;
    const res = this.features.encode(this, ent);
    this._obsCache.set(key, res);
    if (this._obsCache.size > 64) this._obsCache.clear();
    return res;
  }

  /** Who is inside `ent`'s modelled perception right now? (used for reward feedback only) */
  detected(ent) {
    const out = [];
    for (const o of this.entities) {
      if (o === ent || !o.alive || !this.hostile(ent, o)) continue;
      if (this.features.canSee(ent, o) || this.features.canHear(ent, o)) out.push(o);
    }
    return out;
  }

  // ------------------------------------------------------------------ action
  setAction(ent, actionIndex) {
    if (actionIndex !== ent.action) ent.actionStart = this.t;
    ent.action = actionIndex;
  }

  moveVector(actionIndex) {
    const a = ACTIONS[actionIndex] || ACTIONS[0];
    return a;
  }

  // -------------------------------------------------------------- simulation
  step(dt = PHYS.dt) {
    this.t += dt;
    this.tick++;
    this.telemetry.steps++;
    this.features.decay();
    this._obsCache.clear();
    const alive = [];
    for (const e of this.entities) if (e.alive) alive.push(e);

    for (const e of alive) {
      this.applyMovement(e, dt);
      this.features.stamp(e, e.speedNow);
    }

    // --- combat ------------------------------------------------------------
    const clark = this.clark;
    if (clark && clark.alive) {
      clark.lungeCooldown = Math.max(0, clark.lungeCooldown - dt);
      clark.biteCooldown = Math.max(0, clark.biteCooldown - dt);
      clark.roarCooldown = Math.max(0, clark.roarCooldown - dt);
      if (clark.biteCooldown <= 0) this.tryBite(clark);
      if (clark.action === A.lunge && clark.lungeCooldown <= 0) {
        // pounce
        clark.lungeUntil = this.t + PHYS.lungeDuration;
        clark.lungeDirX = Math.sin(clark.yaw);
        clark.lungeDirZ = Math.cos(clark.yaw);
        clark.lungeCooldown = PHYS.lungeCooldown;
        clark.stats.lunges++;
        this.telemetry.lunges++;
        this.emit({ type: 'lunge', x: clark.x, z: clark.z });
      }
      if (clark.action === A.roar && clark.roarCooldown <= 0) {
        clark.roarCooldown = PHYS.roarCooldown;
        clark.lastRoarAt = this.t;
        clark.stats.roars++;
        this.telemetry.roars++;
        this.emit({ type: 'roar', x: clark.x, z: clark.z, r: PHYS.roarRadius });
        let dazedAny = false;
        for (const o of this.entities) {
          if (!o.alive || !this.hostile(clark, o)) continue;
          if (dist(clark, o) <= PHYS.roarRadius) {
            o.dazedUntil = this.t + PHYS.roarDazeTime;
            o.pendingReward += this.rewards.dazed;
            dazedAny = true;
            o.vx *= 0.2;
            o.vz *= 0.2;
          }
        }
        if (dazedAny) clark.pendingReward += this.rewards.roarUseful;
      }
    }

    // --- per-tick shaping ---------------------------------------------------
    this.computeShaping(dt, alive);

    // --- deaths / respawns --------------------------------------------------
    for (const e of this.entities) {
      if (e.alive) continue;
      if (this.t >= e.respawnTimer) {
        if (e.kind === 'clark') this.spawnClark();
        else this.respawnSurvivor(e);
      }
    }

    // --- bookkeeping --------------------------------------------------------
    for (const e of this.entities) {
      if (!e.alive) continue;
      e.lifetimeReward += e.pendingReward;
      e.sprinting = false;
    }
  }

  /**
   * Body-relative movement.
   *
   * The 8 movement actions are interpreted in the agent's OWN frame: `fwd`
   * always walks where the agent is facing, `fwdR` turns 45 degrees right while
   * advancing, and so on, with yaw chasing the movement direction at turnRate.
   * This has to match the egocentric observation (src/sim/features.js) or the
   * policy learns a non-stationary mapping - which is exactly the failure mode
   * that had agents walking into a wall forever in early testing.
   */
  applyMovement(e, dt) {
    const move = ACTIONS[e.action] || ACTIONS[0];
    const dazed = this.t < e.dazedUntil ? 0.45 : 1;
    const stunned = this.t < e.stunUntil ? 0.3 : 1;
    const mauled = this.t < e.slowUntil ? PHYS.biteSlowFactor : 1;
    // --- explicit turning (never implied by movement)
    if (move.turn && e.kind !== 'player') {
      const rate = e.kind === 'clark' ? PHYS.clarkTurnSpeed : PHYS.turnSpeed;
      e.yaw += move.turn * rate * dazed * stunned * dt;
      e.stats.turning += Math.abs(move.turn) * rate * dt;
    }
    const fx = Math.sin(e.yaw);
    const fz = Math.cos(e.yaw);
    const rx = fz; // right-hand basis in the ground plane
    const rz = -fx;
    // local -> world
    let wx = move.dx * rx + move.dz * fx;
    let wz = move.dx * rz + move.dz * fz;
    let speed = e.kind === 'clark' ? PHYS.clarkSpeed : PHYS.walkSpeed;
    let sprintingNow = false;
    if (e.kind === 'player' && e.sprinting && e.stamina > 0) {
      speed = PHYS.sprintSpeed;
      sprintingNow = true;
    }
    speed *= dazed * stunned * mauled;
    // stamina bookkeeping: running drains it, everything else recovers it
    if (sprintingNow) e.stamina = Math.max(0, e.stamina - PHYS.staminaDrain * dt);
    else if (e.kind !== 'clark') e.stamina = Math.min(1, e.stamina + PHYS.staminaRecover * dt);
    if (e.kind === 'clark' && this.t < e.lungeUntil) {
      // Mid-pounce: fast, straight, and he can barely steer.
      speed = PHYS.clarkLungeSpeed * dazed * stunned;
      wx = e.lungeDirX ?? fx;
      wz = e.lungeDirZ ?? fz;
    }
    const len = Math.hypot(wx, wz);
    const tvx = len > 0 ? (wx / len) * speed : 0;
    const tvz = len > 0 ? (wz / len) * speed : 0;
    const accel = e.kind === 'clark' ? 9 : 13;
    e.vx += (tvx - e.vx) * Math.min(1, accel * dt);
    e.vz += (tvz - e.vz) * Math.min(1, accel * dt);

    const fromX = e.x;
    const fromZ = e.z;
    let nx = e.x + e.vx * dt;
    let nz = e.z + e.vz * dt;
    const res = collideWalls(this.grid, fromX, fromZ, nx, nz, e.radius);
    const pres = collideProps(this.grid, res.x, res.z, e.radius);
    e.x = pres.x;
    e.z = pres.z;
    // Penalise being *blocked*, not sliding along a surface: compare intended
    // against achieved displacement. Grinding into geometry has to cost
    // something, or policies happily learn to press into a wall forever.
    const intended = Math.hypot(e.vx, e.vz) * dt;
    const achieved = Math.hypot(e.x - fromX, e.z - fromZ);
    if (intended > 0.004 && achieved < intended * 0.4) {
      e.pendingReward += this.rewards.wallBump;
      e.hitWallAt = this.t;
    } else if (achieved < 0.25 * speed * dt) {
      // standing (or spinning) still while the body could be moving
      e.pendingReward += this.rewards.still;
    }
    const lim = this.grid.gx * this.grid.cs;
    e.x = Math.max(0.6, Math.min(lim - 0.6, e.x));
    e.z = Math.max(0.6, Math.min(lim - 0.6, e.z));
    const moved = Math.hypot(e.x - fromX, e.z - fromZ);
    e.stats.distance += moved;
    e.lastMoved = moved;
  }

  tryBite(clark) {
    const reach = PHYS.biteRange + clark.radius;
    let hitSomeone = false;
    for (const o of this.entities) {
      if (!o.alive || !this.hostile(clark, o)) continue;
      const d = dist(clark, o);
      if (d > reach) continue;
      const dx = o.x - clark.x;
      const dz = o.z - clark.z;
      const fx = Math.sin(clark.yaw);
      const fz = Math.cos(clark.yaw);
      if ((dx * fx + dz * fz) / (d || 1) < 0.35) continue; // must be roughly in front
      const lunging = this.t < clark.lungeUntil;
      const dmg = PHYS.biteDamage * (lunging ? 1.35 : 1);
      this.damage(clark, o, dmg, 'bite');
      // the wound staggers them: this is what makes a landed bite lead to a kill
      o.slowUntil = this.t + PHYS.biteSlowTime;
      clark.biteCooldown = PHYS.biteCooldown * (lunging ? 0.7 : 1);
      clark.biteFlashUntil = this.t + 0.35;
      clark.stats.bites++;
      this.telemetry.bites++;
      clark.pendingReward += this.rewards.bite;
      if (lunging) clark.pendingReward += this.rewards.lungeHit;
      // shove the victim
      const nx = dx / (d || 1);
      const nz = dz / (d || 1);
      o.vx += nx * 6.5;
      o.vz += nz * 6.5;
      o.stunUntil = this.t + 0.28;
      this.emit({ type: 'bite', x: clark.x, z: clark.z, target: o.id, lunging });
      hitSomeone = true;
      break;
    }
    return hitSomeone;
  }

  damage(attacker, target, amount, cause) {
    if (!target.alive) return;
    target.health -= amount;
    target.lastDamageAt = this.t;
    target.lastDamageFrom = attacker ? attacker.id : null;
    this.emit({ type: 'damage', target: target.id, amount, cause, x: target.x, z: target.z });
    if (target.health <= 0) this.kill(target, attacker, cause);
  }

  kill(target, attacker, cause) {
    target.alive = false;
    target.health = 0;
    target.deathAt = this.t;
    target.respawnTimer = this.t + PHYS.respawnDelay;
    target.stats.deaths++;
    target.vx = target.vz = 0;
    if (target.kind === 'player') this.telemetry.playerDeaths++;
    if (target.kind === 'bot') this.telemetry.botDeaths++;
    if (target.kind === 'clark') {
      // Killing Clark is a big deal. Survivors are rewarded for it, and he is
      // NOT punished - the reward is what he learns "getting shot hurts my
      // long-run value" from.
      for (const o of this.entities) if (this.hostile(o, target)) o.pendingReward += 1.4;
      this.emit({ type: 'death', kind: 'clark', by: attacker ? attacker.kind : cause, x: target.x, z: target.z });
      return;
    }
    // A survivor died.
    if (attacker && attacker.faction === FACTION.CLARK) {
      attacker.stats.kills++;
      attacker.pendingReward += this.rewards.kill;
      this.telemetry.clarkKills++;
      this.emit({ type: 'death', kind: target.kind, by: 'clark', x: target.x, z: target.z });
    } else {
      this.emit({ type: 'death', kind: target.kind, by: cause, x: target.x, z: target.z });
    }
    target.pendingReward += this.rewards.caught;
  }

  respawnSurvivor(e) {
    const p = this.pickSpawn(
      this.entities.filter((o) => o.alive && o.id !== e.id).map((o) => o.id),
      4,
    );
    e.x = p.x;
    e.z = p.z;
    e.vx = e.vz = 0;
    e.alive = true;
    e.health = e.maxHealth;
    e.spawnedAt = this.t;
    e.action = 0;
    e.dazedUntil = 0;
    e.stunUntil = 0;
    e.slowUntil = 0;
    e.stamina = 1;
    this.telemetry.respawns++;
    this.emit({ type: 'respawn', kind: e.kind, x: e.x, z: e.z });
  }

  /** Reward shaping for this tick, written into entity.pendingReward. */
  computeShaping(dt, alive) {
    const R = this.rewards;
    const sparse = R.preset === 'sparse';
    for (const e of alive) {
      const isClark = e.kind === 'clark';
      e.pendingReward += isClark ? (R.preset === 'hunt-only' ? R.step * 0.5 : R.step) : R.survive;
      // per-entity detected set -> previous distances
      let prev = this.detectedPrev.get(e.id);
      if (!prev) {
        prev = new Map();
        this.detectedPrev.set(e.id, prev);
      }
      const det = this.detected(e);
      if (isClark) {
        if (det.length) this.telemetry.clarkSees++;
        for (const o of det) {
          const d = dist(e, o);
          if (d < this.telemetry.clarkClosest) this.telemetry.clarkClosest = d;
        }
      } else if (det.length) this.telemetry.botSees++;
      const nowIds = new Set();
      for (const o of det) {
        const d = dist(e, o);
        nowIds.add(o.id);
        const pd = prev.get(o.id);
        if (pd !== undefined) {
          const delta = d - pd;
          if (isClark) {
            // Engagement shaping: closing on something he can currently
            // perceive. Deleted entirely under the 'sparse' preset.
            if (!sparse && delta < 0) e.pendingReward += -delta * R.closing;
          } else if (delta > 0) {
            // survivors: rewarded for opening the gap on a perceived Clark
            e.pendingReward += delta * R.flee;
          }
        }
        prev.set(o.id, d);
        if (isClark && !sparse && o === det[0]) {
          // ...and turning to look at it. Without this the very first bites are
          // pure luck: nothing in the reward stream has any opinion about which
          // way he is facing until he is already inside biting range.
          const dx = o.x - e.x;
          const dz = o.z - e.z;
          const dd = Math.hypot(dx, dz) || 1;
          const facing = (dx / dd) * Math.sin(e.yaw) + (dz / dd) * Math.cos(e.yaw);
          const prevFacing = prev.get('facing:' + o.id);
          if (prevFacing !== undefined && facing > prevFacing) e.pendingReward += (facing - prevFacing) * R.orient;
          prev.set('facing:' + o.id, facing);
        }
      }
      for (const id of [...prev.keys()]) if (!nowIds.has(id)) prev.delete(id);

      const [cx, cz] = this.grid.cellOf(e.x, e.z);
      if (this.grid.inBounds(cx, cz)) {
        const v = this.features.visitMap(e.id);
        const i = this.grid.idx(cx, cz);
        if (!v[i]) {
          v[i] = 1;
          e.stats.novelty++;
          if (!sparse) e.pendingReward += isClark ? R.clarkNovelty : R.novelty;
        }
      }
      if (!isClark && !sparse) {
        const clark = this.clark;
        if (clark && clark.alive && !this.features.canSee(e, clark)) {
          for (const o of alive) {
            if (o === e || o.faction !== e.faction) continue;
            if (dist(e, o) < 6) e.pendingReward += R.flock * dt * 60 * 0.05;
          }
        }
      }
    }
  }
}
