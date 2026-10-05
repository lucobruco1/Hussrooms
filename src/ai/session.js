/**
 * Session - the experiment harness.
 *
 * Owns one World and one EZBrain per agent, drives the decision cadence, moves
 * rewards from the world into the replay buffers, spends the per-frame learning
 * budget, and exposes the telemetry the UI draws. Runs identically in Node
 * (tools/) and in the browser.
 *
 *   world tick (60 Hz)
 *     |- every decisionInterval ticks: obs -> brain.act() -> world.setAction()
 *     |- rewards accumulate in world.pendingReward, consumed at decisions
 *     `- every frame: brain.update(budgetMs)   (gradient steps)
 */

import { ACTIONS, CLARK_ACTIONS, LEARN, PHYS, SURVIVOR_ACTIONS } from '../sim/config.js';
import { World } from '../sim/world.js';
import { obsDims, poolObs } from '../sim/features.js';
import { EZBrain } from './brain.js';
import { dist } from '../sim/util.js';

export class Session {
  constructor(opts = {}) {
    this.opts = opts;
    this.seed = opts.seed ?? 20260529;
    this.world = new World({ seed: this.seed, rewards: opts.rewards });
    this.dims = obsDims();
    this.learn = opts.learn !== false;
    this.shareBotBrains = opts.shareBotBrains ?? LEARN.shareBotBrains;
    this.simSpeed = opts.simSpeed ?? LEARN.simSpeed;
    this.profile = opts.profile || 'live';
    this.brains = new Map(); // entityId -> EZBrain
    this.brainList = [];
    this.pending = new Map();
    this.pooledBuf = new Map();
    this.rollingKills = [];
    this.rollingReward = new Map();
    this.playerInput = { action: 0, sprint: false, shoot: false };
    this.lastEvents = [];
    this.stats = {
      updates: 0,
      trainMs: 0,
      simSteps: 0,
      killsPerMin: 0,
      lastKillAt: null,
      firstKillAt: null,
      clarkDeaths: 0,
      playerDeaths: 0,
    };
    this.forceExplore = false; // lab switch: sample from the raw search prior, hot
    this.autoPlayer = !!opts.autoPlayer; // headless stand-in for the human
    this.autoPlayerYaw = this.rng ? 0 : 0;
    this.autoPlayerTarget = null;
    this.autoPlayerSprint = false;
    this.buildBrains();
  }

  buildBrains() {
    const botBrain = new EZBrain({
      id: 'bots',
      label: this.shareBotBrains ? 'Survivor bots (shared brain)' : 'Survivor bot',
      obsDims: this.dims,
      actionDim: SURVIVOR_ACTIONS,
      seed: this.seed ^ 0x1234,
      profile: this.profile,
      cfg: this.opts.botCfg || {},
    });
    const clarkBrain = new EZBrain({
      id: 'clark',
      label: "Cap'n Clark",
      obsDims: this.dims,
      actionDim: CLARK_ACTIONS,
      seed: this.seed ^ 0xabcd,
      profile: this.profile,
      cfg: this.opts.clarkCfg || {},
    });
    for (const e of this.world.entities) {
      if (e.kind === 'player') continue;
      const b = e.kind === 'clark' ? clarkBrain : botBrain;
      this.brains.set(e.id, b);
    }
    this.brainList = [botBrain, clarkBrain];
    this.botBrain = botBrain;
    this.clarkBrain = clarkBrain;
  }

  /**
   * Throw away both brains and start from scratch - the lab panel's reset
   * button. The map, the survivors and the player are left exactly as they are.
   */
  resetLearning() {
    this.brains.clear();
    this.pending.clear();
    this.pooledBuf.clear();
    this.buildBrains();
    this.stats.updates = 0;
    this.stats.trainMs = 0;
    this.stats.firstKillAt = null;
    this.rollingKills = [];
    this.lastEvents = this.lastEvents.concat([{ type: 'reset', t: this.world.t }]).slice(-60);
  }

  /** Re-point brains after a respawn created a new entity object. */
  refreshBrainMap() {
    for (const e of this.world.entities) {
      if (e.kind === 'player') continue;
      if (!this.brains.has(e.id)) {
        this.brains.set(e.id, e.kind === 'clark' ? this.clarkBrain : this.botBrain);
      }
    }
  }

  /** One 60 Hz simulation step plus decision + learning bookkeeping. */
  step() {
    const w = this.world;
    const interval = this.botBrain.cfg.decisionInterval || LEARN.decisionInterval;
    if (w.tick % interval === 0) this.runDecisions();
    // The scripted player's sprint flag has to be re-applied every tick, the
    // same way the renderer re-applies the human's key state every frame.
    if (this.autoPlayer && w.player) w.player.sprinting = this.autoPlayerSprint;
    w.step(PHYS.dt);
    this.stats.simSteps++;
    this.trackEvents();
  }

  /**
   * Scripted stand-in for the human player, used only by the headless trainer
   * (opts.autoPlayer). It sprints away from Cap'n Clark when it sees him and
   * wanders otherwise, which is precisely the moving, fleeing target the real
   * player is. It is NOT one of the learning agents - it exists so that
   * headless self-play has prey that behaves like prey.
   */
  autoPlayerTick() {
    const p = this.world.player;
    if (!p || !p.alive) return;
    const w = this.world;
    const clark = w.clark;
    const seen = clark && clark.alive && w.features.canSee(p, clark) && dist(p, clark) < 14;
    if (seen) {
      this.autoPlayerTarget = clark;
      const away = Math.atan2(p.x - clark.x, p.z - clark.z);
      let d = away - p.yaw;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      p.yaw += Math.max(-3.4 * PHYS.dt * 4, Math.min(3.4 * PHYS.dt * 4, d));
      w.setAction(p, d > 0.4 ? 5 : d < -0.4 ? 6 : 1); // fwdL / fwdR / fwd
      // sprint in bursts only: it cannot start a sprint on an empty tank
      this.autoPlayerSprint = p.stamina > PHYS.staminaMinStart;
      if (dist(p, clark) < PHYS.playerRange) this.playerShoot();
    } else {
      if (!this.autoPlayerTarget || this.world.tick % 120 === 0) {
        this.autoPlayerTarget = { yaw: Math.random() * Math.PI * 2, until: this.world.t + 3 + Math.random() * 4 };
      }
      if (this.world.t > this.autoPlayerTarget.until) this.autoPlayerTarget = null;
      if (this.autoPlayerTarget) {
        let d = this.autoPlayerTarget.yaw - p.yaw;
        while (d > Math.PI) d -= Math.PI * 2;
        while (d < -Math.PI) d += Math.PI * 2;
        p.yaw += Math.max(-3.4 * PHYS.dt * 4, Math.min(3.4 * PHYS.dt * 4, d));
        w.setAction(p, Math.abs(d) > 0.5 ? 5 : 1);
        this.autoPlayerSprint = false;
      }
    }
  }

  runDecisions() {
    const w = this.world;
    this.refreshBrainMap();
    if (this.autoPlayer) this.autoPlayerTick();
    for (const e of w.entities) {
      const brain = this.brains.get(e.id);
      if (!brain) continue;
      if (!e.alive) {
        const p = this.pending.get(e.id);
        if (p) {
          const r = this.consumeReward(e);
          brain.observe({
            frame: p.frame,
            scalars: p.scalars,
            action: p.action,
            reward: r,
            policy: p.policy,
            value: p.value,
            done: true,
          });
          this.pending.delete(e.id);
        }
        continue;
      }
      const obs = w.observe(e);
      let pooled = this.pooledBuf.get(e.id);
      if (!pooled) {
        pooled = new Float32Array(this.dims.pooled);
        this.pooledBuf.set(e.id, pooled);
      }
      poolObs(obs.frame, pooled);
      const p = this.pending.get(e.id);
      if (p) {
        const r = this.consumeReward(e);
        brain.observe({
          frame: p.frame,
          scalars: p.scalars,
          action: p.action,
          reward: r,
          policy: p.policy,
          value: p.value,
          done: false,
        });
      }
      const res = brain.act(pooled, obs.scalars, {
        explore: true,
        temperature: this.forceExplore ? 1 : undefined,
      });
      w.setAction(e, res.action);
      this.pending.set(e.id, {
        frame: pooled.slice(),
        scalars: obs.scalars.slice(),
        action: res.action,
        policy: res.probs,
        value: res.value,
      });
    }
  }

  consumeReward(e) {
    const r = e.pendingReward;
    e.pendingReward = 0;
    const arr = this.rollingReward.get(e.id) || [];
    arr.push(r);
    if (arr.length > 400) arr.shift();
    this.rollingReward.set(e.id, arr);
    return r;
  }

  trackEvents() {
    const evs = this.world.drainEvents();
    if (!evs.length) return;
    for (const ev of evs) {
      if (ev.type === 'death' && ev.by === 'clark' && ev.kind !== 'clark') {
        this.rollingKills.push(this.world.t);
        if (this.stats.firstKillAt == null) this.stats.firstKillAt = this.world.t;
        this.stats.lastKillAt = this.world.t;
      }
      if (ev.type === 'death' && ev.kind === 'clark') this.stats.clarkDeaths++;
      if (ev.type === 'death' && ev.kind === 'player') this.stats.playerDeaths++;
    }
    this.lastEvents = this.lastEvents.concat(evs).slice(-60);
  }

  /** One rendered frame: `simSpeed` sim steps, then the learning budget. */
  frame() {
    const sp = Math.max(1, Math.round(this.simSpeed));
    for (let i = 0; i < sp; i++) this.step();
    this.learn1();
    this.updateRates();
    this.stats.updates = this.brainList.reduce((a, b) => a + b.updates, 0);
    this.stats.trainMs = this.brainList.reduce((a, b) => a + b.trainMsUsed, 0);
  }

  learn1(budgetOverride) {
    if (!this.learn) return;
    const budget = budgetOverride ?? LEARN.trainBudgetMs;
    for (const b of this.brainList) b.update(budget, LEARN.opsPerFrame);
  }

  updateRates() {
    const t = this.world.t;
    while (this.rollingKills.length && t - this.rollingKills[0] > 60) this.rollingKills.shift();
    this.stats.killsPerMin = this.rollingKills.length;
  }

  // ------------------------------------------------------------------ input
  setPlayerInput(input) {
    this.playerInput = { ...this.playerInput, ...input };
    const p = this.world.player;
    if (!p || !p.alive) return;
    p.sprinting = !!this.playerInput.sprint;
    this.world.setAction(p, this.playerInput.action | 0);
  }

  playerShoot() {
    const p = this.world.player;
    if (!p || !p.alive) return false;
    const w = this.world;
    const target = w.clark;
    if (!target || !target.alive) return false;
    const d = dist(p, target);
    if (d > PHYS.playerRange) return false;
    const dx = (target.x - p.x) / d;
    const dz = (target.z - p.z) / d;
    const fx = Math.sin(p.yaw);
    const fz = Math.cos(p.yaw);
    const dot = dx * fx + dz * fz;
    if (dot < Math.cos((10 * Math.PI) / 180)) return false; // generous cone
    if (!w.grid.lineOfSight(p.x, p.z, target.x, target.z, true)) return false;
    w.damage(p, target, PHYS.playerDamage, 'gun');
    w.emit({ type: 'gunfire', x: p.x, z: p.z });
    return true;
  }

  // ------------------------------------------------------------- telemetry
  agentStats() {
    const w = this.world;
    const out = [];
    for (const e of w.entities) {
      const brain = this.brains.get(e.id);
      out.push({
        id: e.id,
        kind: e.kind,
        alive: e.alive,
        health: e.health,
        maxHealth: e.maxHealth,
        x: e.x,
        z: e.z,
        yaw: e.yaw,
        kills: e.stats.kills,
        deaths: e.stats.deaths,
        novelty: e.stats.novelty,
        distance: e.stats.distance,
        lifetimeReward: e.lifetimeReward,
        brain: brain ? brain.id : 'human',
        value: brain && brain._lastSearch ? brain._lastSearch.value : 0,
        entropy: brain ? brain.policyEntropy : 0,
        updates: brain ? brain.updates : 0,
        replay: brain ? brain.replay.size : 0,
        decisions: brain ? brain.decisions : 0,
        loss: brain ? brain.lossEma : 0,
      });
    }
    return out;
  }

  summary() {
    const w = this.world;
    const bots = w.entities.filter((e) => e.kind === 'bot');
    return {
      t: w.t,
      tick: w.tick,
      clarkKills: w.telemetry.clarkKills,
      playerDeaths: w.telemetry.playerDeaths,
      botDeaths: w.telemetry.botDeaths,
      bites: w.telemetry.bites,
      roars: w.telemetry.roars,
      lunges: w.telemetry.lunges,
      clarkHealth: w.clark ? w.clark.health : 0,
      clarkAlive: !!(w.clark && w.clark.alive),
      playerHealth: w.player.health,
      playerStamina: w.player.stamina ?? 1,
      playerAlive: w.player.alive,
      botsAlive: bots.filter((b) => b.alive).length,
      killsPerMin: this.stats.killsPerMin,
      firstKillAt: this.stats.firstKillAt,
      updates: this.stats.updates,
      trainMs: this.stats.trainMs,
      simSteps: this.stats.simSteps,
      clarkUpdates: this.clarkBrain.updates,
      botUpdates: this.botBrain.updates,
      clarkLoss: this.clarkBrain.lossEma,
      botLoss: this.botBrain.lossEma,
      clarkValue: this.clarkBrain._lastSearch ? this.clarkBrain._lastSearch.value : 0,
      botValue: this.botBrain._lastSearch ? this.botBrain._lastSearch.value : 0,
      clarkEntropy: this.clarkBrain.policyEntropy,
      botEntropy: this.botBrain.policyEntropy,
      clarkReplay: this.clarkBrain.replay.size,
      botReplay: this.botBrain.replay.size,
      clarkHistory: this.clarkBrain.history,
      botHistory: this.botBrain.history,
    };
  }

  /** Replay/value statistics - the numbers that reveal a value collapse. */
  brainStats(brainId) {
    const b = brainId === 'clark' ? this.clarkBrain : this.botBrain;
    const n = b.replay.size;
    if (!n) return null;
    let rSum = 0;
    let rMin = Infinity;
    let rMax = -Infinity;
    let vSum = 0;
    let vMin = Infinity;
    let vMax = -Infinity;
    let done = 0;
    let reanalyzed = 0;
    const N = Math.min(n, 3000);
    for (let i = n - N; i < n; i++) {
      const e = b.replay.at(i);
      rSum += e.reward;
      rMin = Math.min(rMin, e.reward);
      rMax = Math.max(rMax, e.reward);
      vSum += Math.abs(e.value);
      vMin = Math.min(vMin, e.value);
      vMax = Math.max(vMax, e.value);
      if (e.done) done++;
      if (e.reanalyzed) reanalyzed++;
    }
    return {
      size: n,
      updates: b.updates,
      meanReward: rSum / N,
      minReward: rMin,
      maxReward: rMax,
      meanAbsValue: vSum / N,
      minValue: vMin,
      maxValue: vMax,
      done,
      reanalyzed,
      valueMean: b.net.valueMean,
      valueSpread: b.net.valueSpread,
      loss: b.lossEma,
    };
  }

  /** How each brain spends its actions (a decent "is it alive?" sanity check). */
  actionHistogram(brainId) {
    const b = brainId === 'clark' ? this.clarkBrain : this.botBrain;
    const counts = new Array(ACTIONS.length).fill(0);
    for (let i = 0; i < b.replay.size; i++) counts[b.replay.at(i).action]++;
    const total = counts.reduce((a, c) => a + c, 0) || 1;
    return counts.map((c, i) => ({ action: ACTIONS[i].name, p: c / total }));
  }

  // ------------------------------------------------------------ persistence
  checkpoint() {
    return {
      format: 'hussrooms-checkpoint-v1',
      savedAt: new Date().toISOString(),
      seed: this.seed,
      simTime: this.world.t,
      worldTelemetry: this.world.telemetry,
      brains: {
        bots: this.botBrain.serialize(),
        clark: this.clarkBrain.serialize(),
      },
    };
  }

  loadCheckpoint(obj) {
    if (!obj || obj.format !== 'hussrooms-checkpoint-v1') return false;
    const a = this.botBrain.load(obj.brains.bots);
    const b = this.clarkBrain.load(obj.brains.clark);
    return a && b;
  }
}
