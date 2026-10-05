/**
 * Balance probe: is a kill mechanically reachable at all?
 *
 *   node tools/balance-probe.mjs
 *
 * If even a hand-written greedy controller cannot get bites, then no amount of
 * learning will produce them and the balance/mechanics are at fault. This asks
 * that question three ways: random actions, a greedy Clark against random prey,
 * and a greedy Clark against a fleeing bot.
 */

import { World, FACTION } from '../src/sim/world.js';
import { A, ACTIONS, PHYS } from '../src/sim/config.js';

const SEED = 4242;

function freshWorld() {
  const w = new World({ seed: SEED });
  return w;
}

/** Same local->world mapping as World.applyMovement. */
function worldDelta(e, actionIdx) {
  const move = ACTIONS[actionIdx];
  const fx = Math.sin(e.yaw);
  const fz = Math.cos(e.yaw);
  const rx = fz;
  const rz = -fx;
  return { x: move.dx * rx + move.dz * fx, z: move.dx * rz + move.dz * fz, turn: move.turn };
}

/** Place Clark and bot #0 at a given centre distance with clear line of sight. */
function place(w, metres) {
  const clark = w.clark;
  const bots = w.entities.filter((e) => e.kind === 'bot');
  const pts = w.grid.spawnPoints;
  let best = null;
  for (const a of pts) {
    for (const b of pts) {
      const d = Math.hypot(a.x - b.x, a.z - b.z);
      if (Math.abs(d - metres) < 1.2 && w.grid.lineOfSight(a.x, a.z, b.x, b.z)) {
        best = [a, b];
        break;
      }
    }
    if (best) break;
  }
  if (!best) throw new Error(`no clear pair at ${metres} m`);
  const [ca, bb] = best;
  clark.x = ca.x;
  clark.z = ca.z;
  clark.yaw = Math.atan2(bb.x - ca.x, bb.z - ca.z);
  bots[0].x = bb.x;
  bots[0].z = bb.z;
  // park everyone else far away so they do not interfere
  const far = pts[pts.length - 1];
  for (let i = 1; i < bots.length; i++) {
    bots[i].x = 1.5 + i * 0.5;
    bots[i].z = 1.5;
    bots[i].health = 0;
    bots[i].alive = false;
    bots[i].respawnTimer = 1e9;
  }
  w.player.x = 1.2;
  w.player.z = 1.2;
  w.player.health = 0;
  w.player.alive = false;
  w.player.respawnTimer = 1e9;
  return { clark, bot: bots[0] };
}

function greedyClarkAction(w, clark, lungeOk = true) {
  const det = w.detected(clark);
  if (!det.length) {
    // wander: keep walking, occasionally turn
    return w.tick % 120 < 60 ? A.fwd : A.turnL;
  }
  let target = det[0];
  let bd = Infinity;
  for (const o of det) {
    const d = Math.hypot(o.x - clark.x, o.z - clark.z);
    if (d < bd) {
      bd = d;
      target = o;
    }
  }
  const dx = target.x - clark.x;
  const dz = target.z - clark.z;
  const dl = Math.hypot(dx, dz) || 1;
  const desired = { x: dx / dl, z: dz / dl };
  const facing = { x: Math.sin(clark.yaw), z: Math.cos(clark.yaw) };
  const dot = facing.x * desired.x + facing.z * desired.z;
  let err = Math.atan2(dx, dz) - Math.atan2(facing.x, facing.z);
  while (err > Math.PI) err -= Math.PI * 2;
  while (err < -Math.PI) err += Math.PI * 2;
  if (bd < PHYS.roarRadius * 0.9 && clark.roarCooldown <= 0 && dot > 0.8) return A.roar;
  if (bd < 4.6 && clark.lungeCooldown <= 0 && dot > 0.9 && !(w.t < clark.lungeUntil)) return A.lunge;
  if (Math.abs(err) > 0.6) return err > 0 ? A.turnL : A.turnR;
  let best = A.fwd;
  let bestScore = -Infinity;
  for (let a = 0; a <= A.turnR; a++) {
    const d = worldDelta(clark, a);
    const score = d.x * desired.x + d.z * desired.z - Math.abs(d.turn) * 0.01;
    if (score > bestScore) {
      bestScore = score;
      best = a;
    }
  }
  return best;
}

function fleeBotAction(w, bot, clark) {
  const dx = bot.x - clark.x;
  const dz = bot.z - clark.z;
  const dl = Math.hypot(dx, dz) || 1;
  const desired = { x: dx / dl, z: dz / dl };
  const facing = { x: Math.sin(bot.yaw), z: Math.cos(bot.yaw) };
  let err = Math.atan2(dx, dz) - Math.atan2(facing.x, facing.z);
  while (err > Math.PI) err -= Math.PI * 2;
  while (err < -Math.PI) err += Math.PI * 2;
  if (Math.abs(err) > 2.4) return err > 0 ? A.turnL : A.turnR; // spin then run
  if (Math.abs(err) > 0.5) return err > 0 ? A.fwdL : (A.fwdR ?? A.fwd);
  let best = A.fwd;
  let bestScore = -Infinity;
  for (let a = 0; a <= A.turnR; a++) {
    const d = worldDelta(bot, a);
    const score = d.x * desired.x + d.z * desired.z;
    if (score > bestScore) {
      bestScore = score;
      best = a;
    }
  }
  return best;
}

function run(label, { seconds = 45, clarkPolicy, botPolicy, startDist = 10 }) {
  const w = freshWorld();
  const { clark, bot } = place(w, startDist);
  const rngState = { s: 12345 };
  const rnd = () => {
    rngState.s = (rngState.s * 1103515245 + 12345) & 0x7fffffff;
    return rngState.s / 0x7fffffff;
  };
  let minDist = Infinity;
  let firstBiteAt = null;
  let firstKillAt = null;
  let bites = 0;
  const steps = Math.round(seconds / PHYS.dt);
  for (let i = 0; i < steps; i++) {
    const ca = clarkPolicy === 'random' ? Math.floor(rnd() * 13) : greedyClarkAction(w, clark);
    w.setAction(clark, ca);
    if (bot.alive) {
      const ba = botPolicy === 'random' ? Math.floor(rnd() * 11) : fleeBotAction(w, bot, clark);
      w.setAction(bot, ba);
      bot.sprinting = botPolicy === 'flee';
    }
    w.pendingReward = w.pendingReward; // no-op, keeps shape explicit
    clark.pendingReward = 0;
    w.step(PHYS.dt);
    if (bot.alive) {
      const d = Math.hypot(bot.x - clark.x, bot.z - clark.z);
      if (d < minDist) minDist = d;
    }
    if (w.telemetry.bites > bites) {
      bites = w.telemetry.bites;
      if (firstBiteAt == null) firstBiteAt = w.t;
    }
    if (w.telemetry.clarkKills > 0 && firstKillAt == null) firstKillAt = w.t;
    if (!bot.alive) break;
  }
  const fmt = (v) => (v == null ? '—' : v.toFixed(1) + 's');
  console.log(
    `${label.padEnd(38)} min dist ${minDist.toFixed(2)} m · bites ${bites} · first bite ${fmt(firstBiteAt)} · ` +
      `kill ${fmt(firstKillAt)} · bot alive ${bot.alive} (hp ${Math.max(0, bot.health).toFixed(0)}) · t=${w.t.toFixed(1)}s`,
  );
  return { minDist, bites, firstBiteAt, firstKillAt };
}

console.log('=== reachability of a kill (no learning involved) ===');
run('random Clark vs random prey @10m', { clarkPolicy: 'random', botPolicy: 'random', startDist: 10 });
run('random Clark vs random prey @4m', { clarkPolicy: 'random', botPolicy: 'random', startDist: 4 });
run('greedy Clark vs random prey @10m', { clarkPolicy: 'greedy', botPolicy: 'random', startDist: 10 });
run('greedy Clark vs random prey @4m', { clarkPolicy: 'greedy', botPolicy: 'random', startDist: 4 });
run('greedy Clark vs FLEEING prey(walk) @10m', { clarkPolicy: 'greedy', botPolicy: 'flee', startDist: 10 });
run('greedy Clark vs FLEEING prey(walk) @4m', { clarkPolicy: 'greedy', botPolicy: 'flee', startDist: 4 });
