# Architecture

The point of the split is that **the game is optional**. `src/sim` and `src/ai`
know nothing about three.js, the DOM or the browser; `src/render` and `src/ui`
know nothing about learning. `tools/train.mjs` runs the whole simulation and
training loop with no DOM at all, and `src/main.js` runs the same objects with a
renderer attached.

```
                 ┌────────────────────────── src/sim ──────────────────────────┐
                 │ grid.js      level generation (open floor plate / maze)     │
                 │ features.js  perception + the egocentric observation encoder │
                 │ world.js     physics, combat, rewards, telemetry            │
                 │ config.js    every tunable in one object                    │
                 └────────────────────────────────────────────────────────────┘
                                        ▲            │
                                        │            ▼
                 ┌──────────────────────── src/ai ───────────────────────────┐
                 │ core/tape.js  reverse-mode autodiff (Float64)              │
                 │ networks.js   representation / dynamics / prediction nets  │
                 │ mcts.js       PUCT search over the *learned* model         │
                 │ brain.js      EZBrain: act, replay, train, reanalyze       │
                 │ session.js    Session: owns world + brains, steps everything│
                 └────────────────────────────────────────────────────────────┘
                          ▲                                    ▲
        ┌─────────────────┴───────────────┐        ┌───────────┴──────────────┐
        │ src/render, src/ui, src/main.js │        │ tools/*.mjs, lab worker  │
        │ browser: three.js + HUD + lab   │        │ node / WebWorker training│
        └─────────────────────────────────┘        └──────────────────────────┘
```

## Simulation

* **Grid** — 25 × 25 cells of 2.6 m (65 m across), 3.1 m ceiling. The default
  `backrooms` layout is an open floor plate: partition walls with doorways,
  column clusters, and 55 pieces of furniture (ottomans included, for reasons
  that belong to the film). A flood fill verifies full connectivity. Spawn
  points are ranked by local openness so nobody materialises inside a pillar.
* **Perception** — `canSee` = distance + 105° cone + line of sight;
  `canHear` = distance only, through walls, 13 m. `detected()` is the union per
  entity and is the *only* channel through which an agent learns anything about
  another body.
* **Observation** — one egocentric 12 × 12 m crop (6 channels: walls and
  furniture, bodies, own scent, rival scent, noise, unexplored), pooled 2 × 2 by
  fixed max/mean to 216 floats, plus 13 scalars (own pose/health, range and
  bearing of the nearest perceived hostile, clearance ahead, speed, time since
  damage, bearing of last damage). Fixed, weight-free pooling keeps a decision
  affordable at 60 fps on two CPU cores.
* **Physics** — 60 Hz fixed step, body-relative action set (forward/back/strafe/
  diagonals/turn/lunge/roar), so a policy's action never changes meaning as the
  body turns. Sprint is player-only. Collision is resolved against walls then
  props; the reward code distinguishes "pressed into a wall" from "slid along"
  from "stood still".
* **Combat** — Clark bites for 38 damage inside 1.15 m, shoves survivors, and
  roars to daze them (×0.45 speed for 1.15 s inside 9.5 m). Survivors fire a
  weak flashlight-pistol. Survivors respawn away from Clark and vice versa.
* **Determinism** — a seeded `mulberry32` drives level generation, spawns,
  search noise and action sampling. Same seed, same session.

## Learning

`Session` owns one world and two brains: `bots` (optionally shared by the four
survivors) and `clark`. Every `decisionInterval` ticks it runs `runDecisions`,
which is the entire agent loop:

1. consume the reward that accumulated since this entity's last decision;
2. push that transition into the brain's replay buffer;
3. encode the current observation;
4. `brain.act(...)` — representation net, then MCTS over the learned dynamics;
5. write the chosen action back into the world.

Every rendered frame, `Session.frame()` calls `brain.update(budgetMs)` which runs
gradient steps until a wall-clock budget is exhausted — the trainer spends
seconds per frame, the browser spends a few milliseconds, and both call exactly
the same `trainOnce()`.

### Observation, action and reward plumbing

`Session` is the only place where the agent's view of the world and its reward
meet, which is why `diagnose.mjs` can print both head losses and kill telemetry
in the same table.

## Checkpoints

`Session.checkpoint()` produces `hussrooms-checkpoint-v1`: both brains'
parameters and Adam state, their counters and history, as Float32 arrays encoded
as base64 (a deliberate ~1e-7 rounding). The replay buffer is *not* included: a
full buffer is megabytes of frame data, and a monster does not need his memories
to be portable. The lab panel, `tools/train.mjs --out` and
`tools/diagnose.mjs --checkpoint` all speak this format, so a brain trained in a
browser worker can be dropped into node and back.

## Rendering

`WorldView` builds one InstancedMesh per material (walls, light panels, each
prop type) so a 625-cell level is about ten draw calls. `ClarkMesh` and
`SurvivorMesh` are primitive-built bodies animated procedurally from simulation
state — walk cycle from speed, jaw from bite/roar state, body pitch from
`lungeUntil`, collapse when dead. No textures on disk: the yellow wallpaper,
carpet, ceiling tiles and light panels are painted into canvases at load
(`src/render/textures.js`). `three.js` r186 is vendored in `vendor/three/` and
imported through `src/three-shim.js`, because the preview environment has no
outbound network.
