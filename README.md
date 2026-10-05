# HUS$ROOMS

A Backrooms prototype with one rule: **the monster is not scripted.**

You are in a small, endless-feeling yellow level with four respawning survivors.
Hunting you is Cap'n Clark — the Still Life from Kane Parsons' *Backrooms* — and
he is controlled by an EfficientZero agent: a learned latent dynamics model, a
prediction network, MCTS over that learned model, self-supervised consistency,
value-prefix learning and Reanalyze. He starts knowing nothing. There is no
"if player is visible, walk towards player" anywhere in this repository, no
waypoint, no pathfinder, no target memory. If he ever hunts you, it is because
the reward stream plus search taught him that hunting pays.

Everything is plain JavaScript, ES modules, no build step, no bundler, no
dependencies at runtime (three.js is vendored). The same `src/sim` and `src/ai`
code runs in the browser and in node.

```
node server.js            # then open http://localhost:5173
```

The start screen offers two ways in: **start fresh** (he knows nothing, which is
the point — the untrained Clark is the default), or **load the trained Clark**,
which installs `public/brain-pretrained.json`: the brain from run #5 in
`docs/LEARNING.md`, the first one that learned to hunt on its own. The same file
can be requested with `?checkpoint=public/brain-pretrained.json`.

## Controls

| | |
|---|---|
| `W A S D` | move (walk) |
| `Shift` | run — faster than him, but only for ~3 s of breath at a time |
| mouse | look (click to capture the pointer; hold the button and drag if the browser refuses) |
| `← → ↑ ↓` / `Q E` | look, no mouse needed — this is how it stays playable in an iframe |
| click / `space` | fire the flashlight-pistol |
| `Tab` | **the EfficientZero lab panel** — his search tree, live |
| `P` | toggle learning |

You can also watch him think: the lab panel shows the root search (which actions
he considered, their visit counts, priors and Q values), the principal variation
with predicted rewards along it, his value estimate, replay size, gradient-step
count, loss, search entropy and his action histogram. `Tab` also has
save/load/reset for the brain.

## What he can actually perceive

Nothing about you is handed to him as "the player". Each decision he gets:

* a 12×12 m egocentric crop of the level (6 channels: walls & furniture, bodies,
  his own scent, rival scent, noise, unexplored), max/mean-pooled to 216 numbers;
* 13 scalars: his own position, facing, health, the range and bearing of the
  nearest *perceived* hostile, clearance ahead, his speed, time since damage,
  and the bearing of whoever last hit him.

Perception is modelled, not omniscient: sight needs line of sight inside a 105°
cone, hearing works through walls within 13 m. He is rewarded for closing on
hostiles he currently perceives (+0.45/m), for turning to face them, for
biting, for roaring when it dazes someone, and heavily for a kill. He can also
roar (dazes survivors within 9.5 m) and lunge 7.4 m/s for half a second — both
are just actions in his action set, never used by script.

## Honest status of the acceptance criterion

The goal is: **Clark eventually learns on his own to chase and try to kill the
player and the bots.**

He has. In run #5 (`docs/LEARNING.md`), starting from random actions with no
scripted behaviour of any kind, he lands his first bites at t ≈ 300–360 s of
simulation and his first kill shortly after, reaching 3 kills and 9 bites within
13 minutes of simulated play. Replaying that checkpoint cold — fresh world,
fresh spawn, learning off — he closes to 2.0 m of his nearest victim, against
9.0 m in every earlier attempt. That brain is the one shipped as
`public/brain-pretrained.json`.

What the claim does not cover, and the log records in full: kills are still
sparse (roughly one every few minutes of sim), the survivor brain can degenerate
into a strafe loop over a long run, and the losses that are still missing would
matter in a longer run.

Four findings from building it were worth the wall clock, and all four were
design bugs rather than code bugs:

1. A **dense maze makes hunting unlearnable in practice.** With 40–60% walls,
   closing reward is almost unreachable, exploration never stumbles into a bite,
   and the value function stays flat. The default layout is now the film's open
   floor plate (18% walls, partitions with doorways, column clusters), which is
   both more faithful to *Backrooms* and lets "walk at the thing" be a
   discoverable policy. `?layout=maze` restores the old generator.
2. **A predator needs a reason to move when it cannot see prey.** With novelty
   exhausted and nobody in perception, every nearby reward was a penalty, and
   the policy froze at ~59% `idle` — a self-reinforcing attractor, because a
   frozen monster never meets prey. Motion is now weakly positive everywhere
   (`roam: 0.004`/metre travelled, capped per tick). That is a locomotion prior,
   never a direction: it says "keep moving", not "move towards something".
3. **Shaping must be potential-based.** Paying per tick for closing distance
   (or for opening it, on the survivor side) pays any periodic motion, and a
   farmed shaping term is what the policy learns *instead of* hunting — one run
   ended with the surviving bots circling in place at 63% `strafeL`. Approach,
   facing and bite-range contact are now differences of potentials, so
   oscillating nets exactly zero while approaching nets the full difference.
4. **A one-off bite cannot teach anything.** The first bite used to be the last
   one, because the victim simply kept running. A bite now staggers whoever it
   lands on, so a landed bite usually leads to a kill — which is what turns a
   sparse, unreachable event into a learnable one.

## The browser lab (`/src/ai/lab.html`)

A headless copy of the session running in a Web Worker as fast as the machine
allows, with a scripted fleeing stand-in for the player, live charts, and
save/load. Leave the tab open and it trains; the checkpoint it writes is the
same format `node tools/train.mjs` writes and can be dropped into the game, into
`tools/diagnose.mjs --checkpoint`, or into `public/brain.json` (which the game
loads automatically if present).

URL parameters for `index.html`: `?learn=0`, `?explore=1`, `?speed=2`,
`?profile=turbo`, `?layout=maze`, `?quality=low`, `?checkpoint=public/brain.json`.

## Node tools

```
node tools/test-grad.mjs        # reverse-mode autodiff gradient checks (7)
node tools/smoke.mjs 12         # headless session smoke test
node tools/bench.mjs            # timings for sim step / decision / train step
node tools/map-preview.mjs      # ASCII view of the generated level + stats
node tools/balance-probe.mjs    # is a kill reachable, and does reward prefer hunting?
node tools/ui-smoke.mjs         # loads every browser module against a stub DOM
node tools/train.mjs --minutes 20 --profile turbo --out public/brain
node tools/diagnose.mjs 120 --learn        # learning diagnostics, no renderer
node tools/diagnose.mjs 60 --checkpoint public/brain.json
```

`train.mjs` and `diagnose.mjs` run the real session with a scripted fleeing
player (`--static-player` disables it) so headless runs give the monster a
moving target. `balance-probe.mjs` also compares the reward a scripted hunter
earns against a scripted idler over the same 60 s: if the idler ever wins, the
reward function — not the agent — is the thing that needs fixing.

## Layout

```
index.html, src/main.js      game entry (browser): sim + three.js + HUD
src/sim/                     the world: grid/props/perception/obs, no AI
src/ai/                      EfficientZero: tape autodiff, nets, MCTS, brain, session
src/ai/lab.html + lab-*.js   headless in-browser trainer (Web Worker)
src/render/                  three.js only: level geometry, Clark's body, actors
src/ui/                      HUD + lab panel (DOM only)
server.js                    zero-dependency static server
tools/                       node harness: train, diagnose, gradient checks, probes
docs/                        architecture + the learning log
vendor/three/                three.js r186, vendored so the preview works offline
```

The renderer contains no gameplay logic and the simulator contains no AI logic;
`tools/train.mjs` proves it by running the whole thing with no DOM at all.

## Cap'n Clark

Modelled from the film's design: a towering Still Life in the costume from the
"Cap'n Clark's Ottoman Empire" advert — teal shirt with an oversized collar,
yellow vest, purple sash, tricorn hat with feathers, a misaligned bulging face
that stretched upward, one arm jutting out at the angle of the crutch, and a leg
fused into a splintered table leg. He is built from primitives in
`src/render/clark.js` (no assets anywhere in this repo) and animated from his
simulation state, so what you see is what the agent actually did: the hobbling
limp, the jaw snapping open on a bite, the body pitching forward on a lunge.
Ottomans are a prop type for the same reason.
