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

## Controls

| | |
|---|---|
| `W A S D` | move (walk) |
| `Shift` | run — the only way to outpace him |
| mouse | look (click to capture the pointer) |
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
player and the bots.** The infrastructure for that is complete and verified —
learned model, search, rewards, checkpointing, a headless trainer and a browser
lab — and the mechanics support it: with a hand-written greedy controller he
kills a fleeing bot from 10 m in ~6.6 s, so a kill is reachable, and with
entirely random actions (no learning at all) he already reaches biting range and
bites. Whether the *learning* reliably produces a hunter within a few minutes of
CPU is the open question, and `docs/LEARNING.md` keeps the measured log rather
than a claim.

Two findings from building it are worth knowing, because both were bugs of
design rather than of code:

1. A **dense maze makes the task unlearnable in practice.** With 40–60% walls,
   closing reward is almost unreachable, exploration never stumbles into a bite,
   and the value function stays flat — the agent's best policy is to stand
   still. The default layout is now the film's open floor plate (18% walls,
   partitions with proper doorways, column clusters), which is both more
   faithful to *Backrooms* and lets "walk at the thing" be a discoverable
   policy. `?layout=maze` restores the old generator for comparison.
2. The agent must not be able to reach a safe equilibrium. Standing still is
   penalised, exploration never fully anneals, and the policy target is
   deliberately kept slightly flat, because in a sparse-reward world a saturated
   policy cannot stumble onto the behaviour you want it to learn.

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
node tools/balance-probe.mjs    # is a kill mechanically reachable? (no learning)
node tools/ui-smoke.mjs         # loads every browser module against a stub DOM
node tools/train.mjs --minutes 20 --profile turbo --out public/brain
node tools/diagnose.mjs 120 --learn        # learning diagnostics, no renderer
node tools/diagnose.mjs 60 --checkpoint public/brain.json
```

`train.mjs` and `diagnose.mjs` run the real session with a scripted fleeing
player (`--static-player` disables it) so headless runs give the monster a
moving target.

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
