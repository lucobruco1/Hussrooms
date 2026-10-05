# The learning log

This file is the honest record for the acceptance criterion:

> **Cap'n Clark must eventually learn, on his own, to chase and try to kill the
> player and the bots. No hardcoded chase behaviour, no scripted targeting.**

Nothing in `src/ai` or `src/sim` knows what a chase is. Clark perceives a
192-cell egocentric crop and 13 scalars, picks from 13 body-relative actions
(walk/turn/strafe/lunge/roar) with MCTS over a learned latent model, and is paid
by the reward stream in `config.js REWARDS`. If he hunts, this is the whole
mechanism.

Evidence is kept as measurements, not adjectives. The proxies are:

| proxy | why |
|---|---|
| `bites`, `kills`, `firstKillAt` | the behaviour itself |
| `clarkSees` (% of ticks he has anybody in perception) | is contact even happening? |
| `clarkClosest` (closest perceived approach, m) | is he closing distance? |
| his action histogram | chasing looks like `fwd`/`fwdL`/`fwdR`/`lunge`/`roar`, not `idle`/`turnL` |
| `lost` / `V` / search entropy | is learning happening at all, or is the head saturated? |

## The reference point: what *should* happen

`node tools/balance-probe.mjs` answers "is a kill mechanically reachable, with
no learning involved?" On the current level, a hand-written greedy controller
that walks at the nearest perceived hostile kills a fleeing bot from 10 m in
**6.6 s**, and even a controller picking entirely random actions reaches biting
range and lands a bite on its own. So the mechanics are not the blocker; a
policy that walks at what it sees is sufficient, and the question is only
whether the learning finds it.

## Runs

| run | level | agent config | wall | sim | result |
|---|---|---|---|---|---|
| `tests` #1 | 25×25 dense maze | MuZero-only policy, no value rescale | 12 min | ~11 min | 0 kills, 0 bites; value NaN |
| #2 | dense maze | + value rescaling, + Reanalyze | 10 min | ~9 min | 0 kills, 0 bites; `clarkClosest` 9.0 m |
| #3 (post-NaN-fix) | dense maze | 9-bin support fixed, gradient guard | 659 s | 300 s | 0 kills, 0 bites, 27 roars, 52 lunges; `clarkSees` 78.2%, `clarkClosest` 9.0 m; mix `fwdL` .40 `turnL` .30 `lunge` .07 |
| #4 | dense maze | + still-penalty, policy smoothing, wider search (`turbo`) | 45 min | ~20 min | 0 kills, 0 bites; mix dominated by `idle`/`turnL` |
| **#5** | **open floor plate** | + AWR actor term, bite stagger, sprint stamina, `simSpeed 2` | 18 min (stopped) | 13 min | **3 kills, 9 bites, first bite at t = 347 s**; cold replay closes to 2.0 m |
| #6 | open floor plate | + potential shaping, 30 Hz decisions | 5 min (stopped) | 3 min | 0 bites, `idle` 59% — the freeze attractor, see below |
| #7 | open floor plate | + motion prior (`roam`) | in progress | — | — |

Runs #1–#4 all share the same mistake: they are on a **dense randomised maze**
(40–60% walls). That makes the task require maze navigation before any hunting
can happen, the approach reward is almost never reachable by exploration, and
the value function stays flat — the monster's best available policy really is to
stand still. Everything after #4 is on the open floor plate.

Two changes were made in response, both in the level rather than in the agent,
because the diagnostics said the agent was not the thing that was broken:

1. **The level is now the film's open floor plate** (`MAP.layout: 'backrooms'`):
   18% walls, partitions with doorways, column clusters. `?layout=maze` restores
   the old generator for comparison. Immediately after this change, random-action
   exploration reached biting range and bit (see the balance probe above) — it had
   never once done that in the maze.
2. **Injury staggered the victim** (`biteSlowFactor: 0.5` for 1.1 s) and
   **stamina** was added to sprinting, so a landed bite tends to lead to a kill
   rather than to one lucky hit, and running buys distance rather than immunity.
   Both are gameplay decisions that also happen to be the difference between a
   reward that fires and a reward that never fires.

An auxiliary advantage-weighted actor term was also added
(`wAdvantage: 0.35`, see `docs/EFFICIENTZERO.md`) because while the dynamics net
is still bad, search visits are near uniform (measured entropy 1.9–2.5 of a
possible 2.56) and pure search-imitation has nothing to teach the policy.

## Run #5 — first unaided kill

Open floor plate, `turbo` profile, `simSpeed 2`, training budget 6 ms per brain
per frame, scripted fleeing player stand-in, autosave every 8 wall-minutes,
`public/brain-live-*.json`. 1.3 s wall per sim-second, ~24 frames/s, ~29
gradient steps per brain per sim-second.

| sim time | wall | gradient steps (clark) | bites | kills | roars |
|---|---|---|---|---|---|
| 60 s | 1.2 min | 1 763 | 0 | 0 | 9 |
| 180 s | 3.7 min | 5 364 | 0 | 0 | 24 |
| 300 s | 6.2 min | 8 965 | 0 | 0 | 39 |
| **360 s** | 7.6 min | 10 766 | **3** | **1** | 47 |
| 420 s | 8.9 min | 12 567 | **5** | **1** | 52 |
| 600 s | 13.4 min | 17 998 | **9** | **3** | 75 |
| 780 s | 18.0 min | 23 401 | 9 | 3 | 98 |

The first bite and the first kill both arrive between 300 s and 360 s of
simulation, from a policy that began as random actions. Nothing in the reward
stream changed at that moment and no script touched the monster: what changed is
that the value of being near prey finally propagated into the search
distribution and then into the policy.

Two sanity checks on that claim, both from the same checkpoint:

* the action mix moved from `idle`-dominated to `fwd 23% / roar 23% / lunge 10%`
  (the roar is not decoration: roaring dazes survivors inside 9.5 m, which slows
  them enough to be caught, so it earns its place through `roarUseful` only when
  it actually dazes somebody);
* replaying the checkpoint cold (fresh world, no learning, 120 s) closes to
  **2.0 m**, against 9.0 m in every dense-maze run — the approach behaviour is in
  the weights, not in the episode's luck.

Open questions this run does not answer, recorded so they are not forgotten:

* the *survivor* brain collapsed to `strafeL 63%` by t = 420 s. That is reward
  hacking of the `flee` term (any tick that increases distance pays, and a
  circle straightens out the geometry), and it means prey quality degrades over a
  long run. The principled fix is potential-based shaping
  (`r = γΦ(s') − Φ(s)` with `Φ = flee · d` while Clark is perceived); the
  pragmatic one is to keep bot training off and let them walk away with a
  hand-written fleer. Both are one-line changes, neither is in this run.
* kills are still sparse (1 in 7 minutes of sim). Whether they accelerate with
  more updates, or whether they need the roar-lunge timing to be represented
  better in search, is not yet measured.
* Clark has never yet been killed-and-respawned mid-hunt in a way that changes
  his behaviour: the episode restarts him at the far end of the plate each time,
  which may be all the "forgetting" a hunter needs.

## Run #6 — the freeze attractor (stopped early)

Rebuilt after run #5 with potential-based approach shaping (closing, facing and
a new bite-range potential, all as differences of potentials so that oscillating
pays exactly nothing), a stronger bite/kill reward, and 30 Hz decisions instead
of 15 Hz so that a committed action cannot carry him past his target at bite
range. Result: `idle 54% → 58% → 59%` of his actions by t = 180 s, zero bites.

The diagnosis is the interesting part, because it is not really about the
shaping. Once the floor has been mapped, novelty is exhausted, so with **no prey
in perception** there is nothing positive to do — while moving risks a wall bump
(`-0.002`) and freezing costs only the stillness penalty (`-0.006`/tick). The
policy froze; a frozen policy never meets prey, so it never learns anything
else. The attractor is self-reinforcing, and it is not a bug in any one term: it
is what the *sum* of the terms says when every nearby reward is a penalty.

Fix: motion is now weakly positive everywhere — `roam: 0.004` per metre
travelled, capped at `0.01` per tick, with the stillness penalty reduced to
`-0.004`. Standing still is now strictly worse than walking at every moment
(`-0.24/s` versus roughly `+0.01/s`), while remaining useless as a substitute
for hunting (a bite pays 1.2, a kill 8.0, crossing into bite range 1.2). It is a
locomotion prior like the stillness penalty: it says "keep moving", never "move
towards something".

## Run #7 — motion prior, potential shaping, 30 Hz decisions

Training as this is written (`--minutes 75 --profile turbo --sim-speed 2
--budget 6`, autosaving to `public/brain3-*.json`). Its table will be appended
here when it finishes or is stopped, whatever it shows.

To reproduce or extend:

```
node tools/train.mjs --minutes 75 --profile turbo --sim-speed 2 --budget 6 \
    --report 60 --autosave 6 --out public/brain3
node tools/diagnose.mjs 180 --learn --checkpoint public/brain3-3600s.json
node tools/map-preview.mjs                    # what the level looks like
node tools/balance-probe.mjs                  # what is mechanically reachable
```

The in-browser equivalent is `src/ai/lab.html`, which trains in a Web Worker and
saves the same checkpoint format.

## Determinism

Level generation, spawns, Dirichlet root noise and action sampling all come from
seeded `mulberry32` streams, so a run is reproducible given `--seed`. The
scheduler is not: sleep/wake order and the time-based training budget make wall
clock non-deterministic, which is why every number in this file is reported with
the sim time it was measured at.
