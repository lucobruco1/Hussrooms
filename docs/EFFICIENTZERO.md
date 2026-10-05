# EfficientZero, in JavaScript, in this repository

Reference: Ye, Liu, Kurutach, Abbeel, Gao — *Mastering Atari Games with Limited
Data* (arXiv:2111.00210). EfficientZero is MuZero plus three ideas that make it
sample-efficient enough to work in the 100k-frame regime: a self-supervised
latent consistency loss, end-to-end value-prefix prediction, and a
model-based off-policy correction ("Reanalyze") that refreshes stale value and
policy targets using the *current* model.

Everything below is implemented from scratch in this repo: no ML library, no
Python, no WebAssembly. The autodiff engine (`src/ai/core/tape.js`) is
reverse-mode over Float64 tensors and was verified by finite differences
(`node tools/test-grad.mjs`, 7/7).

## Where each piece lives

| EfficientZero | here | notes |
|---|---|---|
| Representation function | `Networks.represent` / `representBatch` | MLP over the pooled 216-float crop + 13 scalars → latent |
| Dynamics function | `Networks.dynamics` | residual: `h' = 0.95·(h + 0.5·LN(dyn([h, a])))`, with a reward head on the edge |
| Prediction function | `Networks.predict` | policy logits + 9-bin categorical value (mean & spread rescaled) |
| MCTS with the learned model | `src/ai/mcts.js` | PUCT, one edge expanded per simulation, min-max normalised Q, Dirichlet root noise 0.22 (α = 0.3) |
| Value prefix | `trainOnce()` loss 3 | root trained on the K-step bootstrapped target, every imagined step trained on its own prefix target |
| Self-supervised consistency | loss 4 | SimSiam-style negative cosine between the imagined latent at step k and the target net's representation of the real observation that followed |
| Reanalyze | `EZBrain.reanalyze` | every few updates, re-runs search on sampled windows with the target net and rewrites their value/policy targets |
| Prioritised replay + IS | `trainOnce()` sampling | priority `(err + 1e-4)^0.6`, importance weights `(p/maxP)^-0.4`, normalised |
| Target network | `syncTarget` | hard refresh every 80–120 updates; also carries the value rescaler |
| Multi-step policy targets | loss 1 | the search distribution is imitated at t and at every imagined step t+k, weighted `0.5·w/k` |

Not implemented, and why: the paper's ResNet-v2 encoder and its image-based
Atari input are replaced by the hand-built egocentric crop encoder (this is a
prototype with no GPU); Gaussian/unimix heads are replaced by a plain
categorical head; and the batching/vectorised environments that make the paper's
wall-clock numbers are impossible on two CPU cores. What remains is the
algorithm.

## Deviations worth knowing about

**1. An auxiliary advantage-weighted actor term** (`wAdvantage: 0.35`,
`advBeta: 1.5`). MuZero's policy is trained purely by imitating search. That is
correct when search is strong; here, while the dynamics net is still bad, search
visits are nearly uniform (measured search entropy ≈ 1.9–2.5 out of 2.56 max)
and there is nothing to imitate — the policy has no route to "walk at the thing
I can see". The auxiliary term is a standard off-policy actor update: weight
each sampled action by `exp(A/β)` with `A` the K-step bootstrapped return minus
the value search reported at collection time, and take a weighted
cross-entropy step on the action that was taken. Weights are detached numbers,
so it adds no new autodiff machinery. It cannot invent behaviour that the reward
does not pay for; it only propagates reward into the policy faster than search
imitation does.

**2. Value rescaling.** The value head is `tanh·spread + mean`, with
`mean`/`spread` EMAs (0.995) of the batch statistics of the bootstrapped
targets, clamped to ±40 / [1, 40], mirrored to the target net; rescaled targets
are clamped to ±6. Without this the head saturates and then self-reinforces
through MCTS bootstrapping — a failure mode that is easy to mistake for a NaN
bug, and one this repo hit.

**3. Two profiles.** `live` (10 Hz decisions, 10 simulations, latent 64) fits
the browser frame budget; `turbo` (15 Hz, 24 simulations, max depth 8, latent
80, lr 1.2e-3) is for headless runs. Same code, different numbers.

**4. Float64 math, Float32 checkpoints.** The tape always computes in Float64
(the operators are cheap relative to the net forwards and avoid the slow drift
that killed earlier runs); serialisation stores Float32 (`f64ToBase64`), so a
round trip is lossy at ~1e-7.

## Bugs that cost the most time, in case they happen again

* **`categoricalToScalar` with the wrong support.** The reward head had 9 bins
  and the helper defaulted to a local 5-bin support, so `bins[i]` was
  `undefined`, every edge reward became `NaN`, and the first `backup` spread it
  through the entire value function. The fix was one import
  (`SUPPORT` is now the single source of truth in `networks.js`) and a throw on
  any bin-count mismatch. Symptom to remember: every value in the replay is
  `NaN` from the first frames, and `valueTargets`, `nu` and value/prefix losses
  all inherit it.
* **Non-finite gradient guard.** `trainOnce()` now checks the loss and gradient
  norms and skips the optimiser step when they are not finite, counting the
  event in `brain.skippedUpdates` (shown in the lab panel). That is the safety
  net that replaces the ad-hoc NaN bisecting.
* **On-policy drift in a continuous world.** With temperature annealed hard,
  the policy locks onto whatever the value head believed in the first seconds.
  Collection temperature now floors at 0.85, and the collected policy target is
  mixed with 25% uniform (`policySmoothing`).
