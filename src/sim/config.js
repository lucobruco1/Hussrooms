/**
 * Hussrooms - central tunables.
 *
 * Everything that a researcher might want to poke at lives here: map size,
 * physics, reward shaping, and EfficientZero hyper-parameters. The in-game
 * "Lab" panel edits a live copy of this object, so changes are visible
 * immediately without a rebuild.
 */

export const MAP = {
  // Grid of cells; cellSize is metres. 25 x 2.6m = 65m across: a small, dense
  // Backrooms floor. Size matters for learning, not just for looks - in an
  // earlier 99m-wide version, agents wandered for minutes without ever meeting,
  // so the reward signal that teaches hunting almost never fired.
  gx: 25,
  gz: 25,
  cellSize: 2.6,
  wallHeight: 3.1,
  seed: 2019, // Kane Pixels' original short dropped in 2019.
  openness: 0.42, // fraction of maze walls knocked out to make it "backroomsy"
  // Layout. 'backrooms' = open floor plate, partition walls with doorways and
  // column clusters (what the film actually looks like, and what a learner can
  // actually navigate: the prey is usually visible across the room).
  // 'maze' = the original randomised-DFS maze plus wall removal, kept for
  // comparison runs via ?layout=maze.
  layout: 'backrooms',
  partitions: 15, // partition walls, each with at least one doorway
  pillars: 20, // column clusters (1x1 and 2x2)
  props: 55, // ottomans, tables, crates ("Cap'n Clark's Ottoman Empire")
  lightsEvery: 2, // ceiling light panel every N cells
};

export const PHYS = {
  dt: 1 / 60,
  playerRadius: 0.34,
  agentRadius: 0.34,
  clarkRadius: 0.62,
  walkSpeed: 3.15, // bots + player walk
  sprintSpeed: 5.15, // player sprint — faster than Clark, but not for long
  // Sprint stamina (players only; Clark is a Still Life and never tires). It
  // exists for the horror beat as much as the gameplay: he cannot outrun you,
  // he just never has to stop, so running buys distance rather than safety.
  // It also matters for learning - a prey that can sprint forever is prey the
  // monster can never catch, and a reward that never fires teaches nothing.
  staminaDrain: 0.30, // per second of sprinting (3.3 s of full sprint)
  staminaRecover: 0.18, // per second while not sprinting
  staminaMinStart: 0.25, // cannot start a sprint below this
  // A bite staggers the victim (mauled), which is what turns a single lucky
  // bite into a kill: without it the first bite is always the last one.
  biteSlowFactor: 0.5,
  biteSlowTime: 1.1,
  clarkSpeed: 4.0, // faster than a walking survivor (3.15), slower than a
  //                  sprinting player (5.15): he can run down bots, you can
  //                  outrun him if you commit to running
  clarkLungeSpeed: 7.4, // ...and he can throw himself forward
  lungeDuration: 0.55,
  lungeCooldown: 3.0,
  turnRate: 7.5, // rad/s, how fast bodies pivot to their move direction
  biteRange: 1.15,
  biteCooldown: 1.1,
  biteDamage: 38,
  roarRadius: 9.5,
  roarDazeTime: 1.15,
  roarCooldown: 6.0,
  respawnDelay: 2.5,
  botCount: 4,
  playerHealth: 100,
  botHealth: 100,
  clarkHealth: 260,
  // Player fire is deliberately weak; this is not an action game.
  playerDamage: 6,
  playerRange: 11,
  turnSpeed: 3.1, // rad/s for survivor bots (~178 deg/s)
  clarkTurnSpeed: 2.3, // he is enormous and stiff
  sightRange: 17.5, // agents' modelled vision distance
  sightCone: Math.cos((105 * Math.PI) / 180 / 2), // half-angle cosine
  hearingRange: 13.0,
};

/**
 * Discrete action space.
 *
 * Body-relative, exactly like an FPS player: the eight movement actions are
 * interpreted in the agent's own frame (the same frame the egocentric
 * observation is rendered in), movement never rotates the body, and turning is
 * its own explicit decision. That stationarity is what makes "the prey is in my
 * right-hand channels, so strafe right" a learnable mapping - an earlier
 * world-oriented action set made the correct policy change as the agent
 * rotated, and every brain collapsed onto walking into a wall.
 *
 * Survivor bots use indices 0..TURN_R (11 actions).
 * Cap'n Clark additionally gets lunge (11) and roar (12) -> 13 actions.
 */
export const A = {
  idle: 0,
  fwd: 1,
  back: 2,
  strafeL: 3,
  strafeR: 4,
  fwdL: 5,
  fwdR: 6,
  backL: 7,
  backR: 8,
  turnL: 9,
  turnR: 10,
  lunge: 11,
  roar: 12,
};

export const ACTIONS = [
  { name: 'idle', dx: 0, dz: 0, turn: 0 },
  { name: 'fwd', dx: 0, dz: 1, turn: 0 },
  { name: 'back', dx: 0, dz: -1, turn: 0 },
  { name: 'strafeL', dx: -1, dz: 0, turn: 0 },
  { name: 'strafeR', dx: 1, dz: 0, turn: 0 },
  { name: 'fwdL', dx: -0.7071, dz: 0.7071, turn: 0 },
  { name: 'fwdR', dx: 0.7071, dz: 0.7071, turn: 0 },
  { name: 'backL', dx: -0.7071, dz: -0.7071, turn: 0 },
  { name: 'backR', dx: 0.7071, dz: -0.7071, turn: 0 },
  { name: 'turnL', dx: 0, dz: 0, turn: 1 },
  { name: 'turnR', dx: 0, dz: 0, turn: -1 },
  { name: 'lunge', dx: 0, dz: 1, turn: 0 }, // Clark only: forward burst
  { name: 'roar', dx: 0, dz: 0, turn: 0 }, // Clark only: daze nearby survivors
];

/** Action counts per agent type. */
export const SURVIVOR_ACTIONS = 11; // idle .. turnR
export const CLARK_ACTIONS = ACTIONS.length; // 13

/**
 * Reward configuration.
 *
 * DESIGN NOTE (important, and deliberate): nothing in here or anywhere else in
 * the codebase tells Cap'n Clark to chase anybody. There is no "if distance to
 * player < X then steer at player" anywhere. He emits actions, gets numbers
 * back, and EfficientZero does the rest. The only hand-authored signal is what
 * you see below: how those numbers are computed.
 *
 * `closing` rewards him for shrinking the distance to a *detected* target, i.e.
 * one currently inside his vision cone or hearing radius. It is a generic
 * "engagement" term - it fires for any target, it does not aim him, and it does
 * nothing when he has lost the scent. The kill bonus is what makes chasing
 * worth it; closing just bridges the sparse-reward gap.
 *
 * Set preset to 'sparse' in the Lab panel to delete the shaping entirely and
 * watch him figure it out from kills alone.
 */
export const REWARDS = {
  preset: 'shaped', // 'shaped' | 'sparse' | 'hunt-only'
  // --- common ---
  step: -0.001, // per tick, the cost of existing (~-0.06/s)
  wallBump: -0.002, // per tick spent actually blocked by geometry
  // Motion shaping. Without it, a flat value function makes "spin in place"
  // strictly optimal (it costs nothing and risks nothing), and greedy search
  // locks onto it - an attractor that took two long training runs to diagnose.
  // This is a locomotion prior, not a hunting rule: it says "do something",
  // never "do what".
  still: -0.004, // per tick with essentially no displacement
  // Motion prior. Novelty runs out once the floor has been mapped, and after
  // that "stand still" was strictly optimal whenever no prey was in perception:
  // moving risked a wall bump and earned nothing, so the policy froze (measured:
  // 59% idle, which is a dead policy). A small reward per metre travelled makes
  // roaming weakly positive everywhere, which is what a predator needs to be.
  // It cannot be farmed into a substitute for hunting: a bite is 1.2 and a kill
  // is 8.0, against ~0.013/s for walking.
  roam: 0.004, // per metre of ground actually covered (capped per tick)
  roamCap: 0.01, // max roam reward per tick
  // --- predator ---
  kill: 8.0, // killed a player or a bot (dominates everything else)
  bite: 1.2, // landed a bite (killed or not) - a bite now very often leads to a
  //             kill thanks to the stagger, so this is the reward that
  //             actually bootstraps hunting
  // Approach shaping, applied as *potentials* rather than per-tick payments:
  // the reward for a step is Phi(after) - Phi(before), so oscillating or
  // circling nets exactly zero while approaching nets the full difference.
  // Per-tick "you closed 2 cm" payments are farmable by any periodic motion,
  // and a farmed shaping term is what a policy learns instead of hunting.
  closing: 0.45, // Phi_close = -closing * distance  (while the target is perceived)
  orient: 0.4, // Phi_face = orient * cos(bearing to target)
  near: 1.2, // Phi_near = near * (1 - min(d,3)/3): crossing into bite range pays
  lungeHit: 0.3, // connected with a lunge
  roarUseful: 0.05, // roar that dazed at least one agent
  clarkNovelty: 0.15, // first visit to a cell: keeps him searching the floor
  // --- survivors (bots) ---
  survive: 0.002, // per tick alive (~0.12/s): survival matters, but must not
  //                 swamp the signals that teach evasion
  caught: -5.0, // was killed by Cap'n Clark
  flee: 0.06, // per metre of distance opened while Clark is perceived
  novelty: 0.12, // first time entering a cell
  flock: 0.002, // near another living survivor while Clark is not visible
  dazed: -0.1, // got caught in a roar
};

/** Observation geometry fed to the representation network. */
export const OBS = {
  crop: 12, // 12 x 12 egocentric patch
  cell: 1.35, // metres per patch cell (16.2 m across)
  pool: 2, // non-overlapping pooling before the representation net (CPU budget)
  channels: 6, // walls/props, bodies, own-scent, rival-scent, noise, unexplored
  scalars: 13,
  egocentric: true, // rotate patch into the agent's frame ("first person")
  scentDecay: 0.995, // per tick
  noiseDecay: 0.96,
};

/**
 * EfficientZero hyper-parameters (Ye et al., 2021, arXiv:2111.00210).
 * Sizes are scaled way down from the paper - this has to train inside a
 * browser tab on the CPU while a game runs at 60 fps.
 */
export const EZ = {
  latentDim: 96,
  hiddenDim: 128,
  channelDim: 16,
  projDim: 32,
  unroll: 5, // K: dynamics steps unrolled during training
  sims: 14, // MCTS simulations per decision
  maxDepth: 5,
  c1: 1.25, // PUCT exploration constant
  c2: 19652,
  discount: 0.995, // ~7 s value half-life at the 10 Hz decision rate
  rootNoise: 0.22, // Dirichlet(alpha=0.3) at the root during collection
  policySmoothing: 0.25, // mix a uniform distribution into the policy target
  // Collection temperature stays high on purpose. A fast anneal in a continuous
  // world locks the policy onto whatever the value head believed in the first
  // few seconds, and then nothing is ever discovered again.
  temperature: 1.0,
  temperatureFinal: 0.85,
  tempSteps: 600,
  exploreEps: 0.03, // fraction of actions taken uniformly at random (never 0)
  warmupActions: 220, // pure random actions per brain before MCTS takes over
  batch: 24,
  lr: 0.0016,
  weightDecay: 0.0001,
  gradClip: 6.0,
  // loss weights
  // Auxiliary advantage-weighted policy term (not in the paper).
  //
  // MuZero-family agents learn their policy purely by imitating search. That is
  // optimal in the compute regime of the paper, but this prototype has two CPU
  // cores: while the dynamics net is still bad, search visits are close to
  // uniform and there is nothing for the policy to imitate, so the agent has no
  // route to "walk at the thing I can see". This term adds a small, standard
  // off-policy actor loss on the actions actually taken: weight each sampled
  // action by exp(advantage / advBeta), where the advantage is the K-step
  // bootstrapped return minus the value the search estimated at collection
  // time. It nudges the policy towards actions that paid off. Everything else
  // (search, value, reward, consistency, Reanalyze) is unchanged.
  wAdvantage: 0.35,
  advBeta: 1.5,
  advClamp: 4.0,

  wValue: 1.0,
  wReward: 0.6,
  wPolicy: 1.0,
  wConsistency: 0.55, // self-supervised latent consistency (SimSiam-style)
  wL2: 0.0004,
  targetSyncEvery: 160, // hard target network refresh (updates)
  replayCapacity: 9000, // transitions per brain
  reanalyzeRatio: 0.7, // fraction of value/policy targets refreshed by MCTS
  priorityAlpha: 0.6,
  priorityBeta: 0.4,
  valueScale: 8.0, // value head = tanh * this
  rewardScale: 4.0, // raw reward / this -> categorical support [-1, 1]
  useValuePrefix: true, // EfficientZero's end-to-end value prefix loss
  useConsistency: true,
  useReanalyze: true,
};

/**
 * Two cost profiles for the same algorithm.
 *
 *   live  - what the default game runs: small nets, K=3 unroll, batch 8, so a
 *           60 fps frame can hold both the simulation and the gradient steps.
 *   turbo - "Sparring mode": bigger batch and longer unroll, meant to be run
 *           with the renderer throttled or paused (in-browser, or fully
 *           headless via tools/train.mjs).
 *
 * Identical maths, different CPU budget - the honest way to put an
 * EfficientZero-class learner inside a playable game.
 */
export const EZ_PROFILES = {
  live: {
    profile: 'live',
    latentDim: 64,
    hiddenDim: 96,
    projDim: 24,
    unroll: 3,
    batch: 8,
    sims: 10,
    maxDepth: 5,
    decisionInterval: 6, // 10 Hz
    lr: 0.0008,
    warmupActions: 160,
    consistEveryK: 3,
    replayCapacity: 6000,
    targetSyncEvery: 80,
  },
  turbo: {
    profile: 'turbo',
    latentDim: 80,
    hiddenDim: 112,
    projDim: 28,
    unroll: 3,
    batch: 12,
    sims: 24,
    maxDepth: 8,
    decisionInterval: 2, // 30 Hz: at bite range a committed action can carry him
    //                      clear past his target, and sharpening close-quarters
    //                      control is worth more than the extra search cost
    lr: 0.0012,
    warmupActions: 220,
    consistEveryK: 2,
    replayCapacity: 9000,
    targetSyncEvery: 120,
  },
};

export function ezCfg(profile = 'live', overrides = {}) {
  return { ...EZ, ...(EZ_PROFILES[profile] || EZ_PROFILES.live), ...overrides };
}

/** Per-brain training/acting plan. Bots may share one brain ("hive mind"). */
export const LEARN = {
  shareBotBrains: true,
  decisionInterval: 6, // ticks between MCTS decisions (6/60 s = 10 Hz)
  trainBudgetMs: 7.5, // cpu time per rendered frame for gradient updates
  opsPerFrame: 4, // max gradient steps per brain per frame
  simSpeed: 1, // sim steps per rendered frame
  learn: true,
};
