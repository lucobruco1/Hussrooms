/**
 * The floor: a small Backrooms level.
 *
 * Two generators. The default ('backrooms') is the film's floor plate: a long
 * open expanse of yellow rooms broken up by partition walls with proper
 * doorways and clusters of columns, plus furniture. The alternative ('maze') is
 * the original randomised-DFS maze with a chunk of walls knocked out.
 *
 * The default matters for the AI more than for looks. In a dense maze, the
 * monster has to solve maze navigation before it ever gets to bite anybody, and
 * the reward for closing on prey is mostly unreachable; on an open plate, prey
 * is usually visible across the room and "walk at it" is a policy that exists.
 * Connectivity is verified by flood fill in both generators, so no agent can
 * ever be sealed away.
 */

import { MAP, PHYS } from './config.js';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Grid {
  constructor(seed = MAP.seed) {
    this.gx = MAP.gx;
    this.gz = MAP.gz;
    this.cs = MAP.cellSize;
    this.wall = new Uint8Array(this.gx * this.gz); // 1 = wall
    this.props = []; // {x,z,type,r} decorative, movement-blocking
    this.propAt = new Int16Array(this.gx * this.gz).fill(-1);
    this.generate(seed);
  }

  idx(x, z) {
    return x * this.gz + z;
  }

  inBounds(x, z) {
    return x >= 0 && z >= 0 && x < this.gx && z < this.gz;
  }

  isWall(x, z) {
    if (x < 0 || z < 0 || x >= this.gx || z >= this.gz) return true;
    return this.wall[x * this.gz + z] === 1;
  }

  /** World-space metres -> grid cell */
  cellOf(wx, wz) {
    return [Math.floor(wx / this.cs), Math.floor(wz / this.cs)];
  }

  generate(seed) {
    const rnd = mulberry32(seed);
    const { gx, gz } = this;
    // 1. Border walls.
    for (let x = 0; x < gx; x++) {
      this.wall[this.idx(x, 0)] = 1;
      this.wall[this.idx(x, gz - 1)] = 1;
    }
    for (let z = 0; z < gz; z++) {
      this.wall[this.idx(0, z)] = 1;
      this.wall[this.idx(gx - 1, z)] = 1;
    }
    if ((MAP.layout || 'backrooms') === 'backrooms') {
      this.generateBackrooms(rnd);
      this.ensureConnected(rnd);
      this.buildProps(rnd);
      this.buildSpawnPoints(rnd);
      return;
    }
    this.generateMaze(rnd);
    this.buildProps(rnd);
    this.buildSpawnPoints(rnd);
  }

  /**
   * Open floor plate: partition walls with doorways, then columns. Both are
   * placed on the cell lattice so the level reads as built rather than noisy.
   */
  generateBackrooms(rnd) {
    const { gx, gz } = this;
    const put = (x, z) => {
      if (x > 0 && z > 0 && x < gx - 1 && z < gz - 1) this.wall[this.idx(x, z)] = 1;
    };
    for (let i = 0; i < MAP.partitions; i++) {
      const horizontal = rnd() < 0.5;
      const len = 4 + Math.floor(rnd() * 5);
      const cellX = 1 + Math.floor(rnd() * Math.max(1, gx - 2 - (horizontal ? len : 1)));
      const cellZ = 1 + Math.floor(rnd() * Math.max(1, gz - 2 - (horizontal ? 1 : len)));
      const cells = [];
      for (let k = 0; k < len; k++) cells.push(horizontal ? [cellX + k, cellZ] : [cellX, cellZ + k]);
      for (const [x, z] of cells) put(x, z);
      // Doorways: at least one, never at the extremities, sometimes widened to
      // two cells so a chase can flow through without a dead stop.
      const doors = 1 + (rnd() < 0.5 ? 1 : 0);
      for (let d = 0; d < doors; d++) {
        const k = 1 + Math.floor(rnd() * Math.max(1, len - 2));
        const [dx0, dz0] = cells[k];
        this.wall[this.idx(dx0, dz0)] = 0;
        if (rnd() < 0.45 && k + 1 < len) {
          const [dx1, dz1] = cells[k + 1];
          this.wall[this.idx(dx1, dz1)] = 0;
        }
      }
    }
    for (let i = 0; i < MAP.pillars; i++) {
      const size = rnd() < 0.3 ? 2 : 1;
      const x = 2 + Math.floor(rnd() * Math.max(1, gx - 4));
      const z = 2 + Math.floor(rnd() * Math.max(1, gz - 4));
      for (let dx = 0; dx < size; dx++) for (let dz = 0; dz < size; dz++) put(x + dx, z + dz);
    }
  }

  /** Knock out walls until every floor cell is reachable. */
  ensureConnected(rnd) {
    let guard = 0;
    while (!this.floodReachesAll() && guard++ < 6000) {
      const x = 1 + Math.floor(rnd() * (this.gx - 2));
      const z = 1 + Math.floor(rnd() * (this.gz - 2));
      if (!this.isWall(x, z)) continue;
      this.wall[this.idx(x, z)] = 0;
      if (!this.floodReachesAll()) this.wall[this.idx(x, z)] = 1;
    }
  }

  generateMaze(rnd) {
    const { gx, gz } = this;
    // Randomised DFS maze over 2-cell-wide rooms.
    const RW = 2;
    const visited = new Uint8Array(gx * gz);
    const stack = [];
    const roomX = 1;
    const roomZ = 1;
    const startX = roomX + Math.floor(rnd() * Math.floor((gx - 2) / RW)) * RW;
    const startZ = roomZ + Math.floor(rnd() * Math.floor((gz - 2) / RW)) * RW;
    stack.push([startX, startZ]);
    visited[this.idx(startX, startZ)] = 1;
    while (stack.length) {
      const [cx, cz] = stack[stack.length - 1];
      const dirs = [
        [RW, 0],
        [-RW, 0],
        [0, RW],
        [0, -RW],
      ];
      // shuffle
      for (let i = dirs.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        [dirs[i], dirs[j]] = [dirs[j], dirs[i]];
      }
      let advanced = false;
      for (const [dx, dz] of dirs) {
        const nx = cx + dx;
        const nz = cz + dz;
        if (nx < 1 || nz < 1 || nx >= gx - 1 || nz >= gz - 1) continue;
        if (visited[this.idx(nx, nz)]) continue;
        // carve the room block between
        for (let ax = Math.min(cx, nx); ax <= Math.max(cx, nx); ax++) {
          for (let az = Math.min(cz, nz); az <= Math.max(cz, nz); az++) visited[this.idx(ax, az)] = 1;
        }
        // wall between the two rooms (a 1-cell-thick divider)
        const wx = (cx + nx) / 2;
        const wz = (cz + nz) / 2;
        if (dx !== 0) {
          for (let az = cz; az < cz + RW; az++) this.wall[this.idx(wx, az)] = 1;
        } else {
          for (let ax = cx; ax < cx + RW; ax++) this.wall[this.idx(ax, wz)] = 1;
        }
        stack.push([nx, nz]);
        advanced = true;
        break;
      }
      if (!advanced) stack.pop();
    }
    // 3. Knock holes. Each removal is only kept if the floor stays connected.
    const wallList = [];
    for (let x = 1; x < gx - 1; x++)
      for (let z = 1; z < gz - 1; z++) if (this.wall[this.idx(x, z)]) wallList.push([x, z]);
    for (let i = wallList.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [wallList[i], wallList[j]] = [wallList[j], wallList[i]];
    }
    const target = Math.floor(wallList.length * MAP.openness);
    let removed = 0;
    for (const [x, z] of wallList) {
      if (removed >= target) break;
      this.wall[this.idx(x, z)] = 0;
      if (this.floodReachesAll()) removed++;
      else this.wall[this.idx(x, z)] = 1;
    }
    // 4. A few pillars for cover (never disconnecting: isolated 1x1 blocks).
    for (let i = 0; i < Math.floor(gx * gz * 0.012); i++) {
      const x = 1 + Math.floor(rnd() * (gx - 2));
      const z = 1 + Math.floor(rnd() * (gz - 2));
      if (this.isWall(x, z)) continue;
      let wallNeighbours = 0;
      for (let dx = -1; dx <= 1; dx++)
        for (let dz = -1; dz <= 1; dz++) if (this.isWall(x + dx, z + dz)) wallNeighbours++;
      if (wallNeighbours <= 1) this.wall[this.idx(x, z)] = 1;
    }
    this.buildProps(rnd);
    this.buildSpawnPoints(rnd);
  }

  buildProps(rnd) {
    const types = ['ottoman', 'crate', 'table', 'chair', 'sofa'];
    let tries = 0;
    while (this.props.length < MAP.props && tries < MAP.props * 40) {
      tries++;
      const x = 1 + Math.floor(rnd() * (this.gx - 2));
      const z = 1 + Math.floor(rnd() * (this.gz - 2));
      if (this.isWall(x, z)) continue;
      if (this.propAt[this.idx(x, z)] >= 0) continue;
      // keep a little open space around props so they read as furniture
      const type = types[Math.floor(rnd() * types.length)];
      const r = type === 'table' ? 0.75 : type === 'sofa' ? 0.8 : 0.45;
      const jitter = 0.7;
      const p = {
        x: (x + 0.5) * this.cs + (rnd() - 0.5) * jitter,
        z: (z + 0.5) * this.cs + (rnd() - 0.5) * jitter,
        type,
        r,
        rot: rnd() * Math.PI * 2,
        blocksMove: type !== 'chair',
        blocksSight: type === 'sofa' || type === 'table',
      };
      this.propAt[this.idx(x, z)] = this.props.length;
      this.props.push(p);
    }
  }

  buildSpawnPoints(rnd) {
    // "Openness" of a cell = how many of the 8 neighbours are walkable. The
    // maze proper has 2x2 rooms, so strict 8/8 clearance never happens; we rank
    // cells by openness and keep the top slice as spawn candidates instead.
    const scored = [];
    for (let x = 1; x < this.gx - 1; x++) {
      for (let z = 1; z < this.gz - 1; z++) {
        if (this.isWall(x, z)) continue;
        let open = 0;
        for (let dx = -1; dx <= 1; dx++)
          for (let dz = -1; dz <= 1; dz++) if (!this.isWall(x + dx, z + dz)) open++;
        scored.push([x, z, open + rnd() * 0.5]);
      }
    }
    scored.sort((a, b) => b[2] - a[2]);
    const keep = Math.max(24, Math.floor(scored.length * 0.35));
    this.spawnCells = scored.slice(0, keep).map(([x, z]) => [x, z]);
    if (!this.spawnCells.length) this.spawnCells = [[1, 1]];
    this.spawnPoints = this.spawnCells.map(([x, z]) => ({
      x: (x + 0.5) * this.cs,
      z: (z + 0.5) * this.cs,
    }));
    // spawn points must not sit inside furniture
    this.spawnPoints = this.spawnPoints.filter((p) => {
      const [cx, cz] = this.cellOf(p.x, p.z);
      if (!this.inBounds(cx, cz)) return false;
      const pi = this.propAt[this.idx(cx, cz)];
      return !(pi >= 0 && this.props[pi].blocksMove);
    });
    if (!this.spawnPoints.length) {
      this.spawnPoints = [{ x: 1.5 * this.cs, z: 1.5 * this.cs }];
    }
  }

  /** Flood fill from the first floor cell; returns true if it covers every floor cell. */
  floodReachesAll() {
    const { gx, gz } = this;
    const seen = new Uint8Array(gx * gz);
    let total = 0;
    let start = -1;
    for (let i = 0; i < gx * gz; i++) {
      if (!this.wall[i]) {
        total++;
        if (start < 0) start = i;
      }
    }
    if (start < 0) return true;
    const stack = [start];
    seen[start] = 1;
    let reached = 1;
    while (stack.length) {
      const i = stack.pop();
      const x = (i / gz) | 0;
      const z = i % gz;
      const nb = [
        [x + 1, z],
        [x - 1, z],
        [x, z + 1],
        [x, z - 1],
      ];
      for (const [nx, nz] of nb) {
        if (!this.inBounds(nx, nz)) continue;
        const ni = nx * gz + nz;
        if (seen[ni] || this.wall[ni]) continue;
        seen[ni] = 1;
        reached++;
        stack.push(ni);
      }
    }
    return reached === total;
  }

  /** Bresenham line-of-sight between world points. `props` optionally block. */
  lineOfSight(ax, az, bx, bz, propsBlock = true) {
    const cs = this.cs;
    let x0 = Math.floor(ax / cs);
    let z0 = Math.floor(az / cs);
    const x1 = Math.floor(bx / cs);
    const z1 = Math.floor(bz / cs);
    const dx = Math.abs(x1 - x0);
    const dz = Math.abs(z1 - z0);
    const sx = x0 < x1 ? 1 : -1;
    const sz = z0 < z1 ? 1 : -1;
    let err = dx - dz;
    let guard = 0;
    while (guard++ < 256) {
      if (x0 === x1 && z0 === z1) return true;
      const e2 = 2 * err;
      if (e2 > -dz) {
        err -= dz;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        z0 += sz;
      }
      if (this.isWall(x0, z0)) return false;
      if (propsBlock) {
        const pi = this.propAt[this.idx(x0, z0)];
        if (pi >= 0 && this.props[pi].blocksSight) {
          const p = this.props[pi];
          const px = (x0 + 0.5) * cs;
          const pz = (z0 + 0.5) * cs;
          // crude: only if the segment really passes near the furniture centre
          if (distToSegment(p.x, p.z, ax, az, bx, bz) < p.r * 0.85) return false;
          void px;
          void pz;
        }
      }
    }
    return false;
  }

  /** Breadth-first distance (in cells) from a seed cell over walkable cells. */
  distanceField(seedX, seedZ, out) {
    const { gx, gz } = this;
    const dist = out || new Int16Array(gx * gz);
    dist.fill(-1);
    const q = new Int32Array(gx * gz);
    let head = 0;
    let tail = 0;
    const s = seedX * gz + seedZ;
    dist[s] = 0;
    q[tail++] = s;
    while (head < tail) {
      const i = q[head++];
      const x = (i / gz) | 0;
      const z = i % gz;
      const d = dist[i];
      const nb = [i + gz, i - gz, i + 1, i - 1];
      const nbx = [x + 1, x - 1, x, x];
      const nbz = [z, z, z + 1, z - 1];
      for (let k = 0; k < 4; k++) {
        const nx = nbx[k];
        const nz = nbz[k];
        if (!this.inBounds(nx, nz)) continue;
        const ni = nb[k];
        if (this.wall[ni] || dist[ni] >= 0) continue;
        dist[ni] = d + 1;
        q[tail++] = ni;
      }
    }
    return dist;
  }
}

export function distToSegment(px, pz, ax, az, bx, bz) {
  const vx = bx - ax;
  const vz = bz - az;
  const wx = px - ax;
  const wz = pz - az;
  const len2 = vx * vx + vz * vz;
  let t = len2 > 0 ? (wx * vx + wz * vz) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * vx;
  const cz = az + t * vz;
  return Math.hypot(px - cx, pz - cz);
}

/** Slide a circular body through the grid; returns true if a wall was hit. */
export function collideWalls(grid, fromX, fromZ, toX, toZ, radius) {
  let hit = false;
  let x = toX;
  let z = toZ;
  const r = radius;
  const test = (px, pz) => {
    const x0 = Math.floor((px - r) / grid.cs);
    const x1 = Math.floor((px + r) / grid.cs);
    const z0 = Math.floor((pz - r) / grid.cs);
    const z1 = Math.floor((pz + r) / grid.cs);
    for (let gx = x0; gx <= x1; gx++)
      for (let gz = z0; gz <= z1; gz++) if (grid.isWall(gx, gz)) return true;
    return false;
  };
  if (test(x, z)) {
    hit = true;
    // axis-separated resolution: keep whichever component is free
    const keepX = !test(x, fromZ);
    const keepZ = !test(fromX, z);
    if (keepX) z = fromZ;
    else if (keepZ) x = fromX;
    else {
      x = fromX;
      z = fromZ;
    }
  }
  return { x, z, hit };
}

export function collideProps(grid, x, z, radius) {
  let px = x;
  let pz = z;
  let hit = false;
  const gx0 = Math.floor((x - 2) / grid.cs);
  const gx1 = Math.floor((x + 2) / grid.cs);
  const gz0 = Math.floor((z - 2) / grid.cs);
  const gz1 = Math.floor((z + 2) / grid.cs);
  for (let cx = gx0; cx <= gx1; cx++) {
    for (let cz = gz0; cz <= gz1; cz++) {
      if (!grid.inBounds(cx, cz)) continue;
      const pi = grid.propAt[grid.idx(cx, cz)];
      if (pi < 0) continue;
      const p = grid.props[pi];
      if (!p.blocksMove) continue;
      const d = Math.hypot(px - p.x, pz - p.z);
      const minD = p.r + radius;
      if (d < minD && d > 1e-6) {
        const nx = (px - p.x) / d;
        const nz = (pz - p.z) / d;
        px = p.x + nx * minD;
        pz = p.z + nz * minD;
        hit = true;
      }
    }
  }
  return { x: px, z: pz, hit };
}

export const MAP_BOUNDS = () => ({
  min: MAP.cellSize,
  max: (MAP.gx - 1) * MAP.cellSize,
  walk: PHYS.walkSpeed,
});
