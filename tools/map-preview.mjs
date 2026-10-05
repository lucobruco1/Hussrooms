/**
 * Print the generated level as ASCII, plus a few layout statistics.
 *
 *   node tools/map-preview.mjs [seed] [--layout maze]
 *
 * Useful because the level generator is the single biggest lever on whether the
 * monster can learn anything: on a dense maze he must solve navigation before
 * he can ever bite anyone.
 */

import { Grid } from '../src/sim/grid.js';
import { MAP } from '../src/sim/config.js';

const argv = process.argv.slice(2);
const layoutIdx = argv.indexOf('--layout');
const layout = layoutIdx >= 0 ? argv[layoutIdx + 1] : MAP.layout;
MAP.layout = layout;
const seed = Number(argv.find((a) => /^\d+$/.test(a)) || MAP.seed);

const g = new Grid(seed);
const propAt = new Map();
for (const p of g.props) {
  const [cx, cz] = g.cellOf(p.x, p.z);
  propAt.set(`${cx},${cz}`, p.type === 'ottoman' ? 'o' : p.type === 'couch' ? 'c' : 'p');
}
const spawn = new Set(g.spawnPoints.map((p) => `${g.cellOf(p.x, p.z).join(',')}`));

let walls = 0;
let props = 0;
const lines = [];
for (let z = 0; z < g.gz; z++) {
  let line = '';
  for (let x = 0; x < g.gx; x++) {
    const key = `${x},${z}`;
    if (g.isWall(x, z)) {
      line += '█';
      walls++;
    } else if (propAt.has(key)) {
      line += propAt.get(key);
      props++;
    } else if (spawn.has(key)) {
      line += '·';
    } else {
      line += ' ';
    }
  }
  lines.push(line);
}
const interior = (g.gx - 2) * (g.gz - 2);
const border = 2 * (g.gx + g.gz) - 4;
console.log(`Hussrooms map — layout=${layout} seed=${seed} — ${g.gx}x${g.gz} cells @ ${g.cs} m ` +
  `(${(g.gx * g.cs).toFixed(0)}x${(g.gz * g.cs).toFixed(0)} m)`);
console.log(`█ wall ${(((walls - border) / interior) * 100).toFixed(1)}% of interior · · spawn point · ` +
  `o ottoman · p other prop (${props} visible) · connectivity ${g.floodReachesAll() ? 'ok' : 'BROKEN'}`);
console.log(lines.join('\n'));

let los = 0;
let tried = 0;
const pts = g.spawnPoints;
for (let i = 0; i < 600; i++) {
  const a = pts[i % pts.length];
  const b = pts[(i * 13 + 7) % pts.length];
  if (a === b) continue;
  tried++;
  if (g.lineOfSight(a.x, a.z, b.x, b.z, true)) los++;
}
console.log(`\nstats: spawn points ${pts.length} · random spawn-pair line of sight ${((los / tried) * 100).toFixed(1)}% ` +
  `· mean spawn spacing ${meanPairDistance(pts).toFixed(1)} m`);

function meanPairDistance(pts) {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < pts.length; i += 3) {
    for (let j = i + 1; j < pts.length; j += 7) {
      sum += Math.hypot(pts[i].x - pts[j].x, pts[i].z - pts[j].z);
      n++;
    }
  }
  return n ? sum / n : 0;
}
