/**
 * Turns the simulator's grid into three.js geometry.
 *
 * One InstancedMesh per material keeps a 625-cell floor at ~10 draw calls, and
 * the level is rebuilt only when the map actually changes (never, in practice).
 */

import * as THREE from '../three-shim.js';
import { MAP } from '../sim/config.js';
import { makeCeilingTexture, makeFloorTexture, makeLightTexture, makeWallTexture } from './textures.js';

export class WorldView {
  constructor(grid, scene) {
    this.grid = grid;
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.build();
  }

  build() {
    const grid = this.grid;
    const cs = grid.cs;
    const W = grid.gx * cs;
    const H = MAP.wallHeight;

    this.wallTex = makeWallTexture(MAP.seed);
    const floorTex = makeFloorTexture();
    const ceilTex = makeCeilingTexture();
    const lightTex = makeLightTexture();

    // ---------------- walls (instanced)
    const wallCells = [];
    for (let x = 0; x < grid.gx; x++) for (let z = 0; z < grid.gz; z++) if (grid.isWall(x, z)) wallCells.push([x, z]);
    const wallGeo = new THREE.BoxGeometry(cs, H, cs);
    const wallMat = new THREE.MeshLambertMaterial({ map: this.wallTex });
    const walls = new THREE.InstancedMesh(wallGeo, wallMat, wallCells.length);
    const m = new THREE.Matrix4();
    wallCells.forEach(([x, z], i) => {
      m.makeTranslation((x + 0.5) * cs, H / 2, (z + 0.5) * cs);
      walls.setMatrixAt(i, m);
    });
    walls.instanceMatrix.needsUpdate = true;
    walls.receiveShadow = false;
    this.group.add(walls);

    // ---------------- floor + ceiling
    const floorGeo = new THREE.PlaneGeometry(W, W);
    const floor = new THREE.Mesh(floorGeo, new THREE.MeshLambertMaterial({ map: floorTex }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(W / 2, 0, W / 2);
    this.group.add(floor);

    const ceil = new THREE.Mesh(floorGeo.clone(), new THREE.MeshLambertMaterial({ map: ceilTex }));
    ceil.rotation.x = Math.PI / 2;
    ceil.position.set(W / 2, H, W / 2);
    this.group.add(ceil);

    // ---------------- ceiling light panels
    const lightCells = [];
    const every = MAP.lightsEvery;
    for (let x = 1; x < grid.gx - 1; x += every) {
      for (let z = 1; z < grid.gz - 1; z += every) if (!grid.isWall(x, z)) lightCells.push([x, z]);
    }
    const lightGeo = new THREE.PlaneGeometry(cs * 0.8, cs * 0.8);
    const lightMat = new THREE.MeshBasicMaterial({ map: lightTex, toneMapped: false });
    const lights = new THREE.InstancedMesh(lightGeo, lightMat, lightCells.length);
    lightCells.forEach(([x, z], i) => {
      m.makeRotationX(Math.PI / 2);
      m.setPosition((x + 0.5) * cs, H - 0.06, (z + 0.5) * cs);
      lights.setMatrixAt(i, m);
    });
    lights.instanceMatrix.needsUpdate = true;
    this.group.add(lights);

    // ---------------- furniture (ottomans etc.)
    const byType = new Map();
    for (const p of grid.props) {
      if (!byType.has(p.type)) byType.set(p.type, []);
      byType.get(p.type).push(p);
    }
    this.propMeshes = [];
    for (const [type, list] of byType) {
      const dims =
        type === 'ottoman'
          ? [0.9, 0.5, 0.9]
          : type === 'crate'
            ? [0.7, 0.7, 0.7]
            : type === 'table'
              ? [1.5, 0.75, 0.9]
              : type === 'sofa'
                ? [1.8, 0.8, 0.85]
                : [0.5, 0.9, 0.5];
      const color = type === 'ottoman' ? 0x7a4b3a : type === 'table' ? 0x6b4a2a : type === 'sofa' ? 0x4a5a6a : 0x8a7040;
      const geo = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
      const mat = new THREE.MeshLambertMaterial({ color });
      const inst = new THREE.InstancedMesh(geo, mat, list.length);
      const q = new THREE.Quaternion();
      const pos = new THREE.Vector3();
      const scl = new THREE.Vector3(1, 1, 1);
      list.forEach((p, i) => {
        q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), p.rot);
        pos.set(p.x, dims[1] / 2, p.z);
        m.compose(pos, q, scl);
        inst.setMatrixAt(i, m);
      });
      inst.instanceMatrix.needsUpdate = true;
      this.group.add(inst);
      this.propMeshes.push(inst);
    }
  }
}
