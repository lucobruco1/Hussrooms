/**
 * Procedural textures.
 *
 * There are no image assets in this repo on purpose: the prototype has to work
 * offline from a bare static server, and it keeps the "what you see is what the
 * agent sees" story simple. Everything is painted into canvases at load time.
 */

import * as THREE from '../three-shim.js';

function canvas(size = 256) {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  return c;
}

function finish(c, repeat = 1) {
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat, repeat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 2;
  return tex;
}

/** Monochrome-yellow wallpaper with faint vertical stripes and damp stains. */
export function makeWallTexture(seed = 7) {
  const c = canvas(256);
  const g = c.getContext('2d');
  g.fillStyle = '#d9c874';
  g.fillRect(0, 0, 256, 256);
  // subtle stripes
  for (let x = 0; x < 256; x += 16) {
    g.fillStyle = x % 32 === 0 ? 'rgba(255,250,205,0.30)' : 'rgba(160,140,60,0.16)';
    g.fillRect(x, 0, 8, 256);
  }
  // grime
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  for (let i = 0; i < 220; i++) {
    const x = rnd() * 256;
    const y = rnd() * 256;
    const r = 4 + rnd() * 26;
    const a = 0.015 + rnd() * 0.05;
    const grd = g.createRadialGradient(x, y, 0, x, y, r);
    grd.addColorStop(0, `rgba(120,100,40,${a})`);
    grd.addColorStop(1, 'rgba(120,100,40,0)');
    g.fillStyle = grd;
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.fill();
  }
  // damp baseboard
  const grad = g.createLinearGradient(0, 210, 0, 256);
  grad.addColorStop(0, 'rgba(120,100,40,0)');
  grad.addColorStop(1, 'rgba(90,72,28,0.35)');
  g.fillStyle = grad;
  g.fillRect(0, 200, 256, 56);
  return finish(c, 1);
}

/** Cheapest possible office carpet: dark mustard with noise. */
export function makeFloorTexture() {
  const c = canvas(256);
  const g = c.getContext('2d');
  g.fillStyle = '#a8964e';
  g.fillRect(0, 0, 256, 256);
  const img = g.getImageData(0, 0, 256, 256);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 34;
    img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
    img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
    img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n * 0.7));
  }
  g.putImageData(img, 0, 0);
  return finish(c, 12);
}

export function makeCeilingTexture() {
  const c = canvas(128);
  const g = c.getContext('2d');
  g.fillStyle = '#e8dca0';
  g.fillRect(0, 0, 128, 128);
  // ceiling tile grid
  g.strokeStyle = 'rgba(140,120,60,0.35)';
  g.lineWidth = 3;
  g.strokeRect(0, 0, 128, 128);
  return finish(c, 1);
}

/** Emissive light panel with a slight vignette so panels read as fixtures. */
export function makeLightTexture() {
  const c = canvas(64);
  const g = c.getContext('2d');
  g.fillStyle = '#fffbe0';
  g.fillRect(0, 0, 64, 64);
  const grd = g.createRadialGradient(32, 32, 4, 32, 32, 34);
  grd.addColorStop(0, 'rgba(255,255,255,0.95)');
  grd.addColorStop(1, 'rgba(255,246,190,0.55)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  return finish(c, 1);
}
