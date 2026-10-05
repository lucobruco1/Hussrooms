/**
 * Single import point for three.js.
 *
 * The build is vendored (vendor/three/three.module.js) so the prototype runs
 * from any static server with no CDN and no bundler - the preview sandbox has
 * no outbound network, and a game that only works online is not much of a
 * prototype. Swap this file's contents for a CDN URL if you prefer.
 */
export * from '../vendor/three/three.module.js';
