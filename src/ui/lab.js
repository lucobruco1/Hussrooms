/**
 * The Lab panel: everything about the learning process, exposed.
 *
 * This is the part of the prototype that is actually the point. It shows the
 * search tree the monster is running right now, what each head predicts, how
 * much experience it has, and lets you save/restore a brain so a run can be
 * continued in the browser or handed to node tools.
 */

import { A } from '../sim/config.js';

const NAMES = Object.fromEntries(Object.entries(A).map(([k, v]) => [v, k]));

export class LabPanel {
  constructor(session, brain, root) {
    this.session = session;
    this.brain = brain;
    this.root = root;
    this.visible = false;
    this.el = document.createElement('div');
    this.el.className = 'lab hidden';
    this.root.appendChild(this.el);
    this.build();
  }

  build() {
    const b = this.brain;
    this.el.innerHTML = `
      <div class="lab-head">
        <b>EfficientZero lab</b>
        <span class="lab-sub">${b.id} · ${b.cfg.actionDim} actions · ${b.cfg.label || b.cfg.profileName || ''}</span>
        <button class="lab-x" title="close (Tab)">✕</button>
      </div>
      <div class="lab-grid">
        <section>
          <h4>Root search — what Clark considered</h4>
          <div class="lab-search" id="lab-search"></div>
        </section>
        <section>
          <h4>Latent dynamics rollout</h4>
          <canvas id="lab-latent" width="320" height="120"></canvas>
          <div class="lab-note">predicted rewards (green) and value (blue) along the search's principal line</div>
        </section>
        <section>
          <h4>Learning</h4>
          <div class="lab-kv" id="lab-learning"></div>
        </section>
        <section>
          <h4>Controls</h4>
          <div class="lab-controls">
            <label>sim speed <input type="range" id="lab-speed" min="0" max="6" step="1" value="${this.session.simSpeed}"></label>
            <label><input type="checkbox" id="lab-learn" ${this.session.learn ? 'checked' : ''}> train while playing</label>
            <label><input type="checkbox" id="lab-explore"> force exploration (ignore policy)</label>
            <button id="lab-save">save brain (.json)</button>
            <label class="file">load brain <input type="file" id="lab-load" accept=".json"></label>
            <button id="lab-reset">reset learning</button>
          </div>
        </section>
      </div>
      <div class="lab-foot" id="lab-foot"></div>`;
    this.el.querySelector('.lab-x').onclick = () => this.toggle(false);
    this.el.querySelector('#lab-speed').oninput = (e) => {
      this.session.simSpeed = parseInt(e.target.value, 10);
    };
    this.el.querySelector('#lab-learn').onchange = (e) => {
      this.session.learn = e.target.checked;
    };
    this.el.querySelector('#lab-explore').onchange = (e) => {
      this.session.forceExplore = e.target.checked;
    };
    this.el.querySelector('#lab-save').onclick = () => this.saveBrain();
    this.el.querySelector('#lab-load').onchange = (e) => this.loadBrain(e.target.files[0]);
    this.el.querySelector('#lab-reset').onclick = () => {
      if (confirm('Throw away everything Clark has learned so far?')) {
        this.session.resetLearning();
        this.brain = this.session.clarkBrain;
        this.build();
      }
    };
    this.searchEl = this.el.querySelector('#lab-search');
    this.learningEl = this.el.querySelector('#lab-learning');
    this.latentEl = this.el.querySelector('#lab-latent');
    this.footEl = this.el.querySelector('#lab-foot');
  }

  toggle(on = !this.visible) {
    this.visible = on;
    this.el.classList.toggle('hidden', !on);
  }

  async saveBrain() {
    const obj = this.session.checkpoint();
    const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `hussrooms-clark-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async loadBrain(file) {
    if (!file) return;
    const text = await file.text();
    try {
      const ok = this.session.loadCheckpoint(JSON.parse(text));
      this.footEl.textContent = ok ? `loaded ${file.name}` : 'not a hussrooms checkpoint';
    } catch (err) {
      this.footEl.textContent = 'bad file: ' + err.message;
    }
  }

  /** Called once per rendered frame; cheap DOM writes only. */
  update() {
    if (!this.visible) return;
    const b = this.brain;
    const s = b.lastTrace;
    if (s && s.children) {
      const maxN = Math.max(1, ...s.children.map((c) => c.n));
      this.searchEl.innerHTML = s.children
        .slice()
        .sort((x, y) => y.n - x.n)
        .slice(0, 8)
        .map(
          (c) =>
            `<div class="lab-row ${c.a === s.action ? 'picked' : ''}">` +
            `<span>${NAMES[c.a] ?? c.a}</span>` +
            `<i style="width:${(c.n / maxN) * 100}%"></i>` +
            `<b>${c.n} visits · p ${(c.p ?? 0).toFixed(2)} · q ${(c.q ?? 0).toFixed(2)}</b>` +
            `</div>`,
        )
        .join('');
    } else {
      this.searchEl.innerHTML = '<div class="lab-note">no search yet</div>';
    }

    this.learningEl.innerHTML =
      kv('gradient steps', b.updates) +
      kv('decisions', b.decisions) +
      kv('replay frames', b.replay.size) +
      kv('loss (ema)', b.lossEma.toFixed(4)) +
      kv('value mean/spread', `${b.net.valueMean.toFixed(2)} / ${b.net.valueSpread.toFixed(2)}`) +
      kv('consistency every k', b.cfg.consistEveryK ?? '—') +
      kv('non-finite updates skipped', b.skippedUpdates ?? 0) +
      kv('reward scale', b.cfg.rewardScale);

    drawLatent(this.latentEl, s);
    const e = this.session.world.telemetry;
    const closest = Number.isFinite(e.clarkClosest) ? e.clarkClosest.toFixed(1) + ' m' : '—';
    this.footEl.textContent =
      `telemetry: clark sees prey ${((e.clarkSees / Math.max(1, e.steps)) * 100).toFixed(1)}% of ticks · ` +
      `closest approach ${closest} · bites ${e.bites} · kills ${e.clarkKills}`;
  }
}

function kv(k, v) {
  return `<div class="kv"><span>${k}</span><b>${v}</b></div>`;
}

function drawLatent(canvas, trace) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = 'rgba(0,0,0,0.35)';
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!trace || !trace.rollout || trace.rollout.length < 2) {
    g.fillStyle = '#8a8360';
    g.font = '11px ui-monospace, monospace';
    g.fillText('no rollout yet', 10, 30);
    return;
  }
  const rs = trace.rollout.map((r) => r.reward);
  const vs = trace.rollout.map((r) => r.value);
  const all = rs.concat(vs);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const plot = (arr, color) => {
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    arr.forEach((v, i) => {
      const x = 8 + (i / (arr.length - 1)) * (canvas.width - 16);
      const y = canvas.height - 8 - ((v - min) / span) * (canvas.height - 24);
      i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    });
    g.stroke();
  };
  plot(vs, '#7fd0ff');
  plot(rs, '#7fe08a');
  // horizon markers
  g.strokeStyle = 'rgba(255,255,255,0.12)';
  g.lineWidth = 1;
  for (let i = 0; i < rs.length - 1; i++) {
    const x = 8 + ((i + 0.5) / (rs.length - 1)) * (canvas.width - 16);
    g.beginPath();
    g.moveTo(x, 4);
    g.lineTo(x, canvas.height - 4);
    g.stroke();
  }
}
