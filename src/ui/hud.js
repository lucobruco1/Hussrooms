/**
 * HUD: the game-facing overlay plus the learning read-out.
 *
 * The learning panel is deliberately prominent. A prototype whose whole point
 * is "watch EfficientZero figure out that it is supposed to hunt" has to show
 * its work: value estimates, search entropy, replay size, gradient steps, the
 * action mix, and time-to-first-kill.
 */

import { ACTIONS } from '../sim/config.js';

const ACTION_NAMES = ACTIONS.map((a) => a.name);

export class Hud {
  constructor(session, root) {
    this.session = session;
    this.root = root;
    this.el = {};
    this.build();
  }

  build() {
    const mk = (tag, cls, parent, text) => {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text != null) e.textContent = text;
      (parent || this.root).appendChild(e);
      return e;
    };
    // --- crosshair
    this.el.crosshair = mk('div', 'crosshair');
    // --- top bar
    const top = mk('div', 'hud-top');
    this.el.status = mk('div', 'status', top);
    this.el.hint = mk('div', 'hint', top, 'WASD move · Shift sprint · mouse look · click fire · Tab lab panel');
    // --- left: vitals
    const left = mk('div', 'panel vitals');
    mk('div', 'panel-title', left, 'Survivors');
    this.el.vitals = mk('div', 'vitals-body', left);
    // --- right: learning
    const right = mk('div', 'panel learning');
    mk('div', 'panel-title', right, "Cap'n Clark — learning read-out");
    this.el.learning = mk('div', 'learning-body', right);
    this.el.clarkVal = mk('canvas', 'spark', right);
    this.el.clarkVal.width = 260;
    this.el.clarkVal.height = 46;
    this.el.clarkBars = mk('div', 'bars', right);
    this.el.clarkKills = mk('div', 'kills', right);
    mk('div', 'panel-title', right, 'Survivor bots');
    this.el.bots = mk('div', 'learning-body', right);
    this.el.botVal = mk('canvas', 'spark', right);
    this.el.botVal.width = 260;
    this.el.botVal.height = 46;
    this.el.botBars = mk('div', 'bars', right);
    // --- bottom: log
    const log = mk('div', 'panel log');
    this.el.log = mk('ul', 'log-list', log);
    this.el.logLines = [];
    // --- toast
    this.el.toast = mk('div', 'toast hidden');
  }

  toast(msg, ms = 2600) {
    this.el.toast.textContent = msg;
    this.el.toast.classList.remove('hidden');
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => this.el.toast.classList.add('hidden'), ms);
  }

  update(fps = 0) {
    const s = this.session;
    const sum = s.summary();
    const t = sum.t;
    const mm = String(Math.floor(t / 60)).padStart(2, '0');
    const ss = String(Math.floor(t % 60)).padStart(2, '0');
    this.el.status.innerHTML =
      `<b>HUS$ROOMS</b> · t ${mm}:${ss} · ` +
      `<span class="warn">Cap'n Clark kills: ${sum.clarkKills}</span> · ` +
      `bot deaths ${sum.botDeaths} · bites ${sum.bites} · ` +
      `updates ${sum.updates} · search ${s.botBrain.cfg.sims} sims` +
      (fps ? ` · ${fps.toFixed(0)} fps` : '');
    if (sum.firstKillAt != null) this.el.status.classList.add('has-kill');
    else this.el.status.classList.remove('has-kill');

    // vitals
    const rows = [];
    const bar = (v, max, cls) => `<div class="bar ${cls}"><i style="width:${Math.max(0, Math.min(100, (v / max) * 100))}%"></i></div>`;
    rows.push(`<div class="row"><span>You</span>${bar(sum.playerHealth, 100, 'hp')}<span>${Math.round(sum.playerHealth)}</span></div>`);
    rows.push(
      `<div class="row"><span>breath</span>${bar(sum.playerStamina * 100, 100, 'st')}<span>${Math.round(sum.playerStamina * 100)}</span></div>`,
    );
    if (sum.clarkAlive) rows.push(`<div class="row"><span>Clark</span>${bar(sum.clarkHealth, 260, 'clark')}<span>${Math.round(sum.clarkHealth)}</span></div>`);
    else rows.push(`<div class="row"><span>Clark</span><em>reforming…</em><span></span></div>`);
    for (const a of s.agentStats()) {
      if (a.kind !== 'bot') continue;
      const label = `${a.kind}#${a.id}`;
      if (!a.alive) rows.push(`<div class="row dead"><span>${label}</span><em>killed by Clark</em><span></span></div>`);
      else rows.push(`<div class="row"><span>${label}</span>${bar(a.health, a.maxHealth, 'hp')}<span>${Math.round(a.health)}</span></div>`);
    }
    this.el.vitals.innerHTML = rows.join('');

    // learning panels
    const brainRow = (b, stats) =>
      `<div class="kv"><span>gradient steps</span><b>${b.updates}</b></div>` +
      `<div class="kv"><span>decisions</span><b>${b.decisions}</b></div>` +
      `<div class="kv"><span>replay</span><b>${b.replay.size}</b></div>` +
      `<div class="kv"><span>value est.</span><b>${stats?.value ?? 0}</b></div>` +
      `<div class="kv"><span>loss (ema)</span><b>${b.lossEma.toFixed(3)}</b></div>` +
      `<div class="kv"><span>search entropy</span><b>${b.policyEntropy.toFixed(2)} / ${Math.log(b.cfg.actionDim).toFixed(2)}</b></div>`;
    this.el.learning.innerHTML =
      `<div class="kv"><span>kills</span><b class="big">${this.session.world.clark.stats.kills}</b></div>` +
      `<div class="kv"><span>time to first kill</span><b>${sum.firstKillAt == null ? '—' : sum.firstKillAt.toFixed(1) + 's'}</b></div>` +
      `<div class="kv"><span>closest approach</span><b>${Number.isFinite(this.session.world.telemetry.clarkClosest) ? this.session.world.telemetry.clarkClosest.toFixed(1) + ' m' : '—'}</b></div>` +
      `<div class="kv"><span>sees prey</span><b>${(
        (this.session.world.telemetry.clarkSees / Math.max(1, this.session.world.telemetry.steps)) *
        100
      ).toFixed(1)}% of ticks</b></div>` +
      `<div class="kv"><span>his action right now</span><b>${ACTION_NAMES[this.session.clarkBrain.lastAction] ?? '—'}</b></div>` +
      brainRow(this.session.clarkBrain, { value: sum.clarkValue.toFixed(3) });
    this.el.bots.innerHTML =
      `<div class="kv"><span>deaths</span><b>${sum.botDeaths}</b></div>` +
      `<div class="kv"><span>sees Clark</span><b>${(
        (this.session.world.telemetry.botSees / Math.max(1, this.session.world.telemetry.steps)) *
        100
      ).toFixed(1)}% of ticks</b></div>` +
      brainRow(this.session.botBrain, { value: sum.botValue.toFixed(3) });

    drawSpark(this.el.clarkVal, this.session.clarkBrain.history);
    drawSpark(this.el.botVal, this.session.botBrain.history);
    renderBars(this.el.clarkBars, this.session.actionHistogram('clark'), true);
    renderBars(this.el.botBars, this.session.actionHistogram('bots'), false);

    // event log
    const evs = this.session.lastEvents.slice(-6);
    const lines = evs.map((e) => {
      if (e.type === 'death') return `<li class="bad">${e.by === 'clark' ? "Cap'n Clark killed" : 'death of'} ${e.kind} at ${e.t.toFixed(1)}s</li>`;
      if (e.type === 'bite') return `<li class="warn">bite on #${e.target}${e.lunging ? ' (lunge)' : ''} at ${e.t.toFixed(1)}s</li>`;
      if (e.type === 'roar') return `<li>roar at ${e.t.toFixed(1)}s</li>`;
      if (e.type === 'respawn') return `<li class="dim">${e.kind} respawned</li>`;
      return '';
    });
    this.el.log.innerHTML = lines.join('');
  }
}

function drawSpark(canvas, history) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = 'rgba(0,0,0,0.35)';
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!history || history.length < 2) {
    g.fillStyle = '#8a8360';
    g.font = '11px ui-monospace, monospace';
    g.fillText('collecting…', 8, 26);
    return;
  }
  const vs = history.map((h) => h.v);
  const ls = history.map((h) => h.loss);
  const draw = (arr, color, scale) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const span = max - min || 1;
    g.strokeStyle = color;
    g.lineWidth = 1.6;
    g.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * (canvas.width - 6) + 3;
      const y = canvas.height - 6 - ((v - min) / span) * (canvas.height - 14) * scale;
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
  };
  draw(vs, '#7fd0ff', 1);
  draw(ls, '#ff9d5c', 0.9);
  g.fillStyle = '#9a9480';
  g.font = '10px ui-monospace, monospace';
  g.fillText('value est. (blue) · loss (orange)', 6, 12);
}

function renderBars(container, hist, clark) {
  const max = Math.max(...hist.map((h) => h.p), 0.01);
  container.innerHTML = hist
    .map((h, i) => {
      if (h.p < 0.005) return '';
      const w = (h.p / max) * 100;
      const cls = clark && (i === 11 || i === 12) ? 'special' : '';
      return `<div class="bar-row ${cls}"><span>${ACTION_NAMES[i]}</span><i style="width:${w}%"></i><b>${(h.p * 100).toFixed(0)}%</b></div>`;
    })
    .join('');
}
