/**
 * UI for the headless trainer worker.
 *
 * Deliberately blunt about whether Clark has learned anything yet: the verdict
 * line below the charts states what the telemetry actually shows, so it is never
 * possible to mistake a lucky lunge for hunting.
 */

const worker = new Worker(new URL('./lab-worker.js', import.meta.url), { type: 'module' });

const el = (id) => document.getElementById(id);
const state = el('state');
let last = null;

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'state') {
    state.textContent = msg.state;
    return;
  }
  if (msg.type === 'log') {
    logLine(msg.text);
    return;
  }
  if (msg.type === 'checkpoint') {
    if (!msg.json) {
      logLine('nothing to save yet');
      return;
    }
    const blob = new Blob([msg.json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `hussrooms-clark-${Math.round(last?.stats.t ?? 0)}s.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    logLine(`checkpoint saved (${(msg.json.length / 1024).toFixed(0)} KiB)`);
    return;
  }
  if (msg.type !== 'stats') return;
  last = msg;
  render(msg);
};

function render({ stats, fps }) {
  const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
  const mins = Math.floor(stats.t / 60);
  const secs = Math.round(stats.t % 60);
  const ratio = stats.wall > 0 ? stats.t / stats.wall : 0;

  const common = (b) => `
    <div class="stat"><span>gradient steps</span><b>${b.updates}</b></div>
    <div class="stat"><span>decisions</span><b>${b.decisions ?? '—'}</b></div>
    <div class="stat"><span>replay frames</span><b>${b.replay}</b></div>
    <div class="stat"><span>loss (ema)</span><b>${fmt(b.loss, 3)}</b></div>
    <div class="stat"><span>value estimate</span><b>${fmt(b.value, 3)}</b></div>
    <div class="stat"><span>search entropy</span><b>${fmt(b.entropy, 2)} / ${Math.log(b.actionDim).toFixed(2)}</b></div>
    ${b.skipped != null ? `<div class="stat"><span>non-finite updates skipped</span><b>${b.skipped}</b></div>` : ''}`;

  el('clark-stats').innerHTML =
    `<div class="stat"><span>sim time</span><b>${mins}m${String(secs).padStart(2, '0')}s</b></div>` +
    `<div class="stat"><span>wall clock</span><b>${(stats.wall / 60).toFixed(1)}m (${ratio.toFixed(2)}× realtime)</b></div>` +
    `<div class="stat"><span>sim frames/s</span><b>${fmt(fps, 0)}</b></div>` +
    `<div class="stat"><span>kills</span><b>${stats.clarkKills}</b></div>` +
    `<div class="stat"><span>time to first kill</span><b>${stats.firstKillAt == null ? '—' : stats.firstKillAt.toFixed(1) + 's'}</b></div>` +
    `<div class="stat"><span>bites / lunges / roars</span><b>${stats.bites} / ${stats.lunges} / ${stats.roars}</b></div>` +
    `<div class="stat"><span>sees prey</span><b>${((stats.telemetry.clarkSees / Math.max(1, stats.telemetry.steps)) * 100).toFixed(1)}% of ticks</b></div>` +
    `<div class="stat"><span>closest approach</span><b>${Number.isFinite(stats.telemetry.clarkClosest) ? stats.telemetry.clarkClosest.toFixed(1) + ' m' : '—'}</b></div>` +
    common(stats.clark);

  el('bot-stats').innerHTML =
    `<div class="stat"><span>bot deaths</span><b>${stats.botDeaths}</b></div>` +
    `<div class="stat"><span>player deaths (scripted)</span><b>${stats.playerDeaths}</b></div>` +
    `<div class="stat"><span>see Clark</span><b>${((stats.telemetry.botSees / Math.max(1, stats.telemetry.steps)) * 100).toFixed(1)}% of ticks</b></div>` +
    common(stats.bot);

  bars(el('clark-bars'), stats.clark.hist, true);
  bars(el('bot-bars'), stats.bot.hist, false);
  spark(el('clark-spark'), stats.clark.history);
  spark(el('bot-spark'), stats.bot.history);

  el('log').innerHTML = stats.events
    .slice(-14)
    .map((e) => {
      if (e.type === 'death') return `<div>${e.t.toFixed(1)}s — ${e.kind} killed by ${e.by}</div>`;
      if (e.type === 'bite') return `<div>${e.t.toFixed(1)}s — bite on #${e.target}${e.lunging ? ' (lunge)' : ''}</div>`;
      if (e.type === 'roar') return `<div>${e.t.toFixed(1)}s — roar</div>`;
      if (e.type === 'respawn') return `<div>${e.t.toFixed(1)}s — ${e.kind} respawned</div>`;
      return '';
    })
    .join('');

  // the verdict: pure telemetry, no optimism
  const sees = (stats.telemetry.clarkSees / Math.max(1, stats.telemetry.steps)) * 100;
  let verdict;
  if (stats.clark.updates < 200) verdict = 'Clark has barely been trained — any behaviour is noise.';
  else if (stats.clarkKills === 0 && stats.bites === 0)
    verdict =
      `No bites yet after ${stats.clark.updates} gradient steps. Clark has seen prey ${sees.toFixed(0)}% of ticks ` +
      `and his closest approach was ${Number.isFinite(stats.telemetry.clarkClosest) ? stats.telemetry.clarkClosest.toFixed(1) + ' m' : '—'}. ` +
      `The hunting behaviour has not emerged in this run yet.`;
  else if (stats.clarkKills === 0)
    verdict = `Contact is happening (${stats.bites} bites, closest ${stats.telemetry.clarkClosest.toFixed(1)} m) but no kill yet.`;
  else verdict = `Clark has learned to kill: ${stats.clarkKills} kills, first at ${stats.firstKillAt.toFixed(1)}s.`;
  el('verdict').textContent = verdict;
}

function bars(container, hist, clark) {
  const max = Math.max(...hist.map((h) => h.p), 0.01);
  container.innerHTML = hist
    .map((h) => {
      if (h.p < 0.005) return '';
      const name = ['idle', 'fwd', 'back', 'strafeL', 'strafeR', 'fwdL', 'fwdR', 'backL', 'backR', 'turnL', 'turnR', 'lunge', 'roar'][h.a];
      const cls = clark && (h.a === 11 || h.a === 12) ? 'special' : '';
      return `<div class="bar-row ${cls}"><span>${name}</span><i style="width:${(h.p / max) * 100}%"></i><b>${(h.p * 100).toFixed(0)}%</b></div>`;
    })
    .join('');
}

function spark(canvas, history) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = 'rgba(0,0,0,0.35)';
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!history || history.length < 2) return;
  const draw = (arr, color) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const span = max - min || 1;
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * (canvas.width - 6) + 3;
      const y = canvas.height - 6 - ((v - min) / span) * (canvas.height - 12);
      i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    });
    g.stroke();
  };
  draw(history.map((h) => h.v), '#7fd0ff');
  draw(history.map((h) => h.loss), '#ff9d5c');
  g.fillStyle = '#9a9480';
  g.font = '11px ui-monospace, monospace';
  g.fillText('value estimate (blue) · loss (orange)', 6, 14);
}

function logLine(text) {
  const d = document.createElement('div');
  d.textContent = text;
  el('log').prepend(d);
}

el('start').onclick = () => {
  worker.postMessage({ type: 'start', profile: el('profile').value, seed: 20260529 });
  logLine('training started');
};
el('pause').onclick = () => {
  worker.postMessage({ type: 'pause' });
  logLine('paused');
};
el('save').onclick = () => worker.postMessage({ type: 'checkpoint' });
el('load').onchange = async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  worker.postMessage({ type: 'load', json: await f.text(), profile: el('profile').value });
};

// start straight away - opening the page should begin training
el('start').click();
