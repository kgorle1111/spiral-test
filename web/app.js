/* Spiral Test app shell: hash routing, screens, local storage.
   No backend: records live in localStorage on this device only. */
'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = Charts.esc;
  const RECORDS_KEY = 'spiral.records.v1';
  const HAND_KEY = 'spiral.hand.v1';
  const DISCLAIMER = 'Research prototype · not a medical device · not for diagnosis';
  const TAGS = [
    { group: 'dose', id: 'dose-under-1h', label: 'Dose <1 h ago' },
    { group: 'dose', id: 'dose-1-3h', label: 'Dose 1–3 h ago' },
    { group: 'dose', id: 'dose-over-3h', label: 'Dose >3 h ago' },
    { id: 'caffeine', label: 'Caffeine' },
    { id: 'poor-sleep', label: 'Poor sleep' },
    { id: 'alcohol-yesterday', label: 'Alcohol yesterday' },
  ];
  const TAG_LABEL = Object.fromEntries(TAGS.map((t) => [t.id, t.label]));
  TAG_LABEL.uncalibrated = 'Uncalibrated';

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) {
      try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; }
    },
  };
  const records = () => { const r = store.get(RECORDS_KEY, []); return Array.isArray(r) ? r : []; };

  /* ------------------------------------------------------------ engine pill */
  Engine.onChange((state, detail) => {
    const el = $('engine'), txt = $('engine-text');
    el.dataset.state = state;
    if (state === 'ready') { txt.textContent = 'ready'; el.title = 'Python analysis engine loaded (Pyodide 0.28.3 + numpy). Runs on this device.'; }
    else if (state === 'error') { txt.textContent = 'failed'; el.title = Engine.errorText(); }
    else { txt.textContent = 'loading…'; el.title = 'Loading the Python analysis engine (about 10 MB, cached after the first visit): ' + detail; }
  });
  $('engine').addEventListener('click', () => {
    if (Engine.state() === 'error') alert(Engine.errorText());
  });

  /* ------------------------------------------------------------ router */
  let current = null;
  const leaveHooks = [];
  function route() {
    const raw = location.hash.replace(/^#\/?/, '');
    const [name] = raw.split('?');
    const routes = { '': 'home', start: 'start', calibrate: 'calibrate', draw: 'draw', result: 'result', history: 'history', real: 'real', evidence: 'evidence' };
    let r = routes[name] || 'home';
    if (r === 'start') { location.replace(Calibration.get() ? '#/draw' : '#/calibrate?next=draw'); return; }
    if (r === 'result' && !lastResult) { location.replace('#/'); return; }
    while (leaveHooks.length) leaveHooks.pop()();
    document.body.classList.remove('drawing');
    document.querySelectorAll('.screen').forEach((s) => { s.hidden = s.dataset.route !== r; });
    document.querySelectorAll('[data-nav]').forEach((a) => { if (a.dataset.nav === r) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    current = r;
    const titles = { home: '', calibrate: 'Calibrate', draw: 'Draw', result: 'Result', history: 'History', real: 'Real recording', evidence: 'Evidence' };
    document.title = (titles[r] ? titles[r] + ' · ' : '') + 'Spiral Test';
    window.scrollTo(0, 0);
    ({ home: showHome, calibrate: showCalibrate, draw: showDraw, result: showResult, history: showHistory, real: showReal, evidence: showEvidence })[r]();
    const h = document.querySelector('.screen:not([hidden]) h1');
    if (h && r !== 'home') h.focus({ preventScroll: true });
  }
  window.addEventListener('hashchange', route);

  /* ------------------------------------------------------------ home */
  let heroDone = false;
  function showHome() {
    if (!heroDone) { Charts.hero($('hero-illus')); heroDone = true; }
    const cal = Calibration.get();
    $('cal-status').textContent = !cal ? 'First time: you\'ll match a bank card to the screen (20 seconds) so millimetres are real.'
      : cal.calibrated ? `Calibrated: ${cal.px_per_mm.toFixed(2)} px per mm.` : 'Not calibrated: mm values are approximate. Calibrate from the link below.';
  }

  /* ------------------------------------------------------------ calibrate */
  function showCalibrate() {
    const range = $('cal-range');
    range.value = Calibration.initialGuess();
    const update = () => {
      const p = +range.value;
      const card = $('cal-card');
      card.style.width = (Calibration.CARD_W_MM * p) + 'px';
      card.style.height = (Calibration.CARD_H_MM * p) + 'px';
      card.style.setProperty('--card-r', (Calibration.CARD_R_MM * p) + 'px');
      $('cal-readout').textContent = `${p.toFixed(2)} px per mm · outline ${Math.round(Calibration.CARD_W_MM * p)} px wide`;
      range.setAttribute('aria-valuetext', `${p.toFixed(2)} pixels per millimetre`);
    };
    range.oninput = update;
    $('cal-minus').onclick = () => { range.value = (+range.value - 0.02).toFixed(2); update(); };
    $('cal-plus').onclick = () => { range.value = (+range.value + 0.02).toFixed(2); update(); };
    const next = () => location.hash = new URLSearchParams(location.hash.split('?')[1] || '').get('next') === 'draw' ? '#/draw' : '#/';
    $('cal-save').onclick = () => { Calibration.save(+range.value, true); next(); };
    $('cal-skip').onclick = () => { Calibration.save(Calibration.BROWSER_GUESS, false); next(); };
    update();
  }

  /* ------------------------------------------------------------ draw */
  let lastCapture = null;
  function showDraw() {
    const cal = Calibration.get();
    if (!cal) { location.replace('#/calibrate?next=draw'); return; }
    $('hand-pick').hidden = false;
    $('draw-area').hidden = true;
    const lastHand = store.get(HAND_KEY, null);
    document.querySelectorAll('.btn-hand').forEach((b) => {
      b.classList.toggle('btn-secondary', b.dataset.hand === lastHand);
      b.onclick = () => startDrawing(b.dataset.hand);
    });
    // "Try again" links carry the hand, so a redo is one tap.
    const h = new URLSearchParams(location.hash.split('?')[1] || '').get('hand');
    if (h === 'right' || h === 'left') startDrawing(h);
  }

  function startDrawing(hand) {
    store.set(HAND_KEY, hand);
    const cal = Calibration.get();
    $('hand-pick').hidden = true;
    $('draw-area').hidden = false;
    document.body.classList.add('drawing');
    const stage = $('draw-stage');
    // Fit the square canvas to the smaller of the free width and the free height.
    const avW = stage.getBoundingClientRect().width;
    const used = $('draw-area').getBoundingClientRect().top + 70 + document.querySelector('.draw-bar').getBoundingClientRect().height + 16;
    const avH = window.innerHeight - used;
    const side = Math.floor(Math.max(220, Math.min(avW, avH, 640)));
    const tpl = Template.make(side, cal.px_per_mm, hand);
    const dev = Template.selfCheck(tpl);
    if (dev > 0.01) console.error('template self-check failed: RMS', dev, 'mm'); // would mean the drawn spiral and the dict disagree
    const instr = $('draw-instr');
    instr.textContent = `${hand === 'right' ? 'Right' : 'Left'} hand. Start at the red dot, follow the line outward, slowly.`;
    if (tpl.rMm < 18) instr.textContent += ' (Small screen: the spiral is small.)';
    Capture.setup($('draw-canvas'), tpl, {
      onUpdate(s) {
        $('draw-live').textContent = `${s.elapsed.toFixed(1)} s · ${s.rate ? Math.round(s.rate) : '–'} pts/s` + (s.lifts ? ` · ${s.lifts} lift${s.lifts > 1 ? 's' : ''}` : '');
        $('draw-done').disabled = s.n < 2;
      },
      onFinish(d) {
        lastCapture = { ...d, hand, calibrated: !!cal.calibrated, px_per_mm: cal.px_per_mm };
        runAnalysis({ kind: 'phone', capture: lastCapture });
      },
    });
    $('draw-done').onclick = () => Capture.finish();
    $('draw-clear').onclick = () => Capture.reset();
    $('draw-cancel').onclick = () => { location.hash = '#/'; };
  }

  /* ------------------------------------------------------------ analysis */
  let lastResult = null; // { kind, capture|uci, res, saved }
  async function runAnalysis(ctx) {
    lastResult = { ...ctx, res: null, pending: true };
    location.hash = '#/result';
    if (current === 'result') showResult();
    try {
      const payload = ctx.kind === 'phone'
        ? { t_ms: ctx.capture.t_ms, x_mm: ctx.capture.x_mm, y_mm: ctx.capture.y_mm, stroke: ctx.capture.stroke, template: ctx.capture.template }
        : { t_ms: ctx.uci.t_ms, x_mm: ctx.uci.x_mm, y_mm: ctx.uci.y_mm };
      const res = await (ctx.promise || Engine.analyze(payload));
      if (lastResult.capture !== ctx.capture || lastResult.uci !== ctx.uci) return; // superseded
      lastResult = { ...ctx, res, pending: false };
    } catch (e) {
      lastResult = { ...ctx, res: null, pending: false, error: String(e.message || e) };
    }
    if (current === 'result') showResult();
  }

  const fmt = (v, d) => (v == null || !Number.isFinite(v) ? '–' : v.toFixed(d));

  function showResult() {
    const body = $('result-body'), src = $('result-source'), live = $('result-live');
    const L = lastResult;
    src.innerHTML = '';
    if (L.kind === 'uci') {
      src.innerHTML = `<div class="banner"><b>Tablet recording, not a phone.</b> ${esc(L.uci.label)} · subject ${esc(L.uci.subject)} · ${esc(L.uci.test)}. Pen on a Wacom tablet; mm assume 0.204 mm per tablet pixel. Not saved to your history.</div>`;
    } else if (L.capture && !L.capture.calibrated) {
      src.innerHTML = `<div class="banner"><b>Uncalibrated.</b> Millimetre values use the browser's guess and may be off by up to 2×. Hz is unaffected. <a href="#/calibrate">Calibrate</a></div>`;
    }
    if (L.pending) {
      const waiting = Engine.state() !== 'ready';
      body.innerHTML = `<div class="card"><p class="eyebrow">Analysing</p><h1 tabindex="-1">${waiting ? 'Waiting for the analysis engine…' : 'Measuring your drawing…'}</h1>
        <p class="muted">${waiting ? 'First visit downloads about 10 MB of Python (cached after). Your drawing is kept and will be analysed as soon as it is ready.' : 'Running spiral/metrics.py on this device.'}</p></div>`;
      live.textContent = 'Analysing';
      return;
    }
    if (L.error) {
      const engineDown = Engine.state() === 'error';
      body.innerHTML = `<div class="card redo"><h1 tabindex="-1">${engineDown ? 'The analysis engine didn\'t load' : 'The analysis failed'}</h1>
        <p>${esc(engineDown ? Engine.errorText() : 'The analysis code raised an error: ' + L.error)}</p>
        <div class="row">${rawButton(L)}<a class="btn btn-primary" href="#/">Home</a></div></div>`;
      wireRaw(L);
      live.textContent = engineDown ? Engine.errorText() : 'Analysis failed';
      return;
    }
    const res = L.res;
    const again = L.kind === 'uci' ? '<a class="btn btn-primary" href="#/real">Pick another recording</a>' : `<a class="btn btn-primary btn-big" href="#/draw?hand=${L.capture.hand}">Try again</a>`;
    if (res.status === 'error' || res.status === 'redo') {
      const reasons = res.status === 'error'
        ? [{ code: 'too_few_points', message: `Too few points were recorded to measure anything (${res.message}). Trace the whole spiral.` }]
        : res.reasons;
      body.innerHTML = `<div class="card redo">
        <p class="eyebrow">No measurement</p>
        <h1 tabindex="-1">Redo: this drawing can't be trusted</h1>
        <p>A wrong number is worse than no number, so nothing was measured. ${reasons.length > 1 ? 'Reasons' : 'Reason'}:</p>
        <ul class="reasons">${reasons.map((r) => `<li><code>${esc(r.code)}</code>${esc(r.message)}</li>`).join('')}</ul>
        <div class="row">${again}${rawButton(L)}</div>
      </div>
      <p class="notice">${DISCLAIMER}.</p>`;
      wireRaw(L);
      live.textContent = 'Redo. ' + reasons.map((r) => r.message).join(' ');
      return;
    }
    const m = res.metrics, q = res.quality;
    const freqTxt = m.freq_hz != null ? `<span class="acc">${m.freq_hz.toFixed(1)}</span><span class="u">Hz</span>` : '<span class="u" style="margin:0">no dominant rhythm</span>';
    const ampTxt = `<span class="acc">${fmt(m.amp_pp_mm, m.amp_pp_mm >= 10 ? 0 : m.amp_pp_mm >= 1 ? 1 : 2)}</span><span class="u">mm</span>`;
    const uncal = L.kind === 'phone' && !L.capture.calibrated;
    body.innerHTML = `
      <div class="card readout">
        <p class="eyebrow">${L.kind === 'uci' ? 'Result on a real recording' : 'Your result · ' + esc(L.capture.hand) + ' hand'}</p>
        <h1 tabindex="-1" class="big"><span class="sr-only">Peak </span>${freqTxt}<span class="sep">·</span>${ampTxt}</h1>
        <p class="sub">${m.freq_hz != null ? 'Tremor rhythm (peak of the spectrum)' : 'No single rhythm stands out, which is normal for a steady hand'} · peak-to-peak tremor size${uncal ? ' (uncalibrated)' : ''}</p>
        <div class="chips" aria-label="Drawing quality">
          <span class="chip ok">${Math.round(q.rate_hz)} pts/s</span>
          <span class="chip">${fmt(q.duration_s, 1)} s</span>
          <span class="chip">${fmt(q.speed_mm_s, 0)} mm/s</span>
          ${q.coverage != null ? `<span class="chip">${Math.round(q.coverage * 100)}% traced</span>` : ''}
          ${q.template_dev_mm != null ? `<span class="chip">${fmt(q.template_dev_mm, 1)} mm off the line</span>` : ''}
          ${q.lifts ? `<span class="chip warn">${q.lifts} lift</span>` : ''}
          ${uncal ? '<span class="chip warn">uncalibrated</span>' : ''}
        </div>
      </div>
      <div class="card stage-wrap">
        <div class="row" style="justify-content:space-between">
          <h2>How the number was found</h2>
          <div class="seg" role="group" aria-label="View">
            <button type="button" id="v-spiral" aria-pressed="true">Spiral</button>
            <button type="button" id="v-wave" aria-pressed="false">Unwound</button>
            <button type="button" id="v-play" aria-label="Replay the animation">▶ Replay</button>
          </div>
        </div>
        <canvas id="res-stage" role="img" aria-label="Your traced spiral, which unwinds into a wave showing its distance from the ideal spiral over time"></canvas>
        <canvas id="res-spec" role="img" aria-label="${m.freq_hz != null ? 'Spectrum with a peak at ' + m.freq_hz.toFixed(1) + ' hertz' : 'Spectrum with no dominant peak'}"></canvas>
        <p class="tiny muted">Unwound: how your line wobbles off the best-fit spiral over time, with slow drawing drift removed. Spectrum: how much of that wobble happens at each rhythm, after removing slow drawing motion. ${m.freq_hz != null ? `The peak stands ${m.prominence > 1000 ? 'over 1000' : fmt(m.prominence, 0)}× above the band's median.` : `No dominant rhythm: a rhythm needs a peak at least ${m.rule ? m.rule.min_prominence : '?'}× above the band's median${m.prominence ? ` (best here: ${fmt(m.prominence, 1)}×)` : ''} and a size of at least ${m.rule ? m.rule.min_rhythm_pp_mm : '?'} mm, above every healthy control in the development data.`}</p>
      </div>
      ${L.kind === 'phone' ? saveBlock(L) : `<div class="card"><p>This was a public research recording, so it isn't saved. <a href="#/draw">Try your own drawing</a>.</p><div class="row">${again}</div></div>`}
      <p class="notice">${DISCLAIMER}. A measurement, never a diagnosis.</p>
      ${L.kind === 'uci' ? `<p class="attr">Data: Isenkul, Sakar &amp; Kursun (2014), UCI Machine Learning Repository dataset 395, CC BY 4.0.</p>` : ''}`;
    live.textContent = `Result: ${m.freq_hz != null ? 'peak ' + m.freq_hz.toFixed(1) + ' hertz' : 'no dominant rhythm'}, ${fmt(m.amp_pp_mm, 2)} millimetres peak to peak.`;

    const st = Charts.unwind($('res-stage'), res);
    const sp = Charts.spectrum($('res-spec'), res);
    const setView = (w) => { $('v-spiral').setAttribute('aria-pressed', String(w === 'spiral')); $('v-wave').setAttribute('aria-pressed', String(w === 'wave')); };
    const play = () => { sp.clear(); setView('spiral'); st.play((animated) => { setView(animated ? 'wave' : 'spiral'); sp.play(); }); };
    $('v-spiral').onclick = () => { st.show('spiral'); setView('spiral'); sp.show(); };
    $('v-wave').onclick = () => { st.show('wave'); setView('wave'); sp.show(); };
    $('v-play').onclick = play;
    if (!Charts.reduced()) play();
    leaveHooks.push(() => st.stop());
    if (L.kind === 'phone') wireSave(L);
    wireRaw(L);
  }

  function saveBlock(L) {
    const dose = TAGS.filter((t) => t.group === 'dose'), other = TAGS.filter((t) => !t.group);
    const btn = (t) => `<button type="button" class="tag" data-tag="${t.id}" ${t.group ? `data-group="${t.group}"` : ''} aria-pressed="false">${esc(t.label)}</button>`;
    return `<div class="card" id="save-card">
      <h2>Save to history</h2>
      <p class="muted">Optional context. Shown next to the trend; the app never claims they caused anything.</p>
      <fieldset><legend>Last dose of tremor medication</legend><div class="tags">${dose.map(btn).join('')}</div></fieldset>
      <fieldset><legend>Anything else today</legend><div class="tags">${other.map(btn).join('')}</div></fieldset>
      <div class="row"><button type="button" class="btn btn-primary btn-big" id="save-btn" ${L.saved ? 'disabled' : ''}>${L.saved ? 'Saved' : 'Save to history'}</button>
      <a class="btn btn-secondary" href="#/draw?hand=${L.capture.hand}">Draw again</a>${rawButton(L)}</div>
      <p id="save-msg" role="status" class="tiny"></p>
    </div>`;
  }

  function wireSave(L) {
    document.querySelectorAll('#save-card .tag').forEach((b) => b.addEventListener('click', () => {
      const on = b.getAttribute('aria-pressed') !== 'true';
      if (on && b.dataset.group) document.querySelectorAll(`#save-card .tag[data-group="${b.dataset.group}"]`).forEach((o) => o.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', String(on));
    }));
    $('save-btn').onclick = () => {
      const res = L.res, m = res.metrics, now = new Date();
      const pad = (n) => String(n).padStart(2, '0');
      const tags = [...document.querySelectorAll('#save-card .tag[aria-pressed="true"]')].map((b) => b.dataset.tag);
      if (!L.capture.calibrated) tags.push('uncalibrated');
      const rec = {
        id: (crypto.randomUUID ? crypto.randomUUID() : 'r' + Date.now() + Math.random().toString(16).slice(2)),
        date: `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`,
        time: `${pad(now.getHours())}:${pad(now.getMinutes())}`,
        amp_pp_mm: +m.amp_pp_mm.toFixed(4),
        freq_hz: m.freq_hz == null ? null : +m.freq_hz.toFixed(3),
        speed_mm_s: m.speed_mm_s == null ? null : +m.speed_mm_s.toFixed(2),
        tags,
        hand: L.capture.hand,
        quality: { ...res.quality, prominence: m.prominence, calibrated: L.capture.calibrated, px_per_mm: L.capture.px_per_mm },
      };
      const all = records(); all.push(rec);
      if (!store.set(RECORDS_KEY, all)) { $('save-msg').textContent = 'Could not save: this browser blocked storage (private mode?). Use "Download raw drawing" to keep it.'; return; }
      L.saved = true;
      $('save-btn').disabled = true; $('save-btn').textContent = 'Saved';
      $('save-msg').innerHTML = `Saved. <a href="#/history">See your history</a>.`;
    };
  }

  /* Raw drawing export: the schema eval/eval_phone.py reads. Field names are a contract. */
  function rawButton(L) {
    return L.kind === 'phone' && L.capture ? '<button type="button" class="btn btn-ghost" id="raw-btn">Download raw drawing</button>' : '';
  }
  function wireRaw(L) {
    const b = $('raw-btn');
    if (!b) return;
    b.onclick = () => {
      const c = L.capture, res = L.res, now = new Date(), pad = (n) => String(n).padStart(2, '0');
      const raw = {
        schema: 'spiral-raw/1',
        t_ms: c.t_ms, x_mm: c.x_mm, y_mm: c.y_mm, stroke: c.stroke,
        template: c.template,
        hand: c.hand,
        calibrated: !!c.calibrated,
        px_per_mm: c.px_per_mm,
        device: { ua: navigator.userAgent, dpr: window.devicePixelRatio || 1, coalesced: !!c.coalesced },
        status: res ? res.status : 'error',
        reasons: res ? (res.reasons || (res.message ? [{ code: 'error', message: res.message }] : [])) : [{ code: 'error', message: L.error || 'not analysed' }],
      };
      download(`spiral-raw-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.json`, JSON.stringify(raw));
    };
  }
  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  /* ------------------------------------------------------------ history */
  let histHand = null;
  async function showHistory() {
    const all = records();
    if (!histHand) {
      const lastRec = all[all.length - 1];
      histHand = lastRec ? lastRec.hand : (store.get(HAND_KEY, 'right') || 'right');
    }
    ['right', 'left'].forEach((h) => {
      const t = $('tab-' + h);
      t.setAttribute('aria-selected', String(h === histHand));
      t.tabIndex = h === histHand ? 0 : -1;
      t.textContent = `${h === 'right' ? 'Right' : 'Left'} hand (${all.filter((r) => r.hand === h).length})`;
      t.onclick = () => { histHand = h; showHistory(); };
      t.onkeydown = (e) => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { histHand = h === 'right' ? 'left' : 'right'; showHistory().then(() => $('tab-' + histHand).focus()); } };
    });
    const panel = $('hist-panel');
    $('hist-panel').setAttribute('aria-labelledby', 'tab-' + histHand);
    const recs = all.filter((r) => r.hand === histHand).sort((a, b) => (b.date + b.time).localeCompare(a.date + a.time));
    if (!recs.length) {
      panel.innerHTML = `<div class="card"><p class="empty">No saved drawings for the ${histHand} hand yet.</p><div class="row"><a class="btn btn-primary" href="#/start">Start a test</a></div></div>`;
    } else {
      panel.innerHTML = `<div class="card verdict no-print" id="verdict-card"><p class="eyebrow">Change verdict · last 7 days vs the 7 before</p><span class="state" id="verdict-state">computing</span><p class="v" id="verdict-text">Computing…</p><p class="tiny muted" id="verdict-basis"></p></div>
        <div class="card" id="print-note">
          <h2>Note for your neurologist</h2>
          <p class="print-only tiny">Spiral Test · ${esc(histHand)} hand · printed ${esc(new Date().toISOString().slice(0, 10))} · ${esc(DISCLAIMER)}</p>
          <pre class="note" id="note-text">Computing…</pre>
          <div class="row no-print"><button type="button" class="btn btn-secondary" id="note-copy">Copy</button><button type="button" class="btn btn-secondary" id="note-print">Print / save PDF</button></div>
          <p id="note-msg" role="status" class="tiny no-print"></p>
          <div id="trend-box"></div>
          <p class="tiny muted">Method: spiral traced on a phone browser; tremor rhythm from a Welch spectrum (4–12 Hz) of the 3.5–12 Hz band-passed trace; size is peak-to-peak along the principal axis; drawings failing the quality gate are never measured. Hollow dots are uncalibrated. ${esc(DISCLAIMER)}.</p>
        </div>
        <div class="card no-print"><h2>Drawings</h2><ul class="rec-list">${recs.map(recRow).join('')}</ul></div>`;
      $('trend-box').innerHTML = Charts.trend(all, null, histHand);
      $('note-copy').onclick = async () => {
        const txt = $('note-text').textContent;
        try { await navigator.clipboard.writeText(txt); $('note-msg').textContent = 'Copied.'; }
        catch (e) {
          const r = document.createRange(); r.selectNodeContents($('note-text'));
          const s = getSelection(); s.removeAllRanges(); s.addRange(r);
          $('note-msg').textContent = 'Selected: press Copy in your browser menu.';
        }
      };
      $('note-print').onclick = () => window.print();
      fillTrend(all, histHand);
    }
    $('hist-export').onclick = exportHistory;
    $('hist-import').onchange = importHistory;
    $('hist-delete').onclick = deleteAll;
  }

  function recRow(r) {
    const unc = r.quality && r.quality.calibrated === false;
    return `<li><span class="rec-when">${esc(r.date)} ${esc(r.time)}</span>
      <span class="rec-val">${Number(r.amp_pp_mm).toFixed(2)} mm · ${r.freq_hz == null ? 'no dominant rhythm' : Number(r.freq_hz).toFixed(1) + ' Hz'}${unc ? ' · <span class="pill p-warn">uncalibrated</span>' : ''}</span>
      ${(r.tags || []).filter((t) => t !== 'uncalibrated').length ? `<span class="rec-tags">${r.tags.filter((t) => t !== 'uncalibrated').map((t) => `<span class="pill p-grey">${esc(TAG_LABEL[t] || t)}</span>`).join('')}</span>` : ''}</li>`;
  }

  async function fillTrend(all, hand) {
    const vt = $('verdict-text'), vb = $('verdict-basis'), note = $('note-text');
    try {
      await Engine.ready();
      const out = await Engine.trend(all);
      if (hand !== histHand || !$('verdict-text')) return;
      if (out.error) {
        $('verdict-state').textContent = 'unavailable';
        vt.textContent = 'Trend module not built yet.';
        vb.textContent = 'spiral/trend.py was not found or failed to load: ' + out.error;
        note.textContent = 'Not available: the trend module is missing.';
        return;
      }
      const ch = out.summary.hands[hand].change;
      vt.textContent = ch.verdict.charAt(0).toUpperCase() + ch.verdict.slice(1);
      vb.textContent = ch.basis + (ch.warning ? ' ⚠ ' + ch.warning : '');
      $('verdict-state').textContent = ch.changed ? `changed · ${ch.direction}` : ch.pct == null ? 'not enough data' : 'no real change';
      const card = $('verdict-card');
      card.classList.toggle('changed-lower', ch.changed && ch.direction === 'lower');
      card.classList.toggle('changed-higher', ch.changed && ch.direction === 'higher');
      note.textContent = out.notes[hand];
      $('trend-box').innerHTML = Charts.trend(all, out.summary, hand);
    } catch (e) {
      $('verdict-state').textContent = 'unavailable';
      vt.textContent = 'Not available.';
      vb.textContent = Engine.state() === 'error' ? Engine.errorText() : 'Trend calculation failed: ' + (e.message || e);
      note.textContent = vb.textContent;
    }
  }

  function exportHistory() {
    const all = records();
    if (!all.length) { $('hist-msg').textContent = 'Nothing to export yet.'; return; }
    const d = new Date().toISOString().slice(0, 10);
    download(`spiral-history-${d}.json`, JSON.stringify({ schema: 'spiral-history/1', exported_at: new Date().toISOString(), records: all }, null, 1));
    $('hist-msg').textContent = `Exported ${all.length} drawings. The file is now outside this app: share it only with people you choose.`;
  }

  /* Import is a trust boundary: reject the whole file on the first bad record. */
  function validateRecord(r, i) {
    const where = `Record ${i + 1}`;
    if (!r || typeof r !== 'object' || Array.isArray(r)) return `${where} is not an object.`;
    if (typeof r.id !== 'string' || !r.id || r.id.length > 80) return `${where}: "id" must be a short text.`;
    const realDate = (d) => typeof d === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(d) && !isNaN(Date.parse(d + 'T00:00:00Z')) && new Date(d + 'T00:00:00Z').toISOString().slice(0, 10) === d; // rejects 2026-02-30
    if (!realDate(r.date)) return `${where}: "date" must be a real date, YYYY-MM-DD.`;
    if (typeof r.time !== 'string' || !/^([01]\d|2[0-3]):[0-5]\d$/.test(r.time)) return `${where}: "time" must be HH:MM.`;
    if (typeof r.amp_pp_mm !== 'number' || !Number.isFinite(r.amp_pp_mm) || r.amp_pp_mm <= 0 || r.amp_pp_mm > 100) return `${where}: "amp_pp_mm" must be a number between 0 and 100.`;
    if (r.freq_hz !== null && (typeof r.freq_hz !== 'number' || !Number.isFinite(r.freq_hz) || r.freq_hz <= 0 || r.freq_hz > 50)) return `${where}: "freq_hz" must be null or a number between 0 and 50.`;
    if (r.speed_mm_s !== null && r.speed_mm_s !== undefined && (typeof r.speed_mm_s !== 'number' || !Number.isFinite(r.speed_mm_s) || r.speed_mm_s < 0)) return `${where}: "speed_mm_s" must be a positive number.`;
    if (!Array.isArray(r.tags) || r.tags.length > 12 || r.tags.some((t) => typeof t !== 'string' || t.length > 32)) return `${where}: "tags" must be a list of short texts.`;
    if (r.hand !== 'right' && r.hand !== 'left') return `${where}: "hand" must be "right" or "left".`;
    if (!r.quality || typeof r.quality !== 'object' || Array.isArray(r.quality)) return `${where}: "quality" must be an object.`;
    return null;
  }
  function importHistory(ev) {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = '';
    if (!file) return;
    const msg = $('hist-msg');
    if (file.size > 5e6) { msg.textContent = `Rejected: ${file.name} is ${(file.size / 1e6).toFixed(1)} MB; a history file is far smaller. Nothing was imported.`; return; }
    const fr = new FileReader();
    fr.onerror = () => { msg.textContent = 'Rejected: the file could not be read. Nothing was imported.'; };
    fr.onload = () => {
      let data;
      try { data = JSON.parse(fr.result); } catch (e) { msg.textContent = `Rejected: ${file.name} is not valid JSON (${e.message}). Nothing was imported.`; return; }
      const list = Array.isArray(data) ? data : data && data.schema === 'spiral-history/1' && Array.isArray(data.records) ? data.records : null;
      if (!list) { msg.textContent = 'Rejected: this is not a Spiral Test history export (expected schema "spiral-history/1"). Nothing was imported.'; return; }
      if (list.length > 5000) { msg.textContent = 'Rejected: more than 5000 records. Nothing was imported.'; return; }
      for (let i = 0; i < list.length; i++) {
        const err = validateRecord(list[i], i);
        if (err) { msg.textContent = `Rejected: ${err} Nothing was imported.`; return; }
      }
      const all = records(), have = new Set(all.map((r) => r.id));
      let added = 0;
      for (const r of list) {
        if (have.has(r.id)) continue;
        all.push({ id: r.id, date: r.date, time: r.time, amp_pp_mm: r.amp_pp_mm, freq_hz: r.freq_hz, speed_mm_s: r.speed_mm_s ?? null, tags: r.tags.slice(), hand: r.hand, quality: r.quality });
        have.add(r.id); added++;
      }
      if (!store.set(RECORDS_KEY, all)) { msg.textContent = 'Could not save: storage is blocked or full. Nothing was imported.'; return; }
      msg.textContent = `Imported ${added} drawing${added === 1 ? '' : 's'}${list.length - added ? `; ${list.length - added} already present were skipped` : ''}.`;
      showHistory();
    };
    fr.readAsText(file);
  }
  function deleteAll() {
    const n = records().length;
    if (!n) { $('hist-msg').textContent = 'Nothing to delete.'; return; }
    if (!confirm(`Delete all ${n} saved drawings from this device? This cannot be undone. Export first if you want a copy.`)) return;
    store.set(RECORDS_KEY, []);
    $('hist-msg').textContent = `Deleted ${n} drawings.`;
    showHistory();
  }

  /* ------------------------------------------------------------ real recordings */
  let uciIndex = null, replayer = null, speed2x = false;
  async function showReal() {
    $('real-replay').hidden = true;
    const list = $('real-list');
    if (!uciIndex) {
      list.innerHTML = '<p class="muted">Loading recordings…</p>';
      try {
        const r = await fetch('data/uci-index.json');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        uciIndex = await r.json();
      } catch (e) {
        list.innerHTML = `<p class="banner"><b>Recordings not found.</b> data/uci-index.json didn't load (${esc(e.message)}). Run scripts/build_web.py.</p>`;
        return;
      }
    }
    list.innerHTML = uciIndex.map((u, i) => `<button type="button" class="real-item" data-i="${i}">
      <span class="pill ${u.group === 'control' ? 'p-ok' : 'p-warn'}">${u.group === 'control' ? 'Control' : 'Parkinson\'s'}</span>
      <b>${esc(u.label)}</b><span class="mono">${esc(u.subject)} · ${esc(u.test)} · ${u.n} points</span></button>`).join('');
    list.querySelectorAll('.real-item').forEach((b) => b.addEventListener('click', () => playUci(uciIndex[+b.dataset.i])));
    leaveHooks.push(() => replayer && replayer.stop());
  }
  async function playUci(meta) {
    const box = $('real-replay');
    box.hidden = false;
    $('real-title').textContent = meta.label;
    box.scrollIntoView({ behavior: Charts.reduced() ? 'auto' : 'smooth', block: 'start' });
    let rec;
    try {
      const r = await fetch('data/' + meta.file);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      rec = await r.json();
    } catch (e) { $('real-clock').textContent = 'Could not load ' + meta.file + ': ' + e.message; return; }
    // Analyse while the replay runs, so the result is ready when the pen lifts.
    const promise = Engine.analyze({ t_ms: rec.t_ms, x_mm: rec.x_mm, y_mm: rec.y_mm });
    promise.catch(() => {});
    replayer && replayer.stop();
    speed2x = false;
    $('real-speed').setAttribute('aria-pressed', 'false'); $('real-speed').textContent = 'Play at 2×';
    $('real-speed').onclick = () => { speed2x = !speed2x; $('real-speed').setAttribute('aria-pressed', String(speed2x)); $('real-speed').textContent = speed2x ? 'Playing at 2×' : 'Play at 2×'; };
    let done = false;
    replayer = Charts.replay($('real-canvas'), rec, {
      speed: () => (speed2x ? 2 : 1),
      onTime: (s, tot) => { $('real-clock').textContent = `${s.toFixed(1)} / ${tot.toFixed(1)} s · real speed${speed2x ? ' ×2' : ''}`; },
      onDone: () => {
        if (done) return; done = true;
        setTimeout(() => runAnalysis({ kind: 'uci', uci: rec, promise }), Charts.reduced() ? 0 : 500);
      },
    });
    $('real-skip').onclick = () => replayer.finish();
  }

  /* ------------------------------------------------------------ evidence */
  async function showEvidence() {
    const box = $('ev-results');
    box.innerHTML = '<p class="muted">Loading evaluation results…</p>';
    let ev = null, problem = null;
    try {
      const r = await fetch('results/eval.json', { cache: 'no-cache' });
      if (r.status === 404) problem = 'missing';
      else if (!r.ok) problem = 'HTTP ' + r.status;
      else { try { ev = await r.json(); } catch (e) { problem = 'unreadable: ' + e.message; } }
    } catch (e) { problem = e.message; }
    if (!ev) {
      box.innerHTML = `<div class="card"><h2>Failure map</h2><p class="empty">${problem === 'missing' ? 'Evaluation results not built yet.' : 'Evaluation results could not be loaded (' + esc(problem) + ').'} Nothing is shown rather than something made up. Run the evaluation, then <span class="mono">scripts/build_web.py</span>.</p></div>`;
    } else {
      box.innerHTML = renderEval(ev);
    }
    try {
      const cfg = await Engine.config();
      const g = cfg.GATE;
      $('ev-prom').textContent = cfg.MIN_PROMINENCE;
      const rows = [
        ['min_rate_hz', 'Points per second, at least', g.min_rate_hz, 'pts/s'],
        ['max_gap_ms', 'Longest pause inside a stroke, at most', g.max_gap_ms, 'ms'],
        ['min_duration_s', 'Time with the finger down, at least', g.min_duration_s, 's'],
        ['max_lifts', 'Finger lifts, at most', g.max_lifts, ''],
        ['min_coverage', 'Share of the spiral traced, at least', Math.round(g.min_coverage * 100), '%'],
        ['max_speed_mm_s', 'Drawing speed, at most', g.max_speed_mm_s, 'mm/s'],
        ['max_template_dev_mm', 'RMS distance from the printed line, at most', g.max_template_dev_mm, 'mm'],
      ];
      const known = new Set(rows.map((r) => r[0]));
      for (const k of Object.keys(g)) if (!known.has(k)) rows.push([k, k, g[k], '']);
      $('ev-gate').innerHTML = `<table><thead><tr><th>Rule</th><th>Threshold</th><th>Key</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${esc(r[1])}</td><td class="mono">${esc(r[2])} ${esc(r[3])}</td><td class="mono tiny">${esc(r[0])}</td></tr>`).join('')}</tbody></table>`;
    } catch (e) {
      $('ev-gate').innerHTML = `<p class="banner">${esc(Engine.state() === 'error' ? Engine.errorText() : 'Could not read the gate: ' + (e.message || e))}</p>`;
    }
  }
  function renderEval(ev) {
    let html = '';
    const fm = ev.failure_map;
    if (fm && Array.isArray(fm.columns) && Array.isArray(fm.rows)) {
      const cls = { measured: 'c-measured', abstains: 'c-abstains', fails: 'c-fails', untested: 'c-untested' };
      html += `<div class="card"><h2>Failure map</h2>
        <p class="muted">Each cell says what has actually been checked for that condition and group. Grey means nobody has tested it.</p>
        <div class="legend" aria-label="Legend">
          <span style="color:var(--ok)"><i class="sw c-measured"></i>measured</span>
          <span style="color:var(--warn)"><i class="sw c-abstains"></i>abstains (says Redo)</span>
          <span style="color:var(--accent)"><i class="sw c-fails"></i>fails</span>
          <span style="color:var(--muted)"><i class="sw c-untested"></i>untested</span>
        </div>
        <div class="fmap" tabindex="0" role="region" aria-label="Failure map table, scrolls sideways"><table>
        <thead><tr><th scope="col">Condition</th>${fm.columns.map((c) => `<th scope="col">${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${fm.rows.map((r) => `<tr><th scope="row">${esc(r.condition)}</th>${(r.cells || []).map((c) => `<td class="${cls[c.status] || 'c-untested'}"><span class="sr-only">${esc(c.status)}: </span>${esc(c.text || c.status)}${c.n != null ? `<span class="n">n=${esc(c.n)}</span>` : ''}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>${fm.note ? `<p class="tiny muted">${esc(fm.note)}</p>` : ''}</div>`;
    } else {
      html += '<div class="card"><h2>Failure map</h2><p class="empty">The evaluation file has no failure map yet.</p></div>';
    }
    if (Array.isArray(ev.headline) && ev.headline.length) {
      html += `<div class="card"><h2>Headline results</h2><ul class="claims">${ev.headline.map((h) => `<li><span>${esc(h.text || h.claim || "")}</span>
        <span class="meta">${h.n != null ? 'n=' + esc(h.n) : ''}${h.split ? ' · ' + esc(h.split) + ' split' : ''}</span>
        ${h.caveat ? `<span class="tiny muted">Caveat: ${esc(h.caveat)}</span>` : ''}</li>`).join('')}</ul></div>`;
    }
    return html;
  }

  /* ------------------------------------------------------------ boot */
  Engine.start(); // background: ~10 MB on first visit, cached after
  route();
})();
