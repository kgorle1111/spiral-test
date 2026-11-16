/* Drawing: hero illustration, the "spiral unwinds into a wave, then a spectrum
   peak" moment, UCI replay, and the History trend chart. Colours come from CSS
   tokens at draw time so light and dark mode both work. */
'use strict';
window.Charts = (() => {
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* Content-box width of the canvas's container (bounding width includes padding). */
  function innerWidth(el) {
    const cs = getComputedStyle(el);
    return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }

  function sizeCanvas(cnv, cssW, cssH) {
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    cnv.style.width = cssW + 'px';
    cnv.style.height = cssH + 'px';
    cnv.width = Math.round(cssW * dpr);
    cnv.height = Math.round(cssH * dpr);
    const c = cnv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    return c;
  }

  /* ---------------------------------------------------------- hero (SVG, synthetic) */
  function hero(el) {
    const W = 520, H = 300;
    const sp = [], wave = [], spec = [];
    const N = 700, turns = 3, f = 6, amp = 2.4;
    const cx = 110, cy = 112;
    for (let i = 0; i < N; i++) {
      const u = i / (N - 1), th = u * turns * 2 * Math.PI;
      const d = amp * Math.sin(2 * Math.PI * f * u * 6) * (0.6 + 0.4 * Math.sin(u * 5));
      const r = 6 + th * 4.6 + d;
      sp.push([cx + r * Math.cos(-th), cy + r * Math.sin(-th)]);
      wave.push([250 + u * 250, 82 - d * 4.2]);
    }
    for (let k = 0; k <= 120; k++) {
      const hz = (k / 120) * 15;
      const p = Math.exp(-((hz - 6) ** 2) / 0.18) + 0.05 * Math.exp(-hz / 3) + 0.015;
      spec.push([250 + (hz / 15) * 250, 262 - p * 92]);
    }
    const path = (pts) => 'M' + pts.map((p) => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
    const ideal = [];
    for (let i = 0; i <= 300; i++) { const th = (i / 300) * turns * 2 * Math.PI, r = 6 + th * 4.6; ideal.push([cx + r * Math.cos(-th), cy + r * Math.sin(-th)]); }
    el.innerHTML = `
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="A spiral with tremor wobble, its unwrapped wave, and a spectrum with one peak at 6 hertz">
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="${cx}" cy="${cy}" r="40" stroke="var(--grid)"/><circle cx="${cx}" cy="${cy}" r="80" stroke="var(--grid)"/>
    <path d="${path(ideal)}" stroke="var(--ideal)" stroke-width="1.5" stroke-dasharray="4 5"/>
    <path class="hero-draw" d="${path(sp)}" stroke="var(--ink)" stroke-width="2" pathLength="1"/>
    <path d="M232 112 C 240 112 240 82 248 82" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#ah)"/>
    <line x1="250" y1="82" x2="500" y2="82" stroke="var(--grid)"/>
    <path class="hero-draw d2" d="${path(wave)}" stroke="var(--accent)" stroke-width="1.8" pathLength="1"/>
    <path d="M470 120 L470 160" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#ah)"/>
    <line x1="250" y1="262" x2="500" y2="262" stroke="var(--line)"/>
    <path class="hero-draw d3" d="${path(spec)}" stroke="var(--ink)" stroke-width="2" pathLength="1"/>
    <circle cx="${250 + (6 / 15) * 250}" cy="${262 - 1.06 * 92}" r="5" fill="var(--accent)" stroke="none" class="hero-pop"/>
  </g>
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="var(--muted)"/></marker></defs>
  <g font-family="IBM Plex Mono, monospace" font-size="12" fill="var(--muted)">
    <text x="250" y="40">unwound: off-line distance</text>
    <text x="250" y="190">spectrum</text>
    <text x="${250 + (6 / 15) * 250 + 9}" y="${262 - 1.06 * 92 + 4}" fill="var(--accent)" font-weight="500">6 Hz</text>
    <text x="250" y="282">0</text><text x="${250 + (5 / 15) * 250 - 4}" y="282">5</text><text x="${250 + (10 / 15) * 250 - 7}" y="282">10</text><text x="476" y="282">15 Hz</text>
    <text x="40" y="282">your spiral</text>
  </g>
  <style>
    .hero-draw{stroke-dasharray:1;stroke-dashoffset:0;animation:hd 2.2s ease-out both}
    .hero-draw.d2{animation-delay:1.4s}.hero-draw.d3{animation-delay:2.4s}
    .hero-pop{animation:hp .4s 3.8s both}
    @keyframes hd{from{stroke-dashoffset:1}}
    @keyframes hp{from{opacity:0}}
  </style>
</svg>`;
  }

  /* ---------------------------------------------------------- unwind moment
     The user's own trace in polar form (plot_r, theta) morphs point by point into
     (time, radial residual). Both come from Python; nothing is synthesised. */
  function unwind(cnv, res, opts = {}) {
    const s = res.series;
    const th = s.theta, rad = s.radial_bp || s.radial, rr = res.plot_r; // band-passed: the tremor, drift removed
    const n = Math.min(th.length, rad.length, rr.length);
    const W = Math.max(240, Math.floor(innerWidth(cnv.parentElement)));
    const H = Math.round(Math.min(W * 0.78, 420));
    const c = sizeCanvas(cnv, W, H);
    let rMax = 0, radMax = 1e-9;
    for (let i = 0; i < n; i++) { rMax = Math.max(rMax, rr[i] || 0); radMax = Math.max(radMax, Math.abs(rad[i] || 0)); }
    const pxmmS = (Math.min(W, H) / 2 - 14) / Math.max(rMax, 1e-6);
    const cx = W / 2, cy = H / 2;
    const padL = 44, padR = 14; // left margin holds the mm scale bar
    const waveScale = (H * 0.36) / radMax; // px per mm, vertical
    const spiralPts = new Array(n), wavePts = new Array(n);
    for (let i = 0; i < n; i++) {
      const r = (rr[i] || 0) * pxmmS;
      spiralPts[i] = [cx + r * Math.cos(th[i]), cy - r * Math.sin(th[i])];
      wavePts[i] = [padL + (i / Math.max(1, n - 1)) * (W - padL - padR), cy - (rad[i] || 0) * waveScale];
    }
    const span = res.span_s || n / 100;
    const ink = css('--ink'), acc = css('--accent'), grid = css('--grid'), mut = css('--muted');

    function frame(drawFrac, morph) {
      c.clearRect(0, 0, W, H);
      const m = ease(morph);
      // fading polar paper behind the spiral, fading time axis behind the wave
      c.globalAlpha = 1 - m; c.strokeStyle = grid; c.lineWidth = 1;
      for (let k = 1; k <= 4; k++) { c.beginPath(); c.arc(cx, cy, (k / 4) * (Math.min(W, H) / 2 - 14), 0, 2 * Math.PI); c.stroke(); }
      c.globalAlpha = m;
      c.beginPath(); c.moveTo(padL, cy); c.lineTo(W - padR, cy); c.stroke();
      c.globalAlpha = 1;
      const upto = Math.max(2, Math.floor(drawFrac * n));
      c.lineWidth = 1.8; c.lineJoin = 'round'; c.lineCap = 'round';
      c.strokeStyle = m > 0.5 ? acc : ink;
      c.beginPath();
      for (let i = 0; i < upto; i++) {
        const p = spiralPts[i], q = wavePts[i];
        const X = p[0] + (q[0] - p[0]) * m, Y = p[1] + (q[1] - p[1]) * m;
        i ? c.lineTo(X, Y) : c.moveTo(X, Y);
      }
      c.stroke();
      c.font = '500 12px "IBM Plex Mono", monospace'; c.fillStyle = mut;
      if (m > 0.98) {
        c.textAlign = 'left'; c.fillText(s.radial_bp ? 'wobble off the spiral, slow drift removed' : 'distance from best-fit spiral', 8, 16);
        c.fillText('0 s', padL, H - 8);
        c.textAlign = 'right'; c.fillText(span.toFixed(1) + ' s', W - padR, H - 8);
        // scale bar: 1 mm, or a smaller round value when the wobble is tiny
        const mm = radMax >= 1 ? 1 : radMax >= 0.2 ? 0.2 : 0.05;
        const hpx = mm * waveScale;
        c.strokeStyle = ink; c.lineWidth = 2;
        c.beginPath(); c.moveTo(14, cy - hpx / 2); c.lineTo(14, cy + hpx / 2); c.stroke();
        c.textAlign = 'left'; c.font = '500 10px "IBM Plex Mono", monospace'; c.fillText(mm + ' mm', 2, cy + hpx / 2 + 13);
      } else if (m < 0.02) {
        c.textAlign = 'left'; c.fillText('your trace', padL, 16);
      }
    }

    let rafId = 0;
    function stop() { cancelAnimationFrame(rafId); }
    function show(which) { stop(); frame(1, which === 'wave' ? 1 : 0); }
    function play(done) {
      stop();
      if (reduced()) { frame(1, 0); done && done(false); return; }
      const T_DRAW = 1400, T_HOLD = 450, T_MORPH = 1700;
      const t0 = performance.now();
      const tick = (now) => {
        const e = now - t0;
        if (e < T_DRAW) frame(e / T_DRAW, 0);
        else if (e < T_DRAW + T_HOLD) frame(1, 0);
        else if (e < T_DRAW + T_HOLD + T_MORPH) frame(1, (e - T_DRAW - T_HOLD) / T_MORPH);
        else { frame(1, 1); done && done(true); return; }
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    }
    frame(1, 0);
    return { play, show, stop };
  }

  /* ---------------------------------------------------------- spectrum */
  function spectrum(cnv, res, opts = {}) {
    const f = res.series.freqs, p = res.series.psd;
    const peak = res.metrics ? res.metrics.freq_hz : null;
    const W = Math.max(240, Math.floor(innerWidth(cnv.parentElement)));
    const H = Math.round(Math.min(W * 0.42, 220));
    const c = sizeCanvas(cnv, W, H);
    const L = 12, R = 12, T = 22, B = 26, FMAX = 15;
    const X = (hz) => L + (hz / FMAX) * (W - L - R);
    let pmax = 1e-12;
    for (let i = 0; i < f.length; i++) if (f[i] >= 2 && f[i] <= FMAX && p[i] != null) pmax = Math.max(pmax, p[i]);
    const Y = (v) => H - B - Math.min(1, (v || 0) / pmax) * (H - B - T);
    const band = opts.band || [4, 12];
    const ink = css('--ink'), acc = css('--accent'), mut = css('--muted'), line = css('--line'), grey = css('--grey-cell');

    function frame(frac, showPeak) {
      c.clearRect(0, 0, W, H);
      c.fillStyle = grey; c.fillRect(X(band[0]), T - 6, X(band[1]) - X(band[0]), H - B - T + 6);
      c.font = '500 11px "IBM Plex Mono", monospace'; c.fillStyle = mut; c.textAlign = 'left';
      // Left of the shaded band: a peak is always inside the band, so this never collides.
      c.textAlign = 'left';
      c.fillText('shaded:', L, T + 6);
      c.fillText(band[0] + '–' + band[1] + ' Hz', L, T + 20);
      c.strokeStyle = line; c.lineWidth = 1;
      c.beginPath(); c.moveTo(L, H - B); c.lineTo(W - R, H - B); c.stroke();
      c.textAlign = 'center';
      for (let hz = 0; hz <= FMAX; hz += 3) c.fillText(hz + (hz === FMAX ? ' Hz' : ''), Math.min(X(hz), W - R - 14), H - 8);
      const lim = FMAX * frac;
      c.strokeStyle = ink; c.lineWidth = 2; c.lineJoin = 'round';
      c.beginPath();
      let started = false;
      for (let i = 0; i < f.length && f[i] <= lim; i++) {
        const xx = X(f[i]), yy = Y(p[i]);
        started ? c.lineTo(xx, yy) : (c.moveTo(xx, yy), (started = true));
      }
      c.stroke();
      if (showPeak) {
        if (peak != null) {
          let best = 0, bv = -1;
          for (let i = 0; i < f.length; i++) if (Math.abs(f[i] - peak) <= 0.3 && (p[i] || 0) > bv) { bv = p[i]; best = i; }
          const px = X(peak), py = Y(p[best]);
          c.strokeStyle = acc; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(px, py); c.lineTo(px, H - B); c.stroke(); c.setLineDash([]);
          c.fillStyle = acc; c.beginPath(); c.arc(px, py, 5.5, 0, 2 * Math.PI); c.fill();
          c.font = '600 13px "IBM Plex Mono", monospace'; c.textAlign = px > W * 0.7 ? 'right' : 'left';
          c.fillText(peak.toFixed(1) + ' Hz', px + (px > W * 0.7 ? -9 : 9), Math.max(T + 18, py + 4));
        } else {
          c.font = '500 12px "IBM Plex Mono", monospace'; c.fillStyle = mut; c.textAlign = 'right';
          c.fillText('no dominant rhythm', W - R, T + 22);
        }
      }
    }
    let rafId = 0;
    function play(done) {
      cancelAnimationFrame(rafId);
      if (reduced()) { frame(1, true); done && done(); return; }
      const t0 = performance.now(), T = 900;
      const tick = (now) => {
        const e = (now - t0) / T;
        if (e < 1) { frame(e, false); rafId = requestAnimationFrame(tick); }
        else { frame(1, true); done && done(); }
      };
      rafId = requestAnimationFrame(tick);
    }
    frame(1, true);
    return { play, show: () => frame(1, true), clear: () => c.clearRect(0, 0, W, H) };
  }

  /* ---------------------------------------------------------- UCI replay */
  function replay(cnv, rec, opts) {
    const box = cnv.getBoundingClientRect();
    const S = Math.max(240, Math.floor(Math.min(box.width, window.innerHeight * 0.6)));
    const c = sizeCanvas(cnv, S, S);
    const xs = rec.x_mm, ys = rec.y_mm, ts = rec.t_ms;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < xs.length; i++) { x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]); }
    const pad = 16, k = (S - 2 * pad) / Math.max(x1 - x0, y1 - y0, 1e-6);
    const ox = pad + ((S - 2 * pad) - (x1 - x0) * k) / 2, oy = pad + ((S - 2 * pad) - (y1 - y0) * k) / 2;
    const P = (i) => [ox + (xs[i] - x0) * k, oy + (y1 - ys[i]) * k]; // y_mm is y-up; screen is y-down
    const ink = css('--ink'), grid = css('--grid'), mut = css('--muted');
    let i = 0, rafId = 0, last = null, simT = 0, stopped = false;
    const dur = ts[ts.length - 1] - ts[0];
    c.strokeStyle = grid; c.lineWidth = 1;
    // 10 mm grid so the scale is visible
    for (let gx = Math.ceil(x0 / 10) * 10; gx <= x1; gx += 10) { const X = ox + (gx - x0) * k; c.beginPath(); c.moveTo(X, 0); c.lineTo(X, S); c.stroke(); }
    for (let gy = Math.ceil(y0 / 10) * 10; gy <= y1; gy += 10) { const Y = oy + (y1 - gy) * k; c.beginPath(); c.moveTo(0, Y); c.lineTo(S, Y); c.stroke(); }
    c.fillStyle = mut; c.font = '500 11px "IBM Plex Mono", monospace'; c.fillText('grid 10 mm (assumed scale)', 8, S - 8);
    c.strokeStyle = ink; c.lineWidth = 1.6; c.lineJoin = 'round'; c.lineCap = 'round';
    function drawTo(tTarget) {
      c.beginPath();
      let moved = false;
      while (i < xs.length && ts[i] - ts[0] <= tTarget) {
        const p = P(i);
        if (!moved) { const q = i ? P(i - 1) : p; c.moveTo(q[0], q[1]); moved = true; }
        c.lineTo(p[0], p[1]);
        i++;
      }
      c.stroke();
    }
    function tick(now) {
      if (stopped) return;
      if (last !== null) simT += (now - last) * opts.speed();
      last = now;
      drawTo(simT);
      opts.onTime && opts.onTime(Math.min(simT, dur) / 1000, dur / 1000);
      if (i >= xs.length) { opts.onDone && opts.onDone(); return; }
      rafId = requestAnimationFrame(tick);
    }
    if (reduced()) { drawTo(Infinity); opts.onTime && opts.onTime(dur / 1000, dur / 1000); setTimeout(() => opts.onDone && opts.onDone(), 0); }
    else rafId = requestAnimationFrame(tick);
    return {
      finish() { stopped = true; cancelAnimationFrame(rafId); drawTo(Infinity); opts.onTime && opts.onTime(dur / 1000, dur / 1000); opts.onDone && opts.onDone(); },
      stop() { stopped = true; cancelAnimationFrame(rafId); },
    };
  }

  /* ---------------------------------------------------------- trend (SVG) */
  function trend(records, summary, hand) {
    const recs = records.filter((r) => r.hand === hand && r.amp_pp_mm > 0).sort((a, b) => (a.date + a.time).localeCompare(b.date + b.time));
    if (!recs.length) return '';
    const W = 640, H = 260, L = 46, R = 14, T = 16, B = 34;
    const day = (s) => Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10)) / 864e5;
    const hb = summary && summary.hands ? summary.hands[hand] : null;
    let d0 = day(recs[0].date), d1 = day(recs[recs.length - 1].date);
    if (hb) { d0 = Math.min(d0, day(hb.baseline.start)); d1 = Math.max(d1, day(hb.current.end)); }
    if (d1 - d0 < 6) d0 = d1 - 6;
    const X = (d) => L + ((d - d0 + 0.5) / (d1 - d0 + 1)) * (W - L - R);
    const amps = recs.map((r) => r.amp_pp_mm);
    // Log axis: tremor amplitude is roughly log-normal and spans decades.
    let lo = Math.min(...amps), hi = Math.max(...amps);
    lo = Math.pow(10, Math.floor(Math.log10(lo) * 2) / 2); hi = Math.pow(10, Math.ceil(Math.log10(hi) * 2) / 2);
    if (hi / lo < 4) { lo /= 2; hi *= 2; }
    const Y = (v) => T + (1 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - T - B);
    let g = '';
    if (hb) {
      const wk = [['baseline', 'earlier week', 'var(--grey-cell)'], ['current', 'latest week', 'var(--accent-soft)']];
      for (const [k, label, fill] of wk) {
        const w = hb[k];
        const xa = X(day(w.start)) - (W - L - R) / (d1 - d0 + 1) / 2, xb = X(day(w.end)) + (W - L - R) / (d1 - d0 + 1) / 2;
        g += `<rect x="${xa.toFixed(1)}" y="${T}" width="${(xb - xa).toFixed(1)}" height="${H - T - B}" fill="${fill}" opacity=".8"/>`;
        g += `<text x="${(xa + 4).toFixed(1)}" y="${T + 13}" class="lbl">${label}</text>`;
        if (w.mean_mm) g += `<line x1="${xa.toFixed(1)}" x2="${xb.toFixed(1)}" y1="${Y(w.mean_mm).toFixed(1)}" y2="${Y(w.mean_mm).toFixed(1)}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      }
    }
    const ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20].filter((v) => v >= lo && v <= hi);
    for (const v of ticks) g += `<line x1="${L}" x2="${W - R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="var(--line)" stroke-width=".7"/><text x="${L - 6}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end" class="lbl">${v}</text>`;
    g += `<text x="6" y="${T + 4}" class="lbl">mm</text>`;
    const labelEvery = Math.ceil((d1 - d0 + 1) / 7);
    for (let d = d0; d <= d1; d++) {
      if ((d - d0) % labelEvery) continue;
      const dt = new Date(d * 864e5);
      g += `<text x="${X(d).toFixed(1)}" y="${H - 12}" text-anchor="middle" class="lbl">${dt.getUTCDate()}/${dt.getUTCMonth() + 1}</text>`;
    }
    for (const r of recs) {
      const unc = r.quality && r.quality.calibrated === false;
      g += `<circle cx="${X(day(r.date)).toFixed(1)}" cy="${Y(r.amp_pp_mm).toFixed(1)}" r="4.5" fill="${unc ? 'var(--sheet)' : 'var(--accent)'}" stroke="var(--accent)" stroke-width="1.8"><title>${esc(r.date + ' ' + r.time)}: ${r.amp_pp_mm.toFixed(2)} mm${unc ? ' (uncalibrated)' : ''}</title></circle>`;
    }
    return `<svg class="trend-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Tremor amplitude per drawing over time, ${hand} hand, log scale">
<style>.lbl{font:500 11px 'IBM Plex Mono',monospace;fill:var(--muted)}</style>${g}</svg>`;
  }

  return { hero, unwind, spectrum, replay, trend, esc, css, reduced };
})();
