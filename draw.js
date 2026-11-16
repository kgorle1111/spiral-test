/* Calibration, the printed spiral template, and pointer capture.

   Coordinate frames
   - CSS px: what the browser reports (clientX/Y), y grows downward.
   - mm: x_mm = px / pxPerMm, y_mm = -px / pxPerMm. y is flipped so angles run the
     math way (counter-clockwise positive), the same convention as the UCI loader.
   The template dict handed to metrics.py is in the mm frame:
     arm point at template angle theta (0..2*pi*turns):
       r = a + b*theta, math angle = phi0 + dir*theta
   which is exactly what metrics.template_deviation inverts
   (alpha = dir*(angle - phi0) mod 2*pi). Template.selfCheck() verifies it numerically. */
'use strict';

window.Calibration = (() => {
  const KEY = 'spiral.cal.v1';
  const CARD_W_MM = 53.98; // ID-1 short edge; held upright, this edge spans the screen
  const CARD_H_MM = 85.6;
  const CARD_R_MM = 3.18;
  const BROWSER_GUESS = 96 / 25.4; // CSS px per mm per the CSS spec; phones don't honour it

  function load() {
    try {
      const v = JSON.parse(localStorage.getItem(KEY) || 'null');
      if (v && Number.isFinite(v.px_per_mm) && v.px_per_mm > 1 && v.px_per_mm < 20) return v;
    } catch (e) { /* unreadable storage: treat as not calibrated */ }
    return null;
  }
  function save(pxPerMm, calibrated) {
    const v = { px_per_mm: +pxPerMm.toFixed(4), calibrated, date: new Date().toISOString() };
    try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* private mode: lives for this page only */ }
    current = v;
    return v;
  }
  let current = load();
  return {
    CARD_W_MM, CARD_H_MM, CARD_R_MM, BROWSER_GUESS,
    get: () => current,
    save,
    initialGuess: () => (current ? current.px_per_mm
      : (matchMedia('(pointer: coarse)').matches ? 6.2 : BROWSER_GUESS)),
  };
})();

window.Template = (() => {
  const TURNS = 3;
  const A_MM = 4;
  // Same physical size on laptops as on phones: path ~320 mm, so a 15 s trace is ~21 mm/s,
  // comfortably under metrics.GATE max_speed_mm_s (40).
  const MAX_R_MM = 30;

  /* Build the template for a square canvas of side `sidePx` CSS px. */
  function make(sidePx, pxPerMm, hand) {
    const marginPx = Math.max(14, sidePx * 0.045);
    let rPx = sidePx / 2 - marginPx;
    let rMm = rPx / pxPerMm;
    if (rMm > MAX_R_MM) { rMm = MAX_R_MM; rPx = rMm * pxPerMm; }
    const b = (rMm - A_MM) / (2 * Math.PI * TURNS);
    const right = hand !== 'left';
    // Right hand: clockwise on screen (dir -1 in the y-up frame), outer end on the right.
    // Left hand: mirrored, counter-clockwise, outer end on the left.
    const dir = right ? -1 : 1;
    const phi0 = right ? 0 : Math.PI;
    const cxPx = sidePx / 2;
    const cyPx = sidePx / 2;
    return {
      sidePx, pxPerMm, cxPx, cyPx, rMm,
      dict: { cx: cxPx / pxPerMm, cy: -cyPx / pxPerMm, a: A_MM, b, turns: TURNS, phi0, dir },
    };
  }

  /* Point on the printed spiral at template angle theta, in CSS px. */
  function pointPx(t, theta) {
    const d = t.dict;
    const r = (d.a + d.b * theta) * t.pxPerMm;
    const ang = d.phi0 + d.dir * theta;
    return [t.cxPx + r * Math.cos(ang), t.cyPx - r * Math.sin(ang)];
  }
  const toMm = (t, xPx, yPx) => [xPx / t.pxPerMm, -yPx / t.pxPerMm];

  /* JS mirror of metrics.template_deviation, for the self-check and the live
     auto-finish logic. The authoritative number always comes from Python. */
  function deviationMm(t, xMm, yMm) {
    const d = t.dict;
    const ang = Math.atan2(yMm - d.cy, xMm - d.cx);
    let alpha = (d.dir * (ang - d.phi0)) % (2 * Math.PI);
    if (alpha < 0) alpha += 2 * Math.PI;
    const r = Math.hypot(xMm - d.cx, yMm - d.cy);
    let best = Infinity;
    for (let k = 0; k <= Math.ceil(d.turns); k++) best = Math.min(best, Math.abs(r - (d.a + d.b * (alpha + 2 * Math.PI * k))));
    return best;
  }

  /* Sample the template as if traced, for verification. */
  function samples(t, n = 1200) {
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const th = (i / (n - 1)) * 2 * Math.PI * t.dict.turns;
      const [px, py] = pointPx(t, th);
      const [x, y] = toMm(t, px, py);
      xs.push(x); ys.push(y);
    }
    return { xs, ys };
  }

  function selfCheck(t) {
    const { xs, ys } = samples(t);
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += deviationMm(t, xs[i], ys[i]) ** 2;
    return Math.sqrt(s / xs.length);
  }

  return { TURNS, A_MM, make, pointPx, toMm, deviationMm, samples, selfCheck };
})();

window.Capture = (() => {
  let canvas, ctx, tpl, dpr = 1;
  let t = [], x = [], y = [], stroke = [];
  let strokeIdx = -1, activeId = null, rect = null, finished = false;
  let coalescedSeen = false;
  let unwrapped = 0, lastAng = null; // template-angle progress, for auto-finish
  let onUpdate = () => {}, onFinish = () => {};
  let queue = []; // segments waiting for the next animation frame
  let raf = 0;

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function setup(cnv, template, hooks) {
    canvas = cnv; tpl = template;
    onUpdate = hooks.onUpdate || onUpdate;
    onFinish = hooks.onFinish || onFinish;
    dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.style.width = tpl.sidePx + 'px';
    canvas.style.height = tpl.sidePx + 'px';
    canvas.width = Math.round(tpl.sidePx * dpr);
    canvas.height = Math.round(tpl.sidePx * dpr);
    ctx = canvas.getContext('2d');
    reset();
    if (!canvas._wired) wire();
  }

  function reset() {
    t = []; x = []; y = []; stroke = [];
    strokeIdx = -1; activeId = null; finished = false; coalescedSeen = false;
    unwrapped = 0; lastAng = null; queue = [];
    paintTemplate();
    onUpdate(stats());
  }

  function paintTemplate() {
    const s = tpl.sidePx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, s, s);
    // polar graph paper: rings every 10 mm
    ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1;
    for (let r = 10; r * tpl.pxPerMm < s / 2; r += 10) {
      ctx.beginPath(); ctx.arc(tpl.cxPx, tpl.cyPx, r * tpl.pxPerMm, 0, 2 * Math.PI); ctx.stroke();
    }
    // the printed spiral
    ctx.strokeStyle = css('--ideal'); ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath();
    const n = 900;
    for (let i = 0; i <= n; i++) {
      const [px, py] = Template.pointPx(tpl, (i / n) * 2 * Math.PI * tpl.dict.turns);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    // start dot and end tick
    const [sx, sy] = Template.pointPx(tpl, 0);
    ctx.fillStyle = css('--accent');
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, 2 * Math.PI); ctx.fill();
    const [ex, ey] = Template.pointPx(tpl, 2 * Math.PI * tpl.dict.turns);
    ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(ex, ey, 9, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = '500 12px "IBM Plex Mono", monospace'; ctx.fillStyle = css('--muted');
    ctx.textAlign = tpl.dict.dir < 0 ? 'right' : 'left';
    ctx.fillText('end', ex + (tpl.dict.dir < 0 ? -14 : 14), ey + 4);
  }

  function addPoint(ev) {
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const [mx, my] = Template.toMm(tpl, px, py);
    t.push(ev.timeStamp); x.push(mx); y.push(my); stroke.push(strokeIdx);
    queue.push([px, py, strokeIdx]);
    // progress along the template angle (unwrapped, in the template's direction)
    const d = tpl.dict;
    const ang = Math.atan2(my - d.cy, mx - d.cx);
    if (lastAng !== null) {
      let da = ang - lastAng;
      if (da > Math.PI) da -= 2 * Math.PI;
      if (da < -Math.PI) da += 2 * Math.PI;
      unwrapped += d.dir * da;
    }
    lastAng = ang;
  }

  function flush() {
    raf = 0;
    if (!queue.length) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2.2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath();
    let prev = flush.prev;
    for (const p of queue) {
      if (prev && prev[2] === p[2]) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
      prev = p;
    }
    ctx.stroke();
    flush.prev = prev;
    queue = [];
    onUpdate(stats());
  }
  const schedule = () => { if (!raf) raf = requestAnimationFrame(flush); };

  function stats() {
    const n = t.length;
    let rate = null;
    if (n > 5) {
      const tEnd = t[n - 1];
      let k = n - 1;
      while (k > 0 && tEnd - t[k - 1] <= 1000) k--;
      const span = tEnd - t[k];
      if (span > 250) rate = (n - 1 - k) / (span / 1000);
    }
    return { n, elapsed: n ? (t[n - 1] - t[0]) / 1000 : 0, rate, lifts: Math.max(0, strokeIdx), progress: unwrapped / (2 * Math.PI * tpl.dict.turns) };
  }

  function nearEnd() {
    const n = x.length;
    if (!n) return false;
    const d = tpl.dict;
    const eTheta = 2 * Math.PI * d.turns;
    const er = d.a + d.b * eTheta;
    const ex = d.cx + er * Math.cos(d.phi0 + d.dir * eTheta);
    const ey = d.cy + er * Math.sin(d.phi0 + d.dir * eTheta);
    return unwrapped >= 0.95 * eTheta && Math.hypot(x[n - 1] - ex, y[n - 1] - ey) < 6;
  }

  function wire() {
    canvas._wired = true;
    canvas.addEventListener('pointerdown', (e) => {
      if (finished || !e.isPrimary) return;
      e.preventDefault();
      activeId = e.pointerId;
      try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events have no capture target */ }
      rect = canvas.getBoundingClientRect();
      strokeIdx += 1;
      flush.prev = null;
      addPoint(e);
      schedule();
    });
    canvas.addEventListener('pointermove', (e) => {
      if (finished || e.pointerId !== activeId) return;
      e.preventDefault();
      // Coalesced events carry the points the browser merged between frames,
      // each with its own timestamp. Without them the rate drops to the frame rate.
      const list = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
      if (list.length) coalescedSeen = true;
      for (const ev of (list.length ? list : [e])) addPoint(ev);
      schedule();
      if (nearEnd()) finish();
    });
    const end = (e) => {
      if (e.pointerId !== activeId) return;
      activeId = null;
      schedule();
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    // iOS Safari still scrolls on some touch sequences despite touch-action:none.
    canvas.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  function finish() {
    if (finished) return;
    finished = true;
    activeId = null;
    flush();
    onFinish(data());
  }

  function data() {
    return { t_ms: t.slice(), x_mm: x.slice(), y_mm: y.slice(), stroke: stroke.slice(), template: { ...tpl.dict }, coalesced: coalescedSeen };
  }

  return { setup, reset, finish, data, stats, repaint: () => { paintTemplate(); } };
})();
