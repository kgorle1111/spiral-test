/* Analysis engine: Pyodide + numpy in a Web Worker.
   A worker, not the main thread, because compiling ~10 MB of WebAssembly would
   stall pointer events while someone is drawing, and stalls become "gap" redos.
   Runs web/py/metrics.py and web/py/trend.py, byte-identical copies of spiral/. */
'use strict';
const PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';

// Python glue. Kept here (not in spiral/) because it is web-only plumbing:
// JSON in, JSON out, numpy scalars -> Python scalars, NaN/inf -> null.
const GLUE = `
import json, math, sys
sys.path.insert(0, "/home/pyodide/app")
import numpy as np
import metrics

try:
    import trend
    TREND_ERROR = None
except Exception as e:  # trend.py missing or broken: History says so, analysis still works
    trend = None
    TREND_ERROR = repr(e)


def _j(o):
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def analyze_json(payload):
    d = json.loads(payload)
    try:
        r = metrics.analyze(d["t_ms"], d["x_mm"], d["y_mm"], stroke=d.get("stroke"), template=d.get("template"))
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e) or type(e).__name__})
    theta = np.asarray(r["series"]["theta"])
    step = max(1, theta.size // 600)  # same thinning as metrics.series_for_plot
    # Radius of the resampled trace around the analysis centre, for drawing the
    # user's real spiral (r, theta) before it unwinds into (t, radial).
    t, x, y, s = metrics.clean(d["t_ms"], d["x_mm"], d["y_mm"], d.get("stroke"))
    _, xg, yg = metrics.resample(t, x, y)
    cx, cy = r["center"]
    rr = np.hypot(xg - cx, yg - cy)
    r["plot_r"] = [round(float(v), 4) for v in rr[::step]]
    r["span_s"] = float(theta.size / metrics.FS)
    r["series"] = metrics.series_for_plot(r["series"])
    return json.dumps(_j(r), allow_nan=False)


def config_json():
    return json.dumps(_j({
        "GATE": metrics.GATE, "MIN_PROMINENCE": metrics.MIN_PROMINENCE, "BAND": metrics.BAND,
        "FS": metrics.FS, "HP": metrics.HP, "LP": metrics.LP, "trend_error": TREND_ERROR,
    }))


def summarize_json(records_json, today):
    if trend is None:
        return json.dumps({"error": TREND_ERROR})
    recs = json.loads(records_json)
    out = {"summary": trend.summarize(recs, today or None), "notes": {}}
    for h in ("right", "left"):
        out["notes"][h] = trend.clinician_summary(recs, today or None, hand=h)
    return json.dumps(_j(out), allow_nan=False)
`;

let py = null;
const status = (state, detail) => postMessage({ type: 'status', state, detail });

const boot = (async () => {
  status('loading', 'downloading Python');
  importScripts(PYODIDE_BASE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE_BASE });
  status('loading', 'loading numpy');
  await py.loadPackage('numpy');
  status('loading', 'loading analysis code');
  py.FS.mkdirTree('/home/pyodide/app');
  for (const name of ['metrics.py', 'trend.py']) {
    const res = await fetch('py/' + name, { cache: 'no-cache' });
    if (res.ok) py.FS.writeFile('/home/pyodide/app/' + name, await res.text());
    else if (name === 'metrics.py') throw new Error('py/metrics.py could not be fetched (HTTP ' + res.status + ')');
    // TODO(trend): if py/trend.py is absent, History shows "trend module not built yet".
  }
  py.runPython(GLUE);
  status('ready', '');
})().catch((e) => {
  status('error', String((e && e.message) || e));
  throw e;
});

onmessage = async (ev) => {
  const { id, fn, args } = ev.data;
  try {
    await boot;
    const f = py.globals.get(fn);
    if (!f) throw new Error('unknown engine function ' + fn);
    const out = f(...(args || []));
    f.destroy && f.destroy();
    postMessage({ id, ok: true, result: out });
  } catch (e) {
    postMessage({ id, ok: false, error: String((e && e.message) || e) });
  }
};
