/* Main-thread side of the analysis engine. Starts the worker on first page load,
   exposes promise-returning calls, and reports state to the header pill.
   Never fails silently: every error reaches the UI with its message. */
'use strict';
window.Engine = (() => {
  let worker = null;
  let state = 'idle';
  let detail = '';
  let nextId = 1;
  const pending = new Map();
  const listeners = new Set();
  let readyResolve, readyReject;
  const ready = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
  ready.catch(() => {}); // callers handle it; avoid an unhandled-rejection log

  function set(s, d) {
    state = s; detail = d || '';
    listeners.forEach((f) => f(state, detail));
    if (s === 'ready') readyResolve();
    if (s === 'error') {
      readyReject(new Error(detail));
      pending.forEach((p) => p.reject(new Error(detail)));
      pending.clear();
    }
  }

  function start() {
    if (worker) return ready;
    set('loading', 'starting');
    try {
      worker = new Worker('engine-worker.js');
    } catch (e) {
      set('error', String(e.message || e));
      return ready;
    }
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'status') { set(m.state, m.detail); return; }
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
    };
    // importScripts failing (offline, CDN blocked) surfaces here, not as a status message.
    worker.onerror = (ev) => {
      ev.preventDefault();
      if (state !== 'error') set('error', ev.message || 'the worker crashed');
    };
    return ready;
  }

  function call(fn, ...args) {
    start();
    if (state === 'error') return Promise.reject(new Error(detail));
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, fn, args });
    });
  }

  return {
    start,
    ready: () => ready,
    state: () => state,
    detail: () => detail,
    onChange: (f) => { listeners.add(f); f(state, detail); },
    analyze: async (payload) => JSON.parse(await call('analyze_json', JSON.stringify(payload))),
    config: async () => JSON.parse(await call('config_json')),
    trend: async (records, today) => JSON.parse(await call('summarize_json', JSON.stringify(records), today || '')),
    errorText: () => `The analysis engine didn't load: ${detail}. Check your connection and reload.`,
  };
})();
