/* vega-scramble — character-scramble decode for Martian Mono labels (loader lines, HUD labels).
   VegaScramble.run(el, text, opts) writes scrambled glyphs into el and settles left to right on `text`.
   Visual only: callers keep the real text in an aria-label or a visually hidden twin.
   Under prefers-reduced-motion the final text is written at once. */
(() => {
  if (window.VegaScramble) return;
  const POOL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/<>_-+=*#';
  const rm = matchMedia('(prefers-reduced-motion: reduce)');
  const live = new WeakMap();
  const pick = () => POOL[Math.floor(Math.random() * POOL.length)];

  function run(el, text, opts) {
    if (!el) return Promise.resolve();
    const o = Object.assign({ duration: 480, stagger: 28, frame: 40 }, opts);
    const prev = live.get(el); if (prev) prev.cancel();
    if (rm.matches || !text) { el.textContent = text; return Promise.resolve(); }
    const chars = [...text], n = chars.length;
    // each glyph settles at its own time: a left-to-right sweep capped by `duration`
    const step = Math.min(o.stagger, o.duration / Math.max(1, n));
    const settle = chars.map((_, i) => o.duration * 0.35 + i * step);
    let raf = 0, last = 0, t0 = 0, done;
    const p = new Promise(r => { done = r; });
    const tick = (now) => {
      if (!t0) t0 = now;
      const t = now - t0;
      if (now - last >= o.frame || t >= settle[n - 1]) {
        last = now;
        el.textContent = chars.map((c, i) => (c === ' ' || t >= settle[i]) ? c : pick()).join('');
      }
      if (t >= settle[n - 1]) { el.textContent = text; live.delete(el); done(); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    live.set(el, { cancel: () => { cancelAnimationFrame(raf); el.textContent = text; live.delete(el); done(); } });
    return p;
  }

  window.VegaScramble = { run, reduced: () => rm.matches };
})();
