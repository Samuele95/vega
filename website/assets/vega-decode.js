/* vega-decode — headline decode (spec §2.5). A [data-decode] heading inside a [data-stop] section is
   held hidden until it is reached, then decodes in, letter by letter, word by word. It decodes when:
     · the camera arrives at its stop (window 'vegaarrive' from vega-camera), or
     · it has been wholly on screen for 0.7 s (a visitor resting between stops never sees an empty panel), or
     · focus moves into its stop (keyboard users never land on hidden text).
   Once fully off-screen it is held again, so it decodes again on return. A heading already on screen
   when first seen is never blanked (it is held only from its next visit).
   Visual only: the real text never changes and stays in the DOM (transparent while held or decoding),
   so screen readers and copy-diff always read it. The scramble is drawn by an aria-hidden overlay
   placed in <body> over the heading's box (not inside the heading, so its textContent stays clean),
   each word held at its final width so the line never reflows.
   Fails open: nothing is held until a camera has configured (the first 'vegastop'); under
   prefers-reduced-motion nothing is held or animated at all.
   Load as a classic script at the top of <body>. API: window.VegaDecode { run(el), reset(el) }. */
(() => {
  if (window.VegaDecode) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const SEEN_MS = 700;
  const LOW = 'abcdefghijklmnopqrstuvwxyz', UP = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', DIG = '0123456789';
  const glyph = (c) => /[a-z]/.test(c) ? LOW[Math.random() * 26 | 0] : /[A-Z]/.test(c) ? UP[Math.random() * 26 | 0] : /[0-9]/.test(c) ? DIG[Math.random() * 10 | 0] : c;
  const css = document.createElement('style');
  css.textContent = '[data-decode].vd-hold{color:transparent!important}'
    + '.vd-ov{position:fixed;left:0;top:0;z-index:2;pointer-events:none;color:var(--fg,#E9F0F8);white-space:normal;margin:0}'
    + '.vd-w{display:inline-block;white-space:pre;text-align:left}';
  document.head.appendChild(css);
  const state = new WeakMap(), seen = new WeakMap();
  let armed = false;

  function reset(el) {
    if (RM.matches || !el || !armed) return;
    const s = state.get(el); if (s) { cancelAnimationFrame(s.raf); s.ov.remove(); }
    state.delete(el); el.classList.add('vd-hold');
  }
  function run(el) {
    if (!el || state.has(el)) return;
    if (RM.matches || !el.classList.contains('vd-hold')) { el.classList.remove('vd-hold'); return; }
    const text = el.textContent, cs = getComputedStyle(el), box = () => el.getBoundingClientRect();
    const ov = document.createElement('div'); ov.className = 'vd-ov'; ov.setAttribute('aria-hidden', 'true'); ov.setAttribute('data-copy-ignore', '1');
    for (const k of ['font', 'letterSpacing', 'textAlign', 'lineHeight', 'textTransform', 'wordSpacing']) ov.style[k] = cs[k];
    el.classList.remove('vd-hold'); ov.style.color = getComputedStyle(el).color; el.classList.add('vd-hold'); // the heading's own colour, read past the hold
    const place = () => { const r = box(); ov.style.width = r.width + 'px'; ov.style.transform = 'translate(' + r.left + 'px,' + r.top + 'px)'; };
    const words = text.split(/(\s+)/), plain = words.filter(w => !/^\s+$/.test(w));
    const spans = words.map(w => { if (/^\s+$/.test(w)) { ov.append(w); return null; } const s = document.createElement('span'); s.className = 'vd-w'; s.textContent = w; ov.append(s); return s; }).filter(Boolean);
    place(); document.body.appendChild(ov);
    spans.forEach(s => { s.style.width = s.getBoundingClientRect().width + 'px'; s.textContent = ''; });
    // timing after hubtown-study: ~25–35 ms per letter, words overlapping
    const per = 28, scr = 220, t0 = performance.now(); let k = 0;
    const plan = spans.map((s, i) => { const w = plain[i], start = k * per * 0.6; k += w.length; return { s, w, start }; });
    const end = Math.max(...plan.map(p => p.start + p.w.length * per + scr));
    const st = { ov, raf: 0 }; state.set(el, st);
    const tick = (now) => {
      const t = now - t0; place();
      for (const p of plan) {
        if (t < p.start) { p.s.textContent = ''; continue; }
        let out = '';
        for (let i = 0; i < p.w.length; i++) { const on = p.start + i * per, fix = on + scr; if (t >= fix) out += p.w[i]; else if (t >= on) out += glyph(p.w[i]); else break; }
        p.s.textContent = out;
      }
      if (t >= end) { ov.remove(); el.classList.remove('vd-hold'); state.delete(el); return; }
      st.raf = requestAnimationFrame(tick);
    };
    st.raf = requestAnimationFrame(tick);
  }

  const inStop = (i) => [...document.querySelectorAll('[data-stop="' + i + '"] [data-decode]')];
  // on screen: decode after a short look; fully off screen: hold again for the next visit
  // the look starts only once the whole heading is in view (not at its first pixel), so a slow scroll
  // doesn't play the decode at the bottom edge; it is held again only once wholly out of view
  const io = new IntersectionObserver((es) => es.forEach(en => {
    const el = en.target; clearTimeout(seen.get(el));
    if (en.intersectionRatio >= 0.99) { if (el.classList.contains('vd-hold')) seen.set(el, setTimeout(() => run(el), SEEN_MS)); }
    else if (!en.isIntersecting && !el.classList.contains('vd-hold') && !state.has(el)) reset(el);
  }), { threshold: [0, 0.99] });
  const hold = () => {
    if (!armed || RM.matches) return;
    document.querySelectorAll('[data-decode]').forEach(el => {
      if (el.dataset.vdSeen) return;
      el.dataset.vdSeen = '1';
      // a heading already on screen when first seen (a page's h1, a reload restored mid-scroll) is never
      // blanked; it is only watched, and held from its next visit on
      const r = el.getBoundingClientRect(), onScreen = r.bottom > 0 && r.top < innerHeight && r.width > 0;
      if (!onScreen) el.classList.add('vd-hold');
      io.observe(el);
    });
  };
  // arm only once a camera has configured (fail open if the scene never comes up)
  addEventListener('vegastop', () => { if (!armed) { armed = true; hold(); } });
  addEventListener('vegaarrive', (e) => inStop(e.detail.index).forEach(run));
  document.addEventListener('focusin', (e) => { const s = e.target.closest && e.target.closest('[data-stop]'); if (s) s.querySelectorAll('[data-decode]').forEach(run); });
  new MutationObserver(hold).observe(document.documentElement, { childList: true, subtree: true });
  window.VegaDecode = { run, reset };
})();
