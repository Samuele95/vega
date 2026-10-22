/* vega-reel — film-like chapters (G2, after Hubtown). One scroll gesture (wheel, trackpad, swipe, PageDown /
   PageUp / arrows / Space / Home / End) plays the whole flight to the next camera stop as one shot: the
   document scrolls there on a fixed curve, the camera flies with it, the chapter's text dissolves on the way
   out and comes back in place on arrival (the headline decode runs then). Native scroll stays underneath —
   the chapters are scroll positions, so the scrollbar, find-in-page, Tab focus and links all still work.
   A block that runs past the screen is read first: the gesture glides (eased, after shader.se) to its end,
   and the next gesture plays the shot; going up glides back to the chapter's rest first; the footer glides. Off under prefers-reduced-motion (plain scrolling; the camera cuts), inside dialogs, form fields,
   scrollable boxes, with ctrl (zoom) and behind the loader gate.
   Load as a classic script at the top of <body>. API: VegaReel.on(camera) — the page opts in once its camera
   is configured; VegaReel.go(i) — play to stop i (false when off, so callers fall back to native scroll);
   The last chapter shows the next page's name (a button) and a further gesture at the bottom warps there.
   VegaReel.playing, VegaReel.t (the shot's progress 0..1, for a crane over the whole shot); window 'vegareel' { detail: { from, to, state: 'start' | 'end' } }. */
(() => {
  if (window.VegaReel) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; // in-out cubic
  const fade = (t, a, b) => { const u = clamp((t - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };
  const LEG = 1500, PER = 260, MAXDUR = 2600; // ms for one chapter, each further chapter, the cap
  const GAP = 180;  // a wheel pause (ms) that ends a gesture: trackpad inertia after a play doesn't start another
  let cam = null, play = null, sm = null, lastWheel = 0, acc = 0, fired = false, queued = 0, touch = null;

  const enabled = () => cam && cam.o && !RM.matches && !document.documentElement.classList.contains('vl-holding');
  const rests = () => {
    const st = cam.o.stops, max = Math.max(0, document.documentElement.scrollHeight - innerHeight);
    return st.map(s => clamp(Math.round(s.el.getBoundingClientRect().top + scrollY - cam.off(s)), 0, max));
  };
  // the text blocks: their document box (a sticky block counts as its parent's box: it stays while that's on screen)
  const blocks = () => [...document.querySelectorAll('.vp, [data-reel-fade]')].map(el => {
    const sticky = getComputedStyle(el).position === 'sticky', r = (sticky ? el.parentElement : el).getBoundingClientRect();
    const sec = el.closest('[data-stop]');
    return { el, top: r.top + scrollY, bot: r.bottom + scrollY, k: sec ? +sec.dataset.stop : -1 };
  });
  const seen = (b, y) => b.bot > y + 60 && b.top < y + innerHeight - 60; // on screen at scroll y, clear of the HUD rows
  // where a gesture goes: { k } plays a shot to stop k; { lo, hi } scrolls natively (smoothed) inside that range —
  // the rest of a chapter's block that runs past the screen, the way back up to the chapter's rest, or the footer
  const target = (dir) => {
    const r = rests(), y = scrollY, H = innerHeight, max = document.documentElement.scrollHeight - H;
    const cur = r.reduce((k, v, i) => v <= y + 2 ? i : k, -1);
    if (dir > 0) {
      const k = r.findIndex(v => v > y + 2), next = k < 0 ? max : r[k];
      // a block of this chapter (or one before) still running below the screen is read first
      const end = blocks().filter(b => b.k >= 0 && b.k <= cur && seen(b, y) && getComputedStyle(b.el).position !== 'sticky')
        .reduce((m, b) => Math.max(m, b.bot + 40 - H), -Infinity);
      if (end > y + 2) return { lo: y, hi: Math.min(end, next) };
      if (k < 0) return { lo: y, hi: max };
      if (r[k] - y > H * 1.02) return { lo: y, hi: r[k] };
      return { k };
    }
    if (cur >= 0 && y > r[cur] + 2) return { lo: r[cur], hi: y }; // back to this chapter's rest first
    let k = -1; r.forEach((v, i) => { if (v < y - 2) k = i; });
    if (k < 0) return { lo: 0, hi: y };
    if (y - r[k] > H * 1.02) return { lo: r[k], hi: y };
    return { k };
  };

  // a block's lines: its text-bearing elements, outermost only (a heading, a paragraph, a list item, a row, a figure)
  const ATOM = 'h1,h2,h3,h4,p,li,dt,dd,pre,figure,table,.vp-num,[role="group"],[role="tablist"],[data-reel-atom]';
  const atoms = (el) => [...el.querySelectorAll(ATOM)].filter(a => { for (let n = a.parentElement; n && n !== el; n = n.parentElement) if (n.matches(ATOM)) return false; return a.getClientRects().length > 0; }).slice(0, 24);
  const start = (to) => {
    const r = rests(), y0 = scrollY, y1 = r[to];
    if (Math.abs(y1 - y0) < 2) return false;
    sm = null; // a shot replaces any glide
    const from = cam.active, legs = Math.max(1, Math.abs(to - from));
    const dur = Math.min(MAXDUR, LEG + PER * (legs - 1));
    // a block on screen at both ends stays; one only at the start dissolves out, one only at the end comes
    // in; everything the flight passes stays hidden on the way
    const bs = blocks().map(b => { const a = seen(b, y0), z = seen(b, y1);
      return { el: b.el, role: a && z ? 'keep' : a ? 'out' : z ? 'in' : 'pass' }; })
      .filter(b => b.role !== 'keep');
    // paused Web Animations, driven frame by frame: they sit above the page's own inline opacity (chain dims its
    // other steps through React, with a transition) and leave it untouched when cancelled
    const dir = Math.sign(y1 - y0), D = 28;
    bs.forEach(b => {
      const o = +getComputedStyle(b.el).opacity;
      // G7 depth: going forward, the leaving text passes the lens (grows, blurs); going back it sinks into the scene
      const so = dir > 0 ? 1.12 : 0.9, si = dir > 0 ? 0.88 : 1.1;
      const kf = b.role === 'out' ? [{ opacity: o, translate: '0 0', scale: '1', filter: 'blur(0px)', easing: 'ease-in' }, { opacity: 0, translate: '0 ' + (-dir * D) + 'px', scale: String(so), filter: 'blur(5px)', offset: 0.32 }, { opacity: 0, translate: '0 ' + (-dir * D) + 'px', scale: String(so), filter: 'blur(5px)' }]
        : b.role === 'in' ? [{ opacity: 0 }, { opacity: 0, offset: 0.6 }, { opacity: 1, offset: 0.64 }, { opacity: 1 }] // the block opens; its lines come in on their own (below)
        : [{ opacity: 0 }, { opacity: 0 }];
      b.anim = b.el.animate(kf, { duration: 1000, fill: 'both' }); b.anim.pause();
      // shader.se's reveal: the arriving block's lines rise in one after another out of a soft blur, on real time
      if (b.role === 'in') b.lines = atoms(b.el).map((el, i) => el.animate([{ opacity: 0, translate: '0 ' + (dir * 26) + 'px', scale: String(si), filter: i < 6 ? 'blur(7px)' : 'none' }, { opacity: 1, translate: '0 0', scale: '1', filter: 'blur(0px)' }],
        { duration: 760, delay: dur * 0.6 + Math.min(i, 10) * 75, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'backwards' }));
    });
    play = { from, to, y0, y1, t0: performance.now(), dur, bs };
    window.VegaReel.playing = true;
    dispatchEvent(new CustomEvent('vegareel', { detail: { from, to, state: 'start' } }));
    requestAnimationFrame(step);
    return true;
  };
  const finish = () => {
    if (!play) return;
    play.bs.forEach(b => b.anim.cancel()); // (the lines' own animations run out on their own)
    const { from, to } = play; play = null; window.VegaReel.playing = false;
    dispatchEvent(new CustomEvent('vegareel', { detail: { from, to, state: 'end' } }));
    if (queued) { const d = queued; queued = 0; const g = target(d); if (g.k != null) start(g.k); }
  };
  const step = (now) => {
    if (!play) return;
    const t = clamp((now - play.t0) / play.dur, 0, 1);
    window.VegaReel.t = t;
    // re-read the rest each frame: a layout change mid-play (fonts, resize) lands on the real stop
    const y1 = rests()[play.to]; play.y1 = y1;
    scrollTo({ top: play.y0 + (y1 - play.y0) * ease(t), behavior: 'instant' });
    // the leaving text drifts out against the travel and fades in the first third; the arriving text comes in
    // over the last third, drifting the short way into place — a cut between shots, not a scroll. An arriving block
    // ends at the opacity the page gives it by then (chain's other steps stay dimmed)
    if (t >= 0.6 && !play.fin) { play.fin = true; play.bs.forEach(b => { if (b.role === 'in') { const v = parseFloat(b.el.style.opacity); if (v >= 0 && v < 1) { const k = b.anim.effect.getKeyframes(); k[2].opacity = v; b.anim.effect.setKeyframes(k); } } }); }
    play.bs.forEach(b => { b.anim.currentTime = t * 1000; });
    // the arriving headline decodes as its block settles in (not after the camera settles: the number and arrow
    // would show alone for a beat; not earlier: the overlay lives in <body> and doesn't fade with the block)
    if (t >= 0.88 && !play.dec) { play.dec = true; if (window.VegaDecode) play.bs.forEach(b => { if (b.role === 'in') b.el.querySelectorAll('[data-decode]').forEach(h => window.VegaDecode.run(h)); }); }
    if (t < 1) requestAnimationFrame(step); else finish();
  };

  // ---- input
  const native = (e) => {
    const t = e.target;
    if (!t || !t.closest) return false;
    if (t.closest('[role="dialog"], dialog, input, textarea, select, [contenteditable=""], [contenteditable="true"], [data-reel-native]')) return true;
    // a box that can scroll itself takes its own wheel
    for (let n = t; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      if (n.scrollHeight > n.clientHeight + 1) { const ov = getComputedStyle(n).overflowY; if (ov === 'auto' || ov === 'scroll') return true; }
    }
    return false;
  };
  // the glide (after shader.se): native-range scrolling eases toward its goal instead of jumping per notch;
  // any other scroll (scrollbar, find, focus) cancels it
  const glide = (now) => {
    if (!sm) return;
    if (Math.abs(scrollY - sm.w) > 2) { sm = null; return; }
    const dt = Math.min(0.05, (now - (sm.t || now)) / 1000 || 0.016); sm.t = now;
    sm.y += (sm.goal - sm.y) * (1 - Math.exp(-dt * 9));
    if (Math.abs(sm.goal - sm.y) < 0.5) sm.y = sm.goal;
    scrollTo({ top: sm.y, behavior: 'instant' }); sm.w = scrollY;
    if (sm.y !== sm.goal) requestAnimationFrame(glide); else sm = null;
  };
  const glideBy = (d, g) => {
    const base = sm ? sm.goal : scrollY, goal = clamp(base + d, g.lo, g.hi);
    if (sm) { sm.goal = goal; return; }
    if (Math.abs(goal - scrollY) < 1) return;
    sm = { y: scrollY, w: scrollY, goal }; requestAnimationFrame(glide);
  };
  addEventListener('wheel', (e) => {
    if (!enabled() || e.ctrlKey || native(e) || Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;
    const now = performance.now(), fresh = now - lastWheel > GAP; lastWheel = now;
    const dir = Math.sign(e.deltaY); if (!dir) return;
    // one gesture plays one shot: it ends at a pause in the wheel events, so trackpad inertia still
    // arriving after a shot (or after reading to the end of a long block) doesn't start another (Hubtown's behaviour)
    if (fresh || Math.sign(acc) !== dir) { acc = 0; fired = false; }
    e.preventDefault();
    if (play) return;
    if (dir > 0 && fresh && endGesture(null)) return;
    const d = e.deltaY * (e.deltaMode === 1 ? 32 : e.deltaMode === 2 ? innerHeight : 1);
    const g = target(dir);
    if (g.k == null) { fired = true; glideBy(d, g); return; } // the rest of a long block, the way back, the footer
    acc += d;
    if (!fired && Math.abs(acc) >= 24) { fired = true; start(g.k); }
  }, { passive: false });

  addEventListener('touchstart', (e) => { touch = enabled() && e.touches.length === 1 && !native(e) ? { y: e.touches[0].clientY, done: false, shot: null } : null; }, { passive: true });
  addEventListener('touchmove', (e) => {
    if (!touch || !enabled()) return;
    const dy = touch.y - e.touches[0].clientY, dir = Math.sign(dy);
    if (play || touch.done) { e.preventDefault(); return; }
    if (!dir) return;
    if (dir > 0 && Math.abs(dy) > 36 && atEnd() && endGesture(e)) { touch.done = true; return; }
    // the first move decides: a shot, or the finger scrolls natively (a long block, the footer)
    if (touch.shot == null) { const g = target(dir); touch.shot = g.k != null ? g.k : false; }
    if (touch.shot === false) return;
    e.preventDefault();
    if (Math.abs(dy) > 36) { touch.done = true; start(touch.shot); }
  }, { passive: false });
  addEventListener('touchend', () => { touch = null; }, { passive: true });

  addEventListener('keydown', (e) => {
    if (!enabled() || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || native(e)) return;
    const a = document.activeElement;
    if (a && a !== document.body && a.closest && a.closest('button, [role="button"], [role="group"], [role="grid"], [role="slider"], [role="listbox"], [role="tablist"], vega-rail') && (e.key === ' ' || e.key.startsWith('Arrow'))) return; // the control's own keys
    let dir = 0, to = null, step = innerHeight * 0.85;
    if (e.key === 'PageDown' || (e.key === ' ' && !e.shiftKey)) dir = 1;
    else if (e.key === 'ArrowDown') { dir = 1; step = 80; }
    else if (e.key === 'PageUp' || (e.key === ' ' && e.shiftKey)) dir = -1;
    else if (e.key === 'ArrowUp') { dir = -1; step = 80; }
    else if (e.key === 'Home') { dir = -1; to = 0; }
    else if (e.key === 'End') { dir = 1; to = cam.count - 1; }
    if (!dir) return;
    e.preventDefault();
    if (play) { queued = to == null ? dir : 0; return; }
    if (to != null) { if (!start(to)) glideBy(dir * 1e6, { lo: 0, hi: document.documentElement.scrollHeight - innerHeight }); return; }
    if (dir > 0 && e.key !== 'ArrowDown' && endGesture(null)) return;
    const g = target(dir);
    if (g.k != null) start(g.k); else glideBy(dir * step, g);
  });

  // ---- the end of a page (G7: no pager): the last chapter shows the next page's name; one more gesture at the bottom
  // warps there (vega-fly). The order is the HUD's page links; the last page leads back to the threshold.
  const next = () => {
    const L = [...document.querySelectorAll('vega-hud .vh-links a')]; if (!L.length) return null;
    const here = location.pathname.split('/').pop() || 'index.dc.html', i = L.findIndex(a => a.getAttribute('href') === here);
    const n = L[(i + 1) % L.length]; return n ? { href: n.getAttribute('href'), label: n.getAttribute('aria-label') || n.textContent.trim() } : null;
  };
  const atEnd = () => scrollY >= document.documentElement.scrollHeight - innerHeight - 2;
  let cue = null, leaving = false, endArmed = 0;
  const go2 = () => { const n = next(); if (!n || leaving) return; leaving = true; if (window.VegaFly) window.VegaFly.go(n.href, { x: innerWidth / 2, y: innerHeight * 0.62 }); else location.href = n.href; };
  const cueCss = document.createElement('style');
  cueCss.textContent = '.vr-next{position:fixed;left:50%;bottom:calc(24px + 12px);transform:translateX(-50%);z-index:9;display:flex;flex-direction:column;align-items:center;gap:8px;padding:8px 6px;border:0;background:none;cursor:pointer;color:var(--dim,#8C9EB3);font:500 11px/1 "Martian Mono",ui-monospace,Menlo,monospace;letter-spacing:.22em;text-transform:uppercase;opacity:0;pointer-events:none;transition:opacity .5s ease;text-shadow:0 0 10px var(--bg,#04080E)}'
    + '.vr-next.on{opacity:1;pointer-events:auto}.vr-next b{font-weight:500;color:var(--fg,#E9F0F8)}.vr-next i{display:block;width:7px;height:7px;border-right:1px solid var(--acc,#9CC8FF);border-bottom:1px solid var(--acc,#9CC8FF);transform:rotate(45deg);animation:vr-bob 1.8s ease-in-out infinite}'
    + '.vr-next:hover b,.vr-next:focus-visible b{color:var(--acc,#9CC8FF)}@keyframes vr-bob{50%{transform:translateY(4px) rotate(45deg)}}'
    + '@media (max-width:600px){.vr-next{bottom:calc(12px + 56px)}}@media (prefers-reduced-motion: reduce){.vr-next{transition:none}.vr-next i{animation:none}}'
    + 'html.vl-holding .vr-next,html.vfl-active .vr-next{visibility:hidden}';
  document.head.appendChild(cueCss);
  const showCue = () => {
    if (!cam || !cam.o) return; const n = next(); if (!n) return;
    if (!cue) { cue = document.createElement('button'); cue.type = 'button'; cue.className = 'vr-next'; cue.addEventListener('click', go2); document.body.appendChild(cue); }
    const label = 'NEXT · ' + n.label; if (cue.dataset.l !== label) { cue.dataset.l = label; cue.innerHTML = ''; const t = document.createElement('span'); t.textContent = 'NEXT · '; const b = document.createElement('b'); b.textContent = n.label; t.append(b); const ar = document.createElement('i'); ar.setAttribute('aria-hidden', 'true'); cue.append(t, ar); cue.setAttribute('aria-label', 'Next page: ' + n.label); }
    /* only once the last chapter's text has run past (a long last chapter on a phone would sit under the cue) */
    const on = cam.active === cam.count - 1 && !play && document.documentElement.scrollHeight - innerHeight - scrollY < innerHeight * 0.12;
    cue.classList.toggle('on', on); cue.tabIndex = on ? 0 : -1;
  };
  addEventListener('vegastop', () => requestAnimationFrame(showCue)); addEventListener('vegareel', showCue);
  let cueRaf = 0; addEventListener('scroll', () => { if (!cueRaf) cueRaf = requestAnimationFrame(() => { cueRaf = 0; showCue(); }); }, { passive: true });
  // a fresh gesture at the very bottom of the last chapter leaves for the next page (a held or inertial scroll that
  // just arrived doesn't: the first gesture at the end arms it, the next one goes)
  const endGesture = (e) => { if (!cam || !cam.o || cam.active !== cam.count - 1 || !atEnd()) { endArmed = 0; return false; }
    if (e) e.preventDefault(); const now = performance.now(); if (endArmed && now - endArmed > 350) { go2(); return true; } endArmed = endArmed || now; return true; };

  // G7 text in the scene: the content layer parallaxes against the pointer (the galaxy moves the other way), and
  // headlines take VEGA's light — a rim on the side facing the star and a faint bloom, stronger the closer it is.
  // Both are CSS variables (--vp-px/--vp-py on <html>, --li/--lx/--ly per headline) read by vega-panel.css.
  const LIT = '.vp :is(h1,h2), [data-reel-fade] > p:first-child';
  let px = 0, py = 0, tx = 0, ty = 0, lastLit = 0;
  addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') { tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5; } }, { passive: true });
  const light = (now) => {
    requestAnimationFrame(light);
    if (document.hidden) return;
    const root = document.documentElement;
    if (!RM.matches) { const k = 0.06; const nx = px + (tx - px) * k, ny = py + (ty - py) * k;
      if (Math.abs(nx - px) + Math.abs(ny - py) > 0.0004) { px = nx; py = ny; root.style.setProperty('--vp-px', (-px * 14).toFixed(2) + 'px'); root.style.setProperty('--vp-py', (-py * 9).toFixed(2) + 'px'); } }
    if (now - lastLit < 50) return; lastLit = now; // 20 Hz is plenty for light that moves with the camera
    const g = document.querySelector('vega-galaxy'), st = g && !g.failed && g.star;
    const H = innerHeight, diag = Math.hypot(innerWidth, H);
    document.querySelectorAll(LIT).forEach(h => {
      const b = h.getBoundingClientRect(); if (!b.width || b.bottom < 0 || b.top > H) return;
      let li = 0, lx = 0, ly = 0;
      if (st) { const gx = g.getBoundingClientRect(), sx = gx.left + st.x, sy = gx.top + st.y, dx = sx - (b.left + b.width / 2), dy = sy - (b.top + b.height / 2), d = Math.hypot(dx, dy) || 1;
        li = Math.pow(Math.max(0, 1 - d / (diag * 0.85)), 1.4) * (0.55 + 0.45 * Math.min(1, st.r / 90)); lx = dx / d; ly = dy / d; }
      const k = li.toFixed(2) + ' ' + lx.toFixed(2) + ' ' + ly.toFixed(2); if (h.dataset.lit === k) return; h.dataset.lit = k;
      h.style.setProperty('--li', li.toFixed(3)); h.style.setProperty('--lx', lx.toFixed(3)); h.style.setProperty('--ly', ly.toFixed(3));
    });
  };
  requestAnimationFrame(light);

  window.VegaReel = {
    playing: false,
    on(c) { cam = c; setTimeout(showCue, 300); },
    next,
    off() { cam = null; },
    go(i) { if (!enabled()) return false; if (play) finish(); return start(clamp(i, 0, cam.count - 1)) || true; },
  };
})();
