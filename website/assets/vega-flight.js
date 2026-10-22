/* vega-flight — the spacecraft (galaxy plan G3, the user's idea). FLY in the HUD (or F) takes the visitor out of the page
   into the 3D galaxy (vega-galaxy, WebGL) at the VEGA system: VEGA at the core, the six pages as its planets in INDEX
   order. Free flight: W / S or the wheel — throttle; Shift — boost; ↑ / ↓ — pitch; ← / → or A / D — turn; Q / E — roll;
   drag (mouse or finger) — steer. A planet's label or its entry in the list engages the autopilot, which flies there and
   stops; near a planet LAND (Enter) jumps to its page through the warp (vega-fly). Esc, F or FLY again returns to the
   page where it was. The cockpit readouts are new HUD labels with units only; every planet's name is the HUD nav's text.
   Reduced motion: no flight — a still map of the system from above whose planets link straight to their pages.
   No WebGL: the planet list alone. Loaded on first use by vega-hud. API: window.VegaFlight.toggle() / open() / close();
   window 'vegaflight' { detail: { on } }. */
(() => {
  if (window.VegaFlight) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const MONO = "'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace";
  const css = document.createElement('style');
  css.textContent = `
.vfl{position:fixed;inset:0;z-index:8;color:var(--dim,#8C9EB3);font:500 10px/1.3 ${MONO};letter-spacing:.18em;touch-action:none;outline:none;overflow:hidden}
.vfl.solid{background:var(--bg,#04080E);opacity:0;transition:opacity .45s ease}.vfl.solid.on{opacity:1} /* no page galaxy to fly out of: its own sky */
/* G7: one continuous scene — the page's own galaxy stays; the cockpit fades in once the camera has left the page */
.vfl-c{opacity:0;transition:opacity .6s ease}.vfl.live .vfl-c{opacity:1}
.vfl vega-galaxy{position:absolute;inset:0}
.vfl-lab{position:absolute;left:0;top:0;display:flex;align-items:center;gap:8px;padding:4px 6px;margin:0;border:0;background:none;color:var(--fg,#E9F0F8);font:500 10.5px/1 ${MONO};letter-spacing:.22em;text-transform:uppercase;cursor:pointer;white-space:nowrap;text-shadow:0 0 8px var(--bg,#04080E);will-change:transform}
.vfl-lab::before{content:'';width:16px;height:1px;background:var(--acc,#9CC8FF);opacity:.6} /* a pinned readout, as on the pages (vega-readouts) */
.vfl-lab b{font-weight:500;color:var(--acc,#9CC8FF)}.vfl-lab b::after{content:' ·';color:var(--dim,#8C9EB3)}
.vfl-lab.tgt::before{opacity:1;width:26px}
.vfl-lab.left::before{order:1} /* on a planet's left: the text reads the same, the leader moves to its end, toward the planet */
.vfl-lab span{color:var(--mut,#A3B4C7)}
.vfl-lab:hover span,.vfl-lab:focus-visible span,.vfl-lab.tgt span{color:var(--fg,#E9F0F8)}
.vfl-lab.vega{cursor:default;pointer-events:none}.vfl-lab.vega::before{display:none}
.vfl-lab.vega span{color:var(--fg,#E9F0F8)}
.vfl-ret{position:absolute;left:50%;top:50%;width:46px;height:46px;transform:translate(-50%,-50%);pointer-events:none;opacity:.7}
.vfl-panel{position:absolute;right:calc(24px + 18px);top:calc(24px + 46px);display:flex;flex-direction:column;gap:2px;max-height:calc(100% - 200px);overflow:auto}
.vfl-panel h2{margin:0 0 8px;font:inherit;color:var(--dim,#8C9EB3)}
.vfl-panel button{display:flex;gap:10px;align-items:baseline;text-align:left;padding:6px 4px;border:0;border-top:1px solid var(--line,rgba(156,200,255,.18));background:none;color:var(--mut,#A3B4C7);font:inherit;letter-spacing:inherit;cursor:pointer}
.vfl-panel button b{font-weight:500;color:var(--acc,#9CC8FF)}
.vfl-panel button:hover,.vfl-panel button:focus-visible,.vfl-panel button[aria-current="true"]{color:var(--fg,#E9F0F8)}
.vfl-read{position:absolute;left:calc(24px + 18px);top:calc(24px + 46px);display:grid;grid-template-columns:auto auto;gap:6px 16px;pointer-events:none}
.vfl-read dt{color:var(--dim,#8C9EB3)}.vfl-read dd{margin:0;color:var(--fg,#E9F0F8);font-variant-numeric:tabular-nums}
.vfl-land{position:absolute;left:0;top:0;display:none;align-items:center;gap:12px;padding:12px 2px 6px;border:0;background:none;color:var(--fg,#E9F0F8);font:inherit;letter-spacing:inherit;cursor:pointer;text-shadow:0 0 10px var(--bg,#04080E),0 0 3px var(--bg,#04080E)}
.vfl-land::before{content:'';position:absolute;left:2px;top:0;width:28px;height:1px;background:var(--acc,#9CC8FF)}
.vfl-land.on{display:flex}
@media (min-width:701px){.vfl-land .vfl-land-n{display:none}} /* beside the planet's own label: the name is already there */
.vfl-land b{font-weight:500;color:var(--acc,#9CC8FF)}
.vfl-help{position:absolute;left:50%;bottom:calc(24px + 22px);transform:translateX(-50%);color:var(--dim,#8C9EB3);white-space:nowrap;pointer-events:none}
.vfl-exit{position:absolute;right:calc(24px + 18px);bottom:calc(24px + 10px);padding:8px 4px;border:0;background:none;color:var(--mut,#A3B4C7);font:inherit;letter-spacing:inherit;cursor:pointer}
.vfl-exit:hover,.vfl-exit:focus-visible{color:var(--fg,#E9F0F8)}
.vfl-thr{position:absolute;right:calc(24px + 18px);bottom:calc(24px + 52px);display:none;flex-direction:column;gap:8px}
.vfl-thr button{width:52px;height:44px;border:0;background:none;color:var(--fg,#E9F0F8);font:400 22px/1 inherit;text-shadow:0 0 8px var(--bg,#04080E)}
.vfl-edge{position:absolute;left:0;top:0;width:14px;height:14px;pointer-events:none;display:none}
@media (pointer:coarse){.vfl-thr{display:flex}.vfl-help{display:none}}
@media (max-width:700px){.vfl-panel{flex-direction:row;gap:6px;top:auto;right:auto;left:50%;transform:translateX(-50%);bottom:calc(24px + 58px);max-height:none;overflow:visible}.vfl-panel h2,.vfl-panel button span{display:none}.vfl-panel button{border:0;border-bottom:1px solid transparent;padding:12px 8px}.vfl-panel button[aria-current="true"]{border-bottom-color:var(--acc,#9CC8FF)}.vfl-read{grid-template-columns:auto auto auto auto}.vfl-land{top:auto!important;left:50%!important;bottom:calc(24px + 112px);transform:translateX(-50%)!important}.vfl-land::before{left:50%;margin-left:-14px}.vfl-thr{bottom:auto;top:42%}.vfl-exit{bottom:calc(24px + 10px)}}
.vfl-land{white-space:nowrap}
@media (max-width:600px){.vfl-exit{right:calc(12px + 8px);bottom:calc(12px + 10px)}.vfl-exl{display:none}} /* on a phone's bottom row, beside FLY: BACK alone (no Esc key there) */ /* the HUD's phone inset: on SOUND / FLY's row, where the hidden counter sat */
/* the page steps aside: its text and its scenes fade (then stop drawing, vfl-deep); the galaxy and the HUD stay */
html.vfl-active :is(main,header,footer,[data-foot],canvas[data-fx]),html.vfl-active [data-stage] > :is(:has(vega-constellation,vega-orbits,vega-readouts,vega-sky-lab,canvas[data-sky]),vega-constellation,vega-orbits,vega-readouts,vega-sky-lab,[style*="opacity:.38"]):not(:has(vega-galaxy)){opacity:0!important;visibility:hidden;pointer-events:none;transition:opacity .5s ease,visibility 0s linear .5s}
html.vfl-closing :is(main,header,footer,[data-foot],canvas[data-fx]),html.vfl-closing [data-stage] > *{transition:opacity .6s ease}
html.vfl-deep [data-stage] > :is(:has(vega-constellation,vega-orbits,vega-readouts,vega-sky-lab,canvas[data-sky]),vega-constellation,vega-orbits,vega-readouts,vega-sky-lab):not(:has(vega-galaxy)){display:none!important}
html.vfl-active vega-hud::before,html.vfl-active vega-hud::after{opacity:0!important} /* the page's text bands would cover the cockpit's own controls (BACK) */
html.vfl-active vega-rail,html.vfl-active vega-readouts,html.vfl-active [data-fighud],html.vfl-active [data-hudtext],html.vfl-active [data-figfoot],html.vfl-active [data-ticks],html.vfl-active [data-pagebar],html.vfl-active .vd-ov{visibility:hidden!important}
@media (prefers-reduced-motion: reduce){.vfl,.vfl-c,html.vfl-active *,html.vfl-closing *{transition:none!important}}`;
  document.head.appendChild(css);

  // ---- tiny vector kit
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]), nrm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const rot = (v, k, a) => { const c = Math.cos(a), s = Math.sin(a); return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c))); }; // Rodrigues

  let root = null, gal = null, own = false, saved = null, ship = null, ui = null, back = null, keys = {}, drag = null, onKey = null, onKeyUp = null, landing = false;
  const pages = () => [...document.querySelectorAll('vega-hud .vh-links a')].map(a => ({ href: a.getAttribute('href'), label: a.getAttribute('aria-label') || a.textContent.trim() }))
    .filter(p => !/(^|\/)index\.dc\.html$/.test(p.href)).slice(0, 6);
  const here = () => { const m = location.pathname.match(/([^/]+)\.dc\.html$/); return m ? m[1] : 'index'; };
  const pad = (n) => String(n).padStart(2, '0');
  const el = (t, c, a) => { const e = document.createElement(t); if (c) e.className = c; if (a) for (const k in a) e.setAttribute(k, a[k]); return e; };

  function open() {
    if (root) return;
    const P = pages(); if (!P.length) return;
    back = { y: scrollY, focus: document.activeElement };
    root = el('div', 'vfl', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Flight', tabindex: '-1' });
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0'; root.appendChild(host);
    document.body.appendChild(root);
    document.documentElement.classList.add('vfl-active');
    // the planet list: the keyboard's way to every planet (and the whole map without WebGL)
    const panel = el('nav', 'vfl-panel', { 'aria-label': 'Planets' }), h2 = el('h2'); h2.textContent = 'VEGA SYSTEM'; panel.appendChild(h2);
    const list = P.map((p, k) => { const b = el('button', '', { type: 'button', 'aria-label': pad(k + 1) + ' ' + p.label }); b.innerHTML = '<b>' + pad(k + 1) + '</b>'; const s = el('span'); s.textContent = p.label; b.appendChild(s);
      b.addEventListener('click', () => engage(k)); panel.appendChild(b); return b; });
    const exit = el('button', 'vfl-exit', { type: 'button', 'aria-label': 'ESC · BACK TO PAGE' }); exit.innerHTML = '<span class="vfl-exl">ESC · </span>BACK<span class="vfl-exl"> TO PAGE</span>'; exit.addEventListener('click', close);
    root.append(panel, exit);
    ui = { P, list, panel, exit, labs: [] };
    requestAnimationFrame(() => root && root.classList.add('on'));
    dispatchEvent(new CustomEvent('vegaflight', { detail: { on: true } }));

    onKey = (e) => {
      if (!root) return;
      if (e.key === 'Escape' || ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey)) { e.preventDefault(); e.stopPropagation(); close(); return; }
      if (e.key === 'Tab' || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key, onBtn = e.target && e.target.closest && e.target.closest('button');
      if (k === 'Enter') { if (!onBtn && ship && ship.near != null) { e.preventDefault(); land(ship.near); } e.stopPropagation(); return; }
      if (k === ' ' && onBtn) { e.stopPropagation(); return; }
      if (['w', 's', 'a', 'd', 'q', 'e', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Shift', ' '].includes(k)) { e.preventDefault(); keys[k] = true; if (ship) ship.auto = null; }
      e.stopPropagation(); // the page's own keys (i, reel, …) sleep while flying
    };
    onKeyUp = (e) => { const k = e.key.length === 1 ? e.key.toLowerCase() : e.key; keys[k] = false; };
    addEventListener('keydown', onKey, true); addEventListener('keyup', onKeyUp, true);
    root.addEventListener('wheel', (e) => { e.preventDefault(); if (ship && !RM.matches) { ship.auto = null; ship.want = clamp(ship.want * Math.exp(-e.deltaY * 0.004), 0, 120); if (ship.want < 0.3) ship.want = e.deltaY > 0 ? 0 : 0.3; } }, { passive: false });
    root.focus({ preventScroll: true });
    ui.panel.classList.add('vfl-c'); exit.classList.add('vfl-c');
    // the page's own galaxy becomes the flight's: the camera leaves the page's view instead of cutting to a new screen
    const pg = [...document.querySelectorAll('[data-stage] vega-galaxy')].find(g => g.isConnected && g.ps && !g.failed && g.frame);
    if (pg) { gal = pg; own = false; saved = { drive: gal.drive, system: gal.system, res: gal.res, mark: gal.mark, streak: gal.streak, exposure: gal.exposure, from: null };
      gal.drive = null; gal.mark = null; gal.streak = 0; gal.exposure = 1; gal.system = true; gal.res = 1;
      setTimeout(() => { if (root) document.documentElement.classList.add('vfl-deep'); }, RM.matches ? 0 : 560); // the faded scenes stop drawing
      cockpit(); }
    else { // no page galaxy to fly out of (or no WebGL): a sky of its own, or the planet list alone
      root.classList.add('solid'); document.documentElement.classList.add('vfl-deep');
      const ready = (ok) => { if (!root) return;
        gal = ok ? document.createElement('vega-galaxy') : null; own = !!gal; if (gal) { host.appendChild(gal); if (gal.failed || !gal.ps) { gal.remove(); gal = null; own = false; } }
        if (gal) cockpit(); else { root.classList.add('live'); list[0].focus(); }
      };
      if (window.VegaFly && window.VegaFly.loadGalaxy) window.VegaFly.loadGalaxy().then(ready); else ready(!!customElements.get('vega-galaxy'));
    }
  }

  function cockpit() {
    const rm = RM.matches, P = ui.P;
    gal.system = true; gal.res = 1; // full resolution: the planets come close
    // start: on index outside the system, looking in at VEGA; on a page, just off that page's planet
    const pl = gal.planets(), stem = here(), k = P.findIndex(p => p.href.replace(/\.dc\.html$/, '').split('/').pop() === stem);
    let pos = [0, 32, 122];
    if (rm) pos = [0, 170, 60];
    else if (k >= 0) { const p = pl[k].p, out = nrm([p[0], 0, p[2]]); pos = add(add(p, mul(out, 22)), [0, 6, 0]); }
    let fwd = nrm(sub([0, 0, 0], pos)), right = nrm(cross(fwd, [0, 1, 0])), up = cross(right, fwd);
    ship = { pos, fwd, up, speed: 0, want: rm ? 0 : 4, av: [0, 0, 0], auto: null, near: null, mt: 0 };
    // leaving the page: from the page's camera (and its screen anchor) out to the ship, eased; reduced motion cuts
    const f = gal.frame; if (saved && f) saved.from = { eye: f.eye.slice(), fwd: nrm(f.fwd), up: [0, 1, 0], ax: f.ax || 0, ay: f.ay || 0 };
    if (saved && saved.from && !rm) { ship.move = { t: 0, dur: 2.2, from: saved.from, to: { eye: pos, fwd, up, ax: 0, ay: 0 } }; gal.cam = Object.assign({}, saved.from); }
    else { gal.cam = { eye: pos, fwd, up }; root.classList.add('live'); }
    // labels: VEGA and the planets, pinned to their projections each frame
    const vega = el('div', 'vfl-lab vega vfl-c', { 'aria-hidden': 'true' }); vega.innerHTML = '<span>VEGA</span>'; root.appendChild(vega);
    ui.labs = P.map((p, i) => { const b = el('button', 'vfl-lab vfl-c', { type: 'button', tabindex: '-1', 'aria-hidden': 'true' }); b.innerHTML = '<b>' + pad(i + 1) + '</b>'; const s = el('span'); s.textContent = p.label; b.appendChild(s);
      b.addEventListener('click', () => engage(i)); root.appendChild(b); return b; });
    ui.vega = vega;
    if (!rm) {
      const ret = el('div', 'vfl-ret', { 'aria-hidden': 'true' });
      ret.innerHTML = '<svg viewBox="-23 -23 46 46" width="46" height="46"><g stroke="var(--acc,#9CC8FF)" stroke-width="1" fill="none"><path d="M-22 0H-9M9 0H22M0 -22V-9M0 9V22"/><circle r="2.2"/></g></svg>';
      const read = el('dl', 'vfl-read', { 'aria-hidden': 'true' });
      read.innerHTML = '<dt>SPD</dt><dd data-r="spd"></dd><dt>HDG</dt><dd data-r="hdg"></dd><dt>VEGA</dt><dd data-r="vega"></dd><dt>TGT</dt><dd data-r="tgt"></dd>';
      const help = el('div', 'vfl-help', { 'aria-hidden': 'true' }); help.textContent = 'DRAG · STEER   W S · THROTTLE   A D · TURN   ↑ ↓ · PITCH   SHIFT · BOOST';
      const landB = el('button', 'vfl-land', { type: 'button' }); landB.addEventListener('click', () => ship.near != null && land(ship.near));
      const thr = el('div', 'vfl-thr'), tUp = el('button', '', { type: 'button', 'aria-label': 'THRUST' }), tDn = el('button', '', { type: 'button', 'aria-label': 'BRAKE' });
      tUp.textContent = '+'; tDn.textContent = '−'; thr.append(tUp, tDn);
      const hold = (b, k) => { b.addEventListener('pointerdown', (e) => { e.stopPropagation(); keys[k] = true; ship.auto = null; }); ['pointerup', 'pointerleave', 'pointercancel'].forEach(t => b.addEventListener(t, () => { keys[k] = false; })); };
      hold(tUp, 'w'); hold(tDn, 's');
      const edge = el('div', 'vfl-edge', { 'aria-hidden': 'true' }); edge.innerHTML = '<svg viewBox="-7 -7 14 14" width="14" height="14"><path d="M-5 -5 L6 0 L-5 5" fill="none" stroke="var(--fg,#E9F0F8)" stroke-width="1.2"/></svg>';
      [ret, read, help, landB, thr, edge].forEach(n => n.classList.add('vfl-c')); root.append(ret, read, help, landB, thr, edge);
      Object.assign(ui, { read, landB, edge, rd: { spd: read.querySelector('[data-r="spd"]'), hdg: read.querySelector('[data-r="hdg"]'), vega: read.querySelector('[data-r="vega"]'), tgt: read.querySelector('[data-r="tgt"]') } });
      // drag to steer: the offset from where the drag began sets the turn rate
      root.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; drag = { x: e.clientX, y: e.clientY, dx: 0, dy: 0, id: e.pointerId }; ship.auto = null; try { root.setPointerCapture(e.pointerId); } catch (x) {} });
      root.addEventListener('pointermove', (e) => { if (drag && e.pointerId === drag.id) { drag.dx = e.clientX - drag.x; drag.dy = e.clientY - drag.y; } });
      ['pointerup', 'pointercancel'].forEach(t => root.addEventListener(t, () => { drag = null; }));
      gal.tick = fly;
    }
    gal.onframe = labels;
    ui.list[k >= 0 ? k : 0].setAttribute('aria-current', k >= 0 ? 'true' : 'false');
    (ui.list[k >= 0 ? k : 0]).focus({ preventScroll: true });
  }

  // one frame of flight
  // a camera move between the page's view and the ship: an arc around VEGA (direction slerped, radius eased in log space)
  // so it never cuts through the system, the view aimed at a blended target (each end's point at VEGA's distance)
  const blend = (a, b, e) => { const n = (x, y) => nrm([x[0] + (y[0] - x[0]) * e, x[1] + (y[1] - x[1]) * e, x[2] + (y[2] - x[2]) * e]);
    const r0 = len(a.eye) || 1, r1 = len(b.eye) || 1, d0 = mul(a.eye, 1 / r0), d1 = mul(b.eye, 1 / r1), om = Math.acos(clamp(dot(d0, d1), -1, 1));
    const dir = om < 1e-3 ? n(d0, d1) : add(mul(d0, Math.sin((1 - e) * om) / Math.sin(om)), mul(d1, Math.sin(e * om) / Math.sin(om)));
    const eye = mul(nrm(dir), Math.exp(Math.log(r0) + (Math.log(r1) - Math.log(r0)) * e));
    const T0 = add(a.eye, mul(a.fwd, r0)), T1 = add(b.eye, mul(b.fwd, r1)), T = add(T0, mul(sub(T1, T0), e));
    return { eye, fwd: nrm(sub(T, eye)), up: n(a.up, b.up), ax: a.ax + (b.ax - a.ax) * e, ay: a.ay + (b.ay - a.ay) * e }; };
  function fly(dt) {
    if (ship && ship.move) { const m = ship.move; m.t = Math.min(m.dur, m.t + dt); const u = m.t / m.dur, e = u * u * u * (u * (u * 6 - 15) + 10); // smootherstep
      gal.cam = blend(m.from, m.to, e);
      // index: VEGA's mark (drawn in light at the core) dissolves into the star on the way out and returns on the way back
      if (saved && saved.mark) gal.mark = Object.assign({}, saved.mark, { k: (saved.mark.k || 0) * (m.done ? clamp((e - 0.55) / 0.45, 0, 1) : clamp(1 - e * 2.5, 0, 1)) });
      if (u >= 1) { ship.move = null; if (m.done) m.done(); else if (root) root.classList.add('live'); }
      return; }
    if (!ship || landing) { if (gal && ship) gal.cam = { eye: ship.pos, fwd: ship.fwd, up: ship.up }; return; }
    const s = ship, P = ui.P, pl = gal.planets();
    let right = nrm(cross(s.fwd, s.up));
    // inputs → turn rates (rad/s): yaw about up, pitch about right, roll about forward
    const W = innerWidth, H = innerHeight, m = Math.min(W, H);
    let iy = (keys.a || keys.ArrowLeft ? 1 : 0) - (keys.d || keys.ArrowRight ? 1 : 0), ip = (keys.ArrowUp ? 1 : 0) - (keys.ArrowDown ? 1 : 0), ir = (keys.q ? 1 : 0) - (keys.e ? 1 : 0);
    if (drag) { iy -= clamp(drag.dx / (m * 0.3), -1, 1); ip -= clamp(drag.dy / (m * 0.3), -1, 1); if (Math.abs(drag.dx) + Math.abs(drag.dy) > 4) iy = iy || 1e-6; }
    if (s.auto != null) { // the autopilot: turn to the planet, run, brake to a stop off its day side
      const tp = pl[s.auto].p, sz = pl[s.auto].size, stand = add(tp, mul(nrm(add(add(mul(nrm(mul(tp, -1)), 0.8), mul(nrm(cross(tp, [0, 1, 0])), 0.55)), [0, 0.2, 0])), sz * 6)), // three-quarter lit: on VEGA's side, off to one flank
         to = sub(stand, s.pos), D = len(to), d = nrm(sub(tp, s.pos));
      const ax = cross(s.fwd, d), sa = len(ax), ang = Math.atan2(sa, dot(s.fwd, d));
      if (sa > 1e-5) { const k = nrm(ax), stepA = Math.min(ang, 1.6 * dt * (0.4 + Math.min(1, ang))); s.fwd = nrm(rot(s.fwd, k, stepA)); s.up = nrm(rot(s.up, k, stepA)); }
      // it travels along the line to its standing point (not along the nose, which looks at the planet)
      s.want = clamp(D * 0.9, 0, 120) * (ang > 1.2 ? 0.25 : 1); s.speed += (s.want - s.speed) * (1 - Math.exp(-dt * 1.8));
      s.pos = add(s.pos, mul(nrm(to), Math.min(D, s.speed * dt)));
      if (D < 0.4 && ang < 0.05) { s.hold = s.auto; s.holdP = tp; s.auto = null; s.want = 0; s.speed = 0; if (window.VegaAudio) window.VegaAudio.tick(); } // arrived: keep station
      s.fwd = nrm(s.fwd); const rr = nrm(cross(s.fwd, s.up)); s.up = nrm(cross(rr, s.fwd));
      gal.cam = { eye: s.pos, fwd: s.fwd, up: s.up }; status(pl); return;
    }
    // station-keeping: once arrived, ride along the planet's orbit, nose on it, until the pilot takes the stick
    if (s.hold != null) {
      if (iy || ip || ir || keys.w || keys.s || keys.Shift) s.hold = null;
      else { const tp = pl[s.hold].p; s.pos = add(s.pos, sub(tp, s.holdP)); s.holdP = tp;
        const d = nrm(sub(tp, s.pos)), ax = cross(s.fwd, d), sa = len(ax); if (sa > 1e-5) { const kk = nrm(ax), a2 = Math.atan2(sa, dot(s.fwd, d)); s.fwd = nrm(rot(s.fwd, kk, a2)); s.up = nrm(rot(s.up, kk, a2)); }
        gal.cam = { eye: s.pos, fwd: s.fwd, up: s.up }; status(pl); return; }
    }
    const k = 1 - Math.exp(-dt * 4), av = s.av; av[0] += (iy * 0.9 - av[0]) * k; av[1] += (ip * 0.7 - av[1]) * k; av[2] += (ir * 1.2 - av[2]) * k;
    if (Math.abs(av[0]) > 1e-4) { s.fwd = rot(s.fwd, s.up, av[0] * dt); }
    right = nrm(cross(s.fwd, s.up));
    if (Math.abs(av[1]) > 1e-4) { s.fwd = rot(s.fwd, right, av[1] * dt); s.up = rot(s.up, right, av[1] * dt); }
    if (Math.abs(av[2]) > 1e-4) { s.up = rot(s.up, s.fwd, av[2] * dt); }
    s.fwd = nrm(s.fwd); right = nrm(cross(s.fwd, s.up)); s.up = nrm(cross(right, s.fwd)); // keep the basis square
    // throttle: W / S scale the wanted speed; Shift boosts (for the galaxy beyond the system)
    if (keys.w) s.want = clamp(s.want + dt * (3 + s.want * 0.45), 0, 120);
    if (keys.s) s.want = Math.max(0, s.want - dt * (4 + s.want * 0.9));
    const top = keys.Shift ? s.want * 6 + 30 : s.want;
    s.speed += (top - s.speed) * (1 - Math.exp(-dt * 1.8));
    let pos = add(s.pos, mul(s.fwd, s.speed * dt));
    // no flying through a body: VEGA and the planets push the ship back out
    const push = (c, r) => { const v = sub(pos, c), l = len(v); if (l < r) pos = add(c, mul(nrm(v), r)); };
    push([0, 0, 0], 4); pl.forEach(p => push(p.p, p.size * 1.4));
    s.pos = pos;
    gal.cam = { eye: s.pos, fwd: s.fwd, up: s.up };
    status(pl);
  }
  function status(pl) {
    const s = ship, P = ui.P, dt = 1 / 60;
    // near a planet: LAND
    let near = null, nd = 1e9; pl.forEach((p, i) => { const d = len(sub(p.p, s.pos)); if (d < p.size * 9 && d < nd) { nd = d; near = i; } });
    if (s.dest != null && len(sub(pl[s.dest].p, s.pos)) < pl[s.dest].size * 9) near = s.dest; // the planet flown to wins over a neighbour
    if (near !== s.near) { s.near = near; ui.landB.classList.toggle('on', near != null);
      if (near != null) { ui.landB.innerHTML = '<span>LAND · ENTER</span><b class="vfl-land-n">' + pad(near + 1) + '</b>'; const t = el('span', 'vfl-land-n'); t.textContent = P[near].label; ui.landB.appendChild(t); ui.landB.setAttribute('aria-label', 'LAND · ' + P[near].label); } }
    // readouts (units only: 1 unit = 1 AU) and the drone
    s.mt += dt; if (s.mt > 0.1) { s.mt = 0; // (frame count, not time: readouts ~6 times a second)
      const hdg = (Math.atan2(s.fwd[0], -s.fwd[2]) * 180 / Math.PI + 360) % 360, tgt = s.auto != null ? s.auto : near;
      ui.rd.spd.textContent = s.speed.toFixed(1).padStart(5, '0') + ' AU/S'; ui.rd.hdg.textContent = String(Math.round(hdg)).padStart(3, '0') + '°';
      ui.rd.vega.textContent = len(s.pos).toFixed(1) + ' AU'; ui.rd.tgt.textContent = tgt != null ? pad(tgt + 1) + ' · ' + len(sub(pl[tgt].p, s.pos)).toFixed(1) + ' AU' : '—';
      if (window.VegaAudio && window.VegaAudio.motion) window.VegaAudio.motion(clamp(s.speed / 60, 0, 1));
    }
  }

  // labels at the planets' projections; VEGA's direction on the frame's edge when it's off screen
  function labels(f) {
    if (!ui || !f) return;
    const pl = gal.planets(), W = f.W, H = f.H, M = f.PV;
    const proj = (p) => { const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13], w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
      return w > 0.05 ? [(x / w * 0.5 + 0.5) * W, (0.5 - y / w * 0.5) * H, w] : null; };
    // labels as pinned readouts that never overprint: the target first, then VEGA, then the nearer planets; each tries
    // right, left (leader on the planet side), above and below its disc, and takes the first spot clear of the frame's
    // edges, the labels already placed and every other disc. A non-target label with no free spot is left out.
    const small = W <= 600, fr = [small ? 28 : 34, (small ? 12 : 24) + 58, W - (small ? 28 : 34), H - (small ? 12 : 24) - 60];
    const discs = [], items = [], tgtOf = (i) => ship && (ship.auto === i || ship.near === i);
    pl.forEach((p, i) => { const q = proj(p.p); ui.labs[i].classList.toggle('tgt', !!tgtOf(i)); if (!q) { ui.labs[i].style.visibility = 'hidden'; return; }
      const r = p.size / (q[2] * Math.tan(0.45)) * H / 2; discs.push([q[0], q[1], r, i]); items.push({ node: ui.labs[i], q, r, i, pri: tgtOf(i) ? 0 : 2 + q[2] / 1e4 }); });
    const vq = proj([0, 0, 0]); if (vq) { discs.push([vq[0], vq[1], 12, -1]); items.push({ node: ui.vega, q: vq, r: 12, i: -1, pri: 1 }); } else ui.vega.style.visibility = 'hidden';
    items.sort((a, b) => a.pri - b.pri);
    // the cockpit's own controls (planet list / chips, readouts, help, BACK, the touch thrust buttons) are taken too;
    // their boxes are read twice a second, not every frame
    const now = performance.now(); if (!ui.boxAt || now - ui.boxAt > 500) { ui.boxAt = now; ui.boxes = [...root.querySelectorAll('.vfl-panel, .vfl-read, .vfl-help, .vfl-exit, .vfl-thr' + (W <= 700 ? ', .vfl-land.on' : ''))].map(e => e.getBoundingClientRect()).filter(b => b.width && b.height).map(b => [b.left, b.top, b.right, b.bottom]); }
    const placed = ui.boxes.slice(), hitR = (x, y, w, h) => placed.some(o => x < o[2] + 6 && x + w + 6 > o[0] && y < o[3] + 4 && y + h + 4 > o[1]);
    const hitD = (x, y, w, h, own) => discs.some(d => { if (d[3] === own) return false; const cx = Math.max(x, Math.min(d[0], x + w)), cy = Math.max(y, Math.min(d[1], y + h)); return Math.hypot(cx - d[0], cy - d[1]) < d[2] + 4; });
    items.forEach(it => {
      const { node, q, r } = it, w = node._w || (node._w = node.offsetWidth), lh = ship && it.i === ship.near && innerWidth > 700 ? 48 : 20; // the near planet's label carries LAND under it
      let q2 = q, r2 = r;
      if (q[0] < -40 || q[0] > W + 40 || q[1] < -20 || q[1] > H + 20) {
        // off frame: only the target keeps a label, anchored to the visible part of its disc (or the frame edge toward it)
        if (it.pri !== 0) { node.style.visibility = 'hidden'; return; }
        q2 = [Math.min(Math.max(q[0], fr[0]), fr[2]), Math.min(Math.max(q[1], fr[1] + 10), fr[3] - 10)]; r2 = 0;
      }
      const g = Math.max(8, r2 + 8), C = [['r', q2[0] + g, q2[1] - 10], ['l', q2[0] - g - w, q2[1] - 10], ['r', q2[0] - w / 2, q2[1] - g - lh], ['r', q2[0] - w / 2, q2[1] + g]];
      let pick = C.find(([, x, y]) => x >= fr[0] && x + w <= fr[2] && y >= fr[1] && y + lh <= fr[3] && !hitR(x, y, w, lh) && !hitD(x, y, w, lh, it.i));
      if (!pick && it.pri === 0) { const c = C.find(([, x, y]) => x >= fr[0] && x + w <= fr[2] && !hitR(x, y, w, lh)) || C[0]; pick = [c[0], Math.min(Math.max(fr[0], c[1]), fr[2] - w), Math.min(Math.max(fr[1], c[2]), fr[3] - lh)]; } // the target always keeps its label, inside the frame
      if (!pick) { node.style.visibility = 'hidden'; return; }
      placed.push([pick[1], pick[2], pick[1] + w, pick[2] + lh]);
      node.classList.toggle('left', pick[0] === 'l'); node.style.visibility = 'visible';
      node.style.transform = 'translate(' + pick[1].toFixed(1) + 'px,' + (pick[2] - 2).toFixed(1) + 'px)';
    });
    // LAND sits in the scene, under the near planet's own label (on phones it docks above the planet chips)
    if (ui.landB && ship && ship.near != null && innerWidth > 700) { const lb = ui.labs[ship.near];
      ui.landB.style.transform = lb.style.visibility === 'visible' ? lb.style.transform.replace(/,\s*([-\d.]+)px\)/, (m, y) => ',' + (+y + 26).toFixed(1) + 'px)') : 'translate(' + (W / 2 - 80) + 'px,' + (H / 2 + 64) + 'px)'; }
    if (ui.edge) { // VEGA off screen: a chevron on the frame toward it
      const vOn = vq && vq[0] > -40 && vq[0] < W + 40 && vq[1] > -20 && vq[1] < H + 20, q = vOn ? null : (() => { const x = M[0] * 0 + M[12], y = M[13], w = M[15]; let dx = x / Math.abs(w || 1), dy = -y / Math.abs(w || 1); if (w < 0) { dx = -dx; dy = -dy; } return [dx, dy]; })();
      if (q) { const a = Math.atan2(q[1], q[0]), rx = W / 2 - 54, ry = H / 2 - 54, t = Math.min(rx / Math.abs(Math.cos(a) || 1e-6), ry / Math.abs(Math.sin(a) || 1e-6));
        ui.edge.style.display = 'block'; ui.edge.style.transform = 'translate(' + (W / 2 + Math.cos(a) * t - 7).toFixed(1) + 'px,' + (H / 2 + Math.sin(a) * t - 7).toFixed(1) + 'px) rotate(' + a.toFixed(3) + 'rad)'; }
      else ui.edge.style.display = 'none';
    }
  }

  function engage(i) {
    ui.list.forEach((b, k) => b.setAttribute('aria-current', String(k === i)));
    if (!gal || RM.matches) { land(i); return; } // no flight: the map's planets link straight to their pages
    if (ship.near === i) { land(i); return; }
    ship.auto = i; ship.dest = i; ship.hold = null;
  }
  function land(i) {
    const p = ui.P[i]; if (!p) return;
    if (p.href.replace(/\.dc\.html$/, '').split('/').pop() === here() && saved) { close(); return; } // landing on this page: fly back into its view
    landing = true;
    let x = innerWidth / 2, y = innerHeight / 2;
    if (gal && gal.frame) { const pp = gal.planets()[i].p, M = gal.frame.PV, w = M[3] * pp[0] + M[7] * pp[1] + M[11] * pp[2] + M[15];
      if (w > 0) { x = ((M[0] * pp[0] + M[4] * pp[1] + M[8] * pp[2] + M[12]) / w * 0.5 + 0.5) * innerWidth; y = (0.5 - (M[1] * pp[0] + M[5] * pp[1] + M[9] * pp[2] + M[13]) / w * 0.5) * innerHeight; } }
    if (ui.labs) ui.labs.forEach(l => { l.style.transition = 'opacity .3s'; l.style.opacity = '0'; });
    if (window.VegaFly) window.VegaFly.go(p.href, { x, y, galaxy: gal || undefined }); else location.href = p.href;
  }

  function close() {
    if (!root || closing) return;
    const back2 = saved && saved.from;
    if (gal && !own && back2 && ship && !RM.matches && !landing) { // fly back into the page's view; the page fades in as the camera arrives
      closing = true; root.classList.remove('live'); keys = {}; drag = null;
      document.documentElement.classList.remove('vfl-deep'); document.documentElement.classList.add('vfl-closing');
      const c = gal.cam || { eye: ship.pos, fwd: ship.fwd, up: ship.up };
      ship.move = { t: 0, dur: 1.7, from: { eye: c.eye.slice(), fwd: nrm(c.fwd), up: c.up || [0, 1, 0], ax: c.ax || 0, ay: c.ay || 0 }, to: back2, done: finish };
      setTimeout(() => document.documentElement.classList.remove('vfl-active'), 1000); // (the text returns over the last of the move)
      if (window.VegaAudio && window.VegaAudio.motion) window.VegaAudio.motion(0);
      return; }
    finish();
  }
  let closing = false;
  function finish() {
    if (!root) return;
    const r = root; root = null; ship = null; drag = null; keys = {}; landing = false; closing = false;
    removeEventListener('keydown', onKey, true); removeEventListener('keyup', onKeyUp, true);
    if (gal) { gal.tick = null; gal.onframe = null; if (!own) { gal.cam = null; if (saved) { gal.drive = saved.drive; gal.system = saved.system; gal.res = saved.res; gal.mark = saved.mark; gal.streak = saved.streak; gal.exposure = saved.exposure; } } }
    saved = null; own = false;
    r.classList.remove('on', 'live'); setTimeout(() => r.remove(), RM.matches ? 0 : 640); gal = null; ui = null;
    const de = document.documentElement; de.classList.remove('vfl-active', 'vfl-deep'); de.classList.add('vfl-closing'); setTimeout(() => de.classList.remove('vfl-closing'), 700);
    if (back) { scrollTo({ top: back.y, behavior: 'instant' }); }
    if (window.VegaAudio && window.VegaAudio.motion) window.VegaAudio.motion(0);
    dispatchEvent(new CustomEvent('vegaflight', { detail: { on: false } }));
    const hud = document.querySelector('vega-hud'); if (hud && hud.focusFly) hud.focusFly();
  }

  window.VegaFlight = { open, close, toggle: () => (root ? close() : open()), get active() { return !!root; }, get ship() { return ship; } };
})();
