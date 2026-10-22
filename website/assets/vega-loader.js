/* vega-loader — the loader and ENTER gate (spec §2.1). Load as a classic script at the start of <body>,
   after vega-scramble.js and vega-audio.js. Once per session (sessionStorage 'vega-entered') on whichever
   page is entered first; later loads skip straight in.
   Ten squares, one per real milestone (the chain's ten stages): document parsed, React, ReactDOM, page
   mounted, the three type families, components defined, the first sky frame, window load. A milestone
   counts when it settles (loaded or failed), so a missing resource never hangs the gate.
   API: window.VegaLoader { active, entered: Promise } and window event 'vegaenter' when the visitor enters. */
(() => {
  if (window.VegaLoader) return;
  const KEY = 'vega-entered';
  let seen = false;
  try { seen = sessionStorage.getItem(KEY) === '1'; } catch (e) {}
  let resolveEntered;
  const L = window.VegaLoader = { active: !seen, entered: new Promise(r => { resolveEntered = r; }), milestones: [] };
  if (seen) { resolveEntered(); return; }

  const rm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const base = new URL('.', document.currentScript.src);
  const S = window.VegaScramble || { run: (el, t) => { el.textContent = t; return Promise.resolve(); } };

  // the type, early, so the loader itself renders in Martian Mono
  const fontsLink = document.createElement('link');
  fontsLink.rel = 'stylesheet'; fontsLink.href = new URL('fonts/fonts.css', base).href;
  const fontsCss = new Promise(r => { fontsLink.onload = fontsLink.onerror = r; });
  document.head.appendChild(fontsLink);

  const css = document.createElement('style');
  css.textContent = `
.vl{--bg:#04080E;--bg2:#06171C;--fg:#E9F0F8;--mut:#A3B4C7;--dim:#8C9EB3;--acc:#9CC8FF;--line:rgba(156,200,255,.18);position:fixed;inset:0;z-index:100;display:grid;place-items:center;color:var(--dim);font:500 11px/1 'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;letter-spacing:.22em;transition:opacity .7s cubic-bezier(.2,.8,.2,1)}
.vl-bg{position:absolute;inset:0;background:radial-gradient(110% 80% at 50% 36%,var(--bg2) 0%,var(--bg) 62%);transition:opacity 1.6s ease}
.vl.sky .vl-bg{opacity:.8}
.vl-col{position:relative;display:flex;flex-direction:column;align-items:center;gap:44px}
.vl-mark{display:block;width:84px;height:84px;overflow:visible;color:var(--fg);filter:drop-shadow(0 0 14px rgba(156,200,255,.35))}
.vl-box{width:244px}
.vl-row{display:flex;align-items:center;gap:6px}
.vl-sq{width:10px;height:10px;box-sizing:border-box;border:1px solid var(--line);background:rgba(156,200,255,.04);transition:background .3s,border-color .3s,box-shadow .3s}
.vl-sq.on{background:var(--acc);border-color:var(--acc);box-shadow:0 0 10px rgba(156,200,255,.55)}
.vl-sq.next{animation:vl-breathe 1.2s ease-in-out infinite}
@keyframes vl-breathe{50%{border-color:rgba(156,200,255,.6);background:rgba(156,200,255,.16)}}
.vl-pct{margin-left:auto;color:var(--fg);font-variant-numeric:tabular-nums}
.vl-lines{margin:18px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:11px;min-height:55px}
.vl-lines li{white-space:pre}
.vl-lines li:last-child{color:var(--fg)}
.vl-enter{position:relative;margin-top:30px;appearance:none;background:none;border:0;padding:15px 36px 15px 38px;cursor:pointer;color:var(--fg);font:500 12px/1 'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;letter-spacing:.32em;opacity:0;transform:translateY(6px);transition:opacity .5s,transform .5s;visibility:hidden}
.vl-enter.show{opacity:1;transform:none;visibility:visible}
.vl-enter .vl-ch{position:absolute;inset:0;background:rgba(156,200,255,.6);clip-path:polygon(11px 0,100% 0,100% 100%,0 100%,0 11px)}
.vl-enter .vl-ch::after{content:'';position:absolute;inset:1px;background:var(--bg);clip-path:polygon(10.6px 0,100% 0,100% 100%,0 100%,0 10.6px);transition:background .25s}
.vl-enter:hover .vl-ch::after,.vl-enter:focus-visible .vl-ch::after{background:#0C1A27}
.vl-enter:hover .vl-ch,.vl-enter:focus-visible .vl-ch{background:var(--acc)}
.vl-enter:focus-visible{outline:1.5px solid var(--acc);outline-offset:5px;border-radius:2px}
.vl-enter span:last-child{position:relative}
.vl-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap}
.vl.out{opacity:0;pointer-events:none}
/* behind the gate the page's own content waits (only the faint sky, scene and HUD frame show through) */
html.vl-holding main,html.vl-holding footer,html.vl-holding [data-brand],html.vl-holding [data-pagebar],html.vl-holding [data-ticks],html.vl-holding [data-fighud],html.vl-holding vega-readouts,html.vl-holding vega-rail{opacity:0!important}
html.vl-released main,html.vl-released footer,html.vl-released [data-brand],html.vl-released [data-pagebar],html.vl-released [data-ticks],html.vl-released [data-fighud],html.vl-released vega-readouts,html.vl-released vega-rail{transition:opacity .6s ease}
@media (prefers-reduced-motion: reduce){.vl,.vl *{transition:none!important;animation:none!important}}
`;
  document.head.appendChild(css);

  // the star mark — same geometry as the hero's, drawn in with the vega-intro keyframes
  const MARK = '<svg class="vl-mark" viewBox="0 0 64 64" aria-hidden="true">'
    + '<polygon data-p="sv" fill="currentColor" points="32,5 33.5,32 32,59 30.5,32" style="transform-box:fill-box;transform-origin:center"/>'
    + '<polygon data-p="sh" fill="currentColor" points="9,32 32,30.6 55,32 32,33.4" style="transform-box:fill-box;transform-origin:center"/>'
    + '<g stroke="var(--acc)" stroke-width="1">' + [[27.8, 27.8, 18, 18], [36.2, 27.8, 46, 18], [36.2, 36.2, 46, 46], [27.8, 36.2, 18, 46]].map(l => '<line data-p="dl" x1="' + l[0] + '" y1="' + l[1] + '" x2="' + l[2] + '" y2="' + l[3] + '" stroke-dasharray="14" stroke-dashoffset="0"/>').join('') + '</g>'
    + '<g fill="var(--acc)">' + [[18, 18], [46, 18], [46, 46], [18, 46]].map(c => '<circle data-p="du" cx="' + c[0] + '" cy="' + c[1] + '" r="2" style="transform-box:fill-box;transform-origin:center"/>').join('') + '</g>'
    + '<g fill="currentColor">' + [[32, 5], [55, 32], [32, 59], [9, 32]].map(c => '<circle data-p="tu" cx="' + c[0] + '" cy="' + c[1] + '" r="2.8" style="transform-box:fill-box;transform-origin:center"/>').join('') + '</g>'
    + '<circle data-p="core" cx="32" cy="32" r="5.2" fill="currentColor" style="transform-box:fill-box;transform-origin:center"/></svg>';

  const root = document.createElement('div');
  root.className = 'vl';
  root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'true'); root.setAttribute('aria-label', 'VEGA');
  root.innerHTML = '<div class="vl-bg"></div><div class="vl-col">' + MARK
    + '<div class="vl-box"><div class="vl-row" aria-hidden="true">' + '<span class="vl-sq"></span>'.repeat(10) + '<span class="vl-pct">000%</span></div>'
    + '<ul class="vl-lines" aria-hidden="true"><li></li></ul><p class="vl-sr" aria-live="polite"></p>'
    + '<button type="button" class="vl-enter"><span class="vl-ch" aria-hidden="true"></span><span>ENTER</span></button></div></div>';
  document.body.appendChild(root);
  const sq = [...root.querySelectorAll('.vl-sq')], pct = root.querySelector('.vl-pct'), lines = root.querySelector('.vl-lines'),
    live = root.querySelector('.vl-sr'), btn = root.querySelector('.vl-enter');
  const prevOverflow = document.documentElement.style.overflow;
  document.documentElement.classList.add('vl-holding');
  document.documentElement.style.overflow = 'hidden';

  const say = (text) => {
    const li = lines.lastElementChild.textContent ? lines.appendChild(document.createElement('li')) : lines.lastElementChild;
    live.textContent = text;
    return S.run(li, text, { duration: 520 });
  };

  // ---- the mark draws in
  let markDone = Promise.resolve();
  const svg = root.querySelector('.vl-mark');
  if (!rm && svg.animate) {
    const Q = (s) => [...svg.querySelectorAll(s)], one = (s) => svg.querySelector(s), anims = [];
    const an = (el, kf, o) => { anims.push(el.animate(kf, Object.assign({ fill: 'both', easing: 'cubic-bezier(.2,.8,.2,1)' }, o))); };
    const POP = [{ transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1.9)', opacity: 1, offset: 0.55 }, { transform: 'scale(1)', opacity: 1 }];
    const t0 = 200;
    an(one('[data-p="core"]'), [{ transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1.7)', opacity: 1, offset: 0.6 }, { transform: 'scale(1)', opacity: 1 }], { duration: 380, delay: t0 });
    an(one('[data-p="sv"]'), [{ transform: 'scaleY(0)' }, { transform: 'scaleY(1)' }], { duration: 420, delay: t0 + 170 });
    an(one('[data-p="sh"]'), [{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 420, delay: t0 + 260 });
    Q('[data-p="tu"]').forEach((e, i) => an(e, POP, { duration: 320, delay: t0 + 420 + i * 60 }));
    Q('[data-p="dl"]').forEach((e, i) => an(e, [{ strokeDashoffset: 14 }, { strokeDashoffset: 0 }], { duration: 240, delay: t0 + 680 + i * 55, easing: 'ease-out' }));
    Q('[data-p="du"]').forEach((e, i) => an(e, POP, { duration: 300, delay: t0 + 840 + i * 55 }));
    markDone = Promise.all(anims.map(a => a.finished.catch(() => {})));
  }
  say('LOADING CONTENT');

  // ---- ten real milestones
  const TOTAL = 10; let settled = 0;
  const track = (name, p) => Promise.resolve(p).then(() => 'ok', () => 'failed').then((how) => { settled++; L.milestones.push([name, how, Math.round(performance.now())]); });
  const poll = (test, ms = 30000) => new Promise((res, rej) => { const t0 = performance.now(); const f = () => { let ok = false; try { ok = test(); } catch (e) {} if (ok) res(); else if (performance.now() - t0 > ms) rej(); else setTimeout(f, 40); }; f(); });
  const dcl = document.readyState === 'loading' ? new Promise(r => document.addEventListener('DOMContentLoaded', r, { once: true })) : Promise.resolve();
  const mounted = poll(() => { const r = document.getElementById('dc-root'); return r && r.firstElementChild; });
  const face = (fam, ws) => fontsCss.then(() => Promise.all(ws.map(w => document.fonts.load(w + ' 16px "' + fam + '"')))).then(l => { if (!l.every(x => x.length)) throw 0; });
  track('document', dcl);
  track('react', poll(() => window.React));
  track('react-dom', poll(() => window.ReactDOM));
  track('mounted', mounted);
  track('type-display', face('Newsreader', [300, 400]));
  track('type-body', face('Hanken Grotesk', [400, 500]));
  track('type-label', face('Martian Mono', [400, 500]));
  // every component the page imports
  // (from <x-dc> if dc-runtime hasn't replaced it yet, else from dc-runtime's template-source bridge)
  const tpl = () => { const dc = document.querySelector('x-dc'); if (dc) return dc.innerHTML; try { return window.__dcTemplateSource(window.__dcRootName()) || null; } catch (e) { return null; } };
  const names = dcl.then(() => poll(() => tpl() !== null, 30000)).then(() => [...new Set([...tpl().matchAll(/component-from-global-scope="([^"]+)"/g)].map(m => m[1]))], () => []);
  const comps = names.then(ns => Promise.all(ns.map(n => Promise.race([customElements.whenDefined(n), new Promise((_, rj) => setTimeout(rj, 30000))]))));
  track('components', comps);
  // first sky frame: once the components are in, every vega-sky-lab has built its star field and painted once
  track('sky-frame', Promise.all([names, mounted, comps.catch(() => {})]).then(([ns]) => poll(() => { const s = [...document.querySelectorAll('vega-sky-lab')]; return ns.includes('vega-sky-lab') ? s.length && s.every(e => e.pts && e.pts.length) : true; })).then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))).then(() => root.classList.add('sky')));
  track('window-load', document.readyState === 'complete' ? 0 : new Promise(r => addEventListener('load', r, { once: true })));

  // keep the page inert and quiet behind the gate
  const inert = () => { const r = document.getElementById('dc-root'); if (r && L.active) r.inert = true; };
  mounted.then(inert, () => {});
  const onKey = (e) => {
    if (!L.active) return;
    if (e.key === 'Tab') { e.preventDefault(); if (btn.classList.contains('show')) btn.focus(); }
    e.stopPropagation(); // page shortcuts (i, Enter, arrows) wait; the focused ENTER still activates by default
  };
  addEventListener('keydown', onKey, true);

  // ---- counter: eases toward the settled share, never ahead of it
  let shown = 0, readyShown = false;
  const paint = () => {
    const target = settled / TOTAL * 100;
    shown = rm ? target : Math.min(target, shown + Math.max(0.6, (target - shown) * 0.12));
    const n = Math.floor(shown + 1e-6), lit = Math.floor(shown / 10 + 1e-6);
    pct.textContent = String(n).padStart(3, '0') + '%';
    sq.forEach((s, i) => { s.classList.toggle('on', i < lit); s.classList.toggle('next', i === lit && !rm); });
    if (n >= 100 && !readyShown) { readyShown = true; ready(); return; }
    if (!readyShown) requestAnimationFrame(paint);
  };
  requestAnimationFrame(paint);

  async function ready() {
    await markDone;
    await say('100% LOADED');
    if (!rm) await new Promise(r => setTimeout(r, 220));
    await say('READY TO EXPLORE');
    showEnter();
  }
  // a resource that never settles must not trap the visitor: after 25 s offer ENTER at the real percentage
  const late = setTimeout(() => { if (!readyShown) showEnter(); }, 25000);
  function showEnter() {
    clearTimeout(late);
    if (btn.classList.contains('show')) return;
    btn.classList.add('show');
    btn.focus({ preventScroll: true });
  }

  btn.addEventListener('click', () => {
    if (!L.active) return;
    L.active = false;
    try { sessionStorage.setItem(KEY, '1'); } catch (e) {}
    if (window.VegaAudio) window.VegaAudio.enter(); // inside the gesture: starts only if sound was left on
    removeEventListener('keydown', onKey, true);
    const r = document.getElementById('dc-root'); if (r) r.inert = false;
    document.documentElement.style.overflow = prevOverflow;
    document.documentElement.classList.add('vl-released'); document.documentElement.classList.remove('vl-holding');
    btn.blur();
    root.classList.add('out');
    dispatchEvent(new CustomEvent('vegaenter'));
    resolveEntered();
    setTimeout(() => root.remove(), rm ? 0 : 750);
  });
})();
