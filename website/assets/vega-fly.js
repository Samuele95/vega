/* vega-fly — page change = the warp (spec §2.8, galaxy plan G3). On an internal page link the ship jumps: the text and
   the page's scenes fall away, the camera runs forward through the 3D galaxy (vega-galaxy, WebGL) as every star
   stretches into a streak toward the vanishing point (the star the visitor picked, or where they clicked), a flash,
   then the browser navigates. The incoming page opens inside the same jump, the streaks slowing back into stars as the
   cover lifts. A page without its own galaxy gets one for the jump (vega-galaxy.js is preloaded at idle). Without WebGL
   it falls back to the C3 bloom-and-fade; under prefers-reduced-motion there is none of it (links navigate at once, no
   cover). Not on the session's first entry (the loader plays then).
   Load as a classic script at the top of <body>, after vega-loader.js.
   A link can name the node it dives into with data-fly-node="<index>"; otherwise the jump aims where the visitor clicked
   (or mid-screen for keyboard).
   API: window.VegaFly.go(url, { node, x, y }). */
(() => {
  if (window.VegaFly) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const KEY = 'vega-fly', DUR = 400, WARP = 1150, ARRIVE = 1000;
  const here = new URL('.', location.href).href;
  const css = document.createElement('style');
  css.textContent = '.vf{position:fixed;inset:0;z-index:99;pointer-events:none;background:#04080E}.vf vega-galaxy{position:absolute;inset:0}'
    // the jump: text, scenes, readouts and rail fall away; the galaxy and the HUD stay (the cockpit)
    + 'html.vf-jump main,html.vf-jump footer,html.vf-jump vega-constellation,html.vf-jump vega-orbits,html.vf-jump vega-sky-lab,html.vf-jump vega-readouts,html.vf-jump vega-rail,html.vf-jump [data-brand],html.vf-jump .vd-ov{transition:opacity .35s ease!important;opacity:0!important}'
    + '.vf-out{position:fixed;inset:0;z-index:99;pointer-events:none}.vf-out canvas{width:100%;height:100%;display:block}';
  document.head.appendChild(css);

  // vega-galaxy for pages that don't mount one themselves (the jump needs it); a WebGL probe decides the fallback
  const me = document.currentScript && document.currentScript.src;
  let galP = null;
  const loadGalaxy = () => galP || (galP = new Promise((res) => {
    if (customElements.get('vega-galaxy')) return res(true);
    if (!me) return res(false);
    const sc = document.createElement('script'); sc.src = new URL('vega-galaxy.js', me).href;
    sc.onload = () => res(!!customElements.get('vega-galaxy')); sc.onerror = () => res(false); document.head.appendChild(sc);
  }));
  const freshGalaxy = (host, view) => { const g = document.createElement('vega-galaxy'); host.appendChild(g); if (g.failed || !g.ps) { g.remove(); return null; } g.res = 0.85; g.drive = () => view; return g; };
  if (!RM.matches) (window.requestIdleCallback || setTimeout)(() => loadGalaxy(), { timeout: 2500 });

  // ---- incoming: cover in --bg, fade up once the page has mounted and painted
  let flown = null; try { flown = JSON.parse(sessionStorage.getItem(KEY) || 'null'); sessionStorage.removeItem(KEY); } catch (e) {}
  const arrived = !!(flown && Date.now() - flown.t < 6000 && !RM.matches); let arriving = arrived; // the page's galaxy flies in to its view as the cover lifts (vega-galaxy home)
  if (arrived) {
    const cover = document.createElement('div'); cover.className = 'vf'; document.body.appendChild(cover);
    // arrive inside the jump: full streaks, slowing back into stars, then the cover lifts on the page
    let gal = null, ta = 0;
    loadGalaxy().then(ok => { if (!ok || !cover.isConnected) return; gal = freshGalaxy(cover, { dist: 520, yaw: 0.8, pitch: 0.24 }); if (gal) { gal.cam = { eye: [640, 26, 220], fwd: [-0.42, -0.03, -1], up: [0, 1, 0] }; gal.bloomK = 0.5; gal.warp = 0.75; ta = performance.now(); // along the outer disk, not into the bulge
      const slow = (now) => { if (!gal.isConnected) return; const u = Math.min(1, (now - ta) / ARRIVE); gal.warp = 0.75 * Math.pow(1 - u, 2.2); if (u < 1) requestAnimationFrame(slow); }; requestAnimationFrame(slow); } });
    const lift = () => { const late = gal && ta ? Math.max(0, ARRIVE * 0.55 - (performance.now() - ta)) : 0;
      setTimeout(() => { arriving = false; cover.style.transition = 'opacity .7s cubic-bezier(.2,.8,.2,1)'; cover.style.opacity = '0'; setTimeout(() => cover.remove(), 800); }, late); };
    const t0 = performance.now();
    const wait = () => { const r = document.getElementById('dc-root'); if ((r && r.firstElementChild && (!document.fonts || document.fonts.status === 'loaded')) || performance.now() - t0 > 2500) requestAnimationFrame(() => requestAnimationFrame(lift)); else setTimeout(wait, 40); };
    wait();
  }

  // ---- outgoing
  let flying = false;
  const internal = (a) => {
    if (!a || !a.href || a.target === '_blank' || a.hasAttribute('download')) return false;
    const u = new URL(a.href, location.href);
    if (u.origin !== location.origin || !/\.dc\.html$/.test(u.pathname)) return false;
    if (u.pathname === location.pathname && u.hash) return false; // in-page jump
    return true;
  };
  function go(url, opt) {
    opt = opt || {};
    const dest = new URL(url, location.href);
    if (RM.matches || flying || (window.VegaLoader && window.VegaLoader.active)) { location.href = dest.href; return; }
    flying = true;
    // the incoming page fades up only if it is one of the site's own pages (same folder)
    if (dest.href.startsWith(here)) { try { sessionStorage.setItem(KEY, JSON.stringify({ t: Date.now() })); } catch (e) {} }
    let x = opt.x, y = opt.y;
    const sc = document.querySelector('vega-constellation');
    if (opt.node != null && sc && sc.P && sc.P[opt.node]) { x = sc.P[opt.node][0]; y = sc.P[opt.node][1]; }
    if (x == null) { x = innerWidth / 2; y = innerHeight / 2; }
    const cam = document.querySelector('vega-camera');
    const own = opt.galaxy || document.querySelector('vega-galaxy:not(.vf vega-galaxy)');
    const warp = (gal, host) => { // the 3D jump
      if (cam && cam.dive) cam.dive(WARP * 0.6);
      document.documentElement.classList.add('vf-jump');
      gal.warpAt = [x, y]; gal.res = 0.85; gal.exposure = 1;
      const fx = document.createElement('div'); fx.className = 'vf-out'; const cv = document.createElement('canvas'); fx.appendChild(cv); document.body.appendChild(fx);
      const dpr = Math.min(2, devicePixelRatio || 1), W = innerWidth, H = innerHeight; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      const ctx = cv.getContext('2d'), t0 = performance.now();
      const frame = (now) => {
        const u = Math.min(1, (now - t0) / WARP);
        gal.warp = 1.15 * Math.pow(u, 1.6); // the drive builds, then the streaks run long
        if (host) host.style.opacity = Math.min(1, u * 4).toFixed(3);
        // the flash at the vanishing point and the cut to --bg, over the last quarter
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
        const f = Math.max(0, (u - 0.72) / 0.28), ff = f * f * (3 - 2 * f);
        if (f > 0) { const R = Math.hypot(W, H), g = ctx.createRadialGradient(x, y, 0, x, y, 30 + ff * R);
          g.addColorStop(0, 'rgba(233,240,248,' + (0.95 * Math.sin(f * Math.PI * 0.9)).toFixed(3) + ')'); g.addColorStop(0.25, 'rgba(156,200,255,' + (0.5 * Math.sin(f * Math.PI)).toFixed(3) + ')'); g.addColorStop(1, 'rgba(156,200,255,0)');
          ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); ctx.fillStyle = 'rgba(4,8,14,' + Math.max(0, (f - 0.45) / 0.55).toFixed(3) + ')'; ctx.fillRect(0, 0, W, H); }
        if (u < 1) requestAnimationFrame(frame); else location.href = dest.href;
      };
      requestAnimationFrame(frame);
    };
    if (own && !own.failed && own.ps && own.offsetParent !== null) { warp(own, null); return; }
    // no galaxy of its own: open one over the page for the jump (a WebGL page), else the bloom
    loadGalaxy().then(ok => {
      if (ok) { const host = document.createElement('div'); host.className = 'vf'; host.style.opacity = '0'; host.style.zIndex = '98'; document.body.appendChild(host);
        const g = freshGalaxy(host, { dist: 560, yaw: Math.random() * 6.28, pitch: 0.22 }); if (g) { warp(g, host); return; } host.remove(); }
      bloom();
    });
    // the fallback (no WebGL): C3's bloom + fade, drawn on one canvas
    function bloom() {
    if (cam && cam.dive) cam.dive(DUR);
    const wrap = document.createElement('div'); wrap.className = 'vf-out';
    const cv = document.createElement('canvas'); wrap.appendChild(cv); document.body.appendChild(wrap);
    const dpr = Math.min(2, devicePixelRatio || 1), W = innerWidth, H = innerHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const ctx = cv.getContext('2d'), R = Math.hypot(Math.max(x, W - x), Math.max(y, H - y)), t0 = performance.now();
    const frame = (now) => {
      const u = Math.min(1, (now - t0) / DUR), e = u * u * (3 - 2 * u);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const r = 20 + e * R * 1.1, g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(233,240,248,' + (0.9 * (1 - u * 0.6)).toFixed(3) + ')');
      g.addColorStop(0.18, 'rgba(156,200,255,' + (0.55 * (1 - u * 0.3)).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(156,200,255,0)');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      const f = Math.max(0, (u - 0.35) / 0.65); ctx.fillStyle = 'rgba(4,8,14,' + (f * f * (3 - 2 * f)).toFixed(3) + ')'; ctx.fillRect(0, 0, W, H);
      if (u < 1) requestAnimationFrame(frame); else location.href = dest.href;
    };
    requestAnimationFrame(frame);
    }
  }
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest('a[href]'); if (!internal(a)) return;
    e.preventDefault();
    const n = a.getAttribute('data-fly-node');
    go(a.href, { node: n != null ? +n : null, x: e.detail ? e.clientX : null, y: e.detail ? e.clientY : null });
  });
  // back/forward cache: never come back to a page still covered
  addEventListener('pageshow', (e) => { if (e.persisted) { flying = false; document.documentElement.classList.remove('vf-jump'); document.querySelectorAll('vega-galaxy').forEach(g => { g.warp = 0; g.warpAt = null; g.res = null; }); document.querySelectorAll('.vf-out,.vf').forEach(n => n.remove()); const cam = document.querySelector('vega-camera'); if (cam && cam.dive) cam.dive(0); } });
  window.VegaFly = { go, loadGalaxy, asset: (n) => (me ? new URL(n, me).href : n), arrived, get arriving() { return arriving; } };
})();
