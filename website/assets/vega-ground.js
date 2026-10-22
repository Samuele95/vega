// <vega-ground> — the shared animated ground for Vega's inner pages.
// A field of faint stars that are also the units and weights of a network; light travels along a few connections.
// Pauses when the tab is hidden or the element is off-screen; still composition under prefers-reduced-motion.
// Attributes: level="quiet" (default) | "threshold"; seed (number).
(function () {
  if (customElements.get('vega-ground')) return;
  const rng = (s) => () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  class VegaGround extends HTMLElement {
    connectedCallback() {
      if (this._c) return;
      this.style.display = 'block'; this.style.width = '100%'; this.style.height = '100%';
      this.setAttribute('aria-hidden', 'true');
      const c = this._c = document.createElement('canvas');
      c.style.cssText = 'display:block;width:100%;height:100%';
      this.appendChild(c);
      const ctx = c.getContext('2d'); if (!ctx) return;
      const rm = matchMedia('(prefers-reduced-motion: reduce)');
      const quiet = (this.getAttribute('level') || 'quiet') !== 'threshold';
      const seed = +(this.getAttribute('seed') || 31);
      let W = 0, H = 0, cols = [], edges = [], stars = [], pulses = [];
      let raf = 0, last = 0, T = 0, visible = true, px = 0, py = 0, tx = 0, ty = 0;
      const k = quiet ? 0.55 : 1;
      const spawn = (p) => {
        const col = cols[Math.floor(Math.random() * Math.max(1, cols.length - 1))];
        const a = col[Math.floor(Math.random() * col.length)];
        p.a = a; p.b = a.out[Math.floor(Math.random() * a.out.length)]; p.t = 0;
        p.v = 0.00014 + Math.random() * 0.00018; p.life = 2 + Math.floor(Math.random() * 4);
        p.hue = Math.random() < 0.18 ? 'cy' : 'bw'; return p;
      };
      const build = () => {
        const dpr = Math.min(2, devicePixelRatio || 1);
        W = c.clientWidth; H = c.clientHeight; if (!W || !H) return;
        c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const r = rng(seed); const L = W < 700 ? 5 : 9;
        cols = [];
        for (let i = 0; i < L; i++) {
          const n = (W < 700 ? 4 : 6) + Math.floor(r() * 6); const col = [];
          for (let j = 0; j < n; j++) col.push({ x: (i + 0.5) / L * W + (r() - 0.5) * (W / L) * 0.8, y: (j + r()) / n * H, s: 0.5 + r() * 1.1, b: 0.4 + r() * 0.6, tw: r() * 6.28, out: [] });
          cols.push(col);
        }
        edges = [];
        for (let i = 0; i < L - 1; i++) for (const a of cols[i]) {
          const nx = cols[i + 1].slice().sort((p, q) => Math.abs(p.y - a.y) - Math.abs(q.y - a.y));
          const m = 2 + (r() < 0.35 ? 1 : 0);
          for (let j = 0; j < m && j < nx.length; j++) { a.out.push(nx[j]); edges.push([a, nx[j]]); }
        }
        stars = []; const ns = Math.round(W * H / 4600);
        for (let i = 0; i < ns; i++) stars.push({ x: r() * W, y: r() * H, s: 0.25 + r() * 0.75, b: 0.15 + r() * 0.5, tw: r() * 6.28 });
        pulses = []; const np = quiet ? 6 : 11;
        for (let i = 0; i < np; i++) { const p = spawn({}); p.t = Math.random(); pulses.push(p); }
      };
      const draw = (now) => {
        if (!W) return;
        const day = document.documentElement.dataset.theme === 'day';
        px += (tx - px) * 0.04; py += (ty - py) * 0.04;
        ctx.clearRect(0, 0, W, H);
        const breath = 0.5 + 0.5 * Math.sin(now / 3200);
        const gx = quiet ? W * 0.82 : W / 2, gy = quiet ? H * 0.08 : H * 0.36, R = Math.max(W, H) * (quiet ? 0.55 : 0.42);
        const g = ctx.createRadialGradient(gx, gy, 0, gx, gy, R);
        if (day) { g.addColorStop(0, 'rgba(29,88,184,' + 0.06 * k * (0.8 + 0.2 * breath) + ')'); g.addColorStop(1, 'rgba(29,88,184,0)'); }
        else { g.addColorStop(0, 'rgba(156,200,255,' + 0.1 * k * (0.78 + 0.22 * breath) + ')'); g.addColorStop(0.35, 'rgba(60,150,170,' + 0.04 * k + ')'); g.addColorStop(1, 'rgba(10,40,50,0)'); }
        ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
        const ink = day ? '11,27,43' : '214,232,255';
        for (const s of stars) {
          const a = s.b * (0.6 + 0.4 * Math.sin(now / 1900 + s.tw)) * (day ? 0.28 : 0.7) * k;
          ctx.fillStyle = 'rgba(' + ink + ',' + a + ')'; ctx.fillRect(s.x + px * 0.35, s.y + py * 0.35, s.s, s.s);
        }
        ctx.lineWidth = 0.6;
        ctx.strokeStyle = day ? 'rgba(11,27,43,' + 0.055 * k + ')' : 'rgba(156,200,255,' + 0.07 * k + ')';
        ctx.beginPath();
        for (const [a, b] of edges) { ctx.moveTo(a.x + px, a.y + py); ctx.lineTo(b.x + px, b.y + py); }
        ctx.stroke();
        for (const col of cols) for (const n of col) {
          const a = n.b * (0.7 + 0.3 * Math.sin(now / 2300 + n.tw)) * (day ? 0.4 : 0.8) * k;
          ctx.fillStyle = 'rgba(' + ink + ',' + a + ')'; ctx.beginPath(); ctx.arc(n.x + px, n.y + py, n.s, 0, 6.283); ctx.fill();
        }
        for (const p of pulses) {
          const ax = p.a.x + px, ay = p.a.y + py, bx = p.b.x + px, by = p.b.y + py;
          const x = ax + (bx - ax) * p.t, y = ay + (by - ay) * p.t;
          const t0 = Math.max(0, p.t - 0.22), x0 = ax + (bx - ax) * t0, y0 = ay + (by - ay) * t0;
          const rgb = p.hue === 'cy' ? (day ? '10,102,99' : '111,224,220') : (day ? '29,88,184' : '214,235,255');
          const lg = ctx.createLinearGradient(x0, y0, x, y); lg.addColorStop(0, 'rgba(' + rgb + ',0)'); lg.addColorStop(1, 'rgba(' + rgb + ',' + 0.5 * k + ')');
          ctx.strokeStyle = lg; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x, y); ctx.stroke();
          ctx.fillStyle = 'rgba(' + rgb + ',' + 0.8 * k + ')'; ctx.beginPath(); ctx.arc(x, y, 1.2, 0, 6.283); ctx.fill();
        }
      };
      const step = (now) => {
        const dt = Math.min(64, last ? now - last : 16); last = now; T += dt;
        for (const p of pulses) { p.t += p.v * dt; if (p.t >= 1) { if (p.b.out.length && --p.life > 0) { p.a = p.b; p.b = p.a.out[Math.floor(Math.random() * p.a.out.length)]; p.t = 0; } else spawn(p); } }
        draw(T); raf = requestAnimationFrame(step);
      };
      const running = () => visible && !document.hidden && !rm.matches;
      const sync = () => { cancelAnimationFrame(raf); raf = 0; last = 0; if (running()) raf = requestAnimationFrame(step); else draw(T || 4000); };
      this._io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; sync(); }); this._io.observe(c);
      this._ro = new ResizeObserver(() => { build(); draw(T || 4000); }); this._ro.observe(c); this._ro.observe(this);
      this._mo = new MutationObserver(() => { if (!raf) draw(T || 4000); }); this._mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      this._vis = () => sync(); document.addEventListener('visibilitychange', this._vis);
      this._move = (e) => { if (rm.matches) return; tx = (e.clientX / innerWidth - 0.5) * -10; ty = (e.clientY / innerHeight - 0.5) * -7; };
      addEventListener('pointermove', this._move);
      rm.addEventListener && rm.addEventListener('change', sync);
      let tries = 0;
      const boot = () => { if (!this._c) return; if (!c.clientWidth || !c.clientHeight) { if (++tries < 120) this._boot = requestAnimationFrame(boot); return; } build(); sync(); };
      this._stop = () => { cancelAnimationFrame(raf); cancelAnimationFrame(this._boot); };
      boot();
    }
    disconnectedCallback() {
      this._stop && this._stop(); this._io && this._io.disconnect(); this._ro && this._ro.disconnect(); this._mo && this._mo.disconnect();
      document.removeEventListener('visibilitychange', this._vis); removeEventListener('pointermove', this._move);
      this._c && this._c.remove(); this._c = null;
    }
  }
  customElements.define('vega-ground', VegaGround);
})();
