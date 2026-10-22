// <vega-orbits>: planets around Vega, moons around planets, satellites around moons. Canvas 2D, fake perspective.
// API: el.setData({ planets:[{ label, sub, moons:[{ label, ring:'acc'|'cy'|null, dash, sats:[{label}] }] }], showMoons:'always'|'focus' })
//      el.focus(pi, mi, si) ; events: 'pick' {detail:{planet, moon, sat}} (indices, -1 for none)
// Optional hooks (inert unless set; set by vega-camera / the page on the redesigned pages):
//      el.frameBox = { x, y, w, h } — screen box (canvas px) the system is centred and sized in (default: the canvas)
//      el.clip — the camera's free area { L, rects }; text near its edge slides inside or fades whole, never cut
//      el.textAlpha — share of all canvas text (the loader gate); el.textRects / el.inkPts — what it drew, for checks
(function () {
  if (customElements.get('vega-orbits')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const cl = (v, a, b) => Math.max(a, Math.min(b, v)), ease = (t) => 1 - Math.pow(1 - cl(t, 0, 1), 3), lerp = (a, b, t) => a + (b - a) * t;
  const rgbOf = (c) => { c = (c || '').trim(); if (c[0] === '#') { const n = parseInt(c.slice(1, 7), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; } const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [230, 240, 250]; };
  class VegaOrbits extends HTMLElement {
    constructor() { super(); this.d = { planets: [] }; this.f = [-1, -1, -1]; this.cam = { x: 0, y: 0, z: 1 }; this.hover = null; this.t0 = performance.now(); this.C = { fg: [233, 240, 248], mut: [163, 180, 199], dim: [140, 158, 179], acc: [156, 200, 255], cy: [111, 224, 220] }; }
    connectedCallback() {
      if (this.cv) return;
      if (getComputedStyle(this).position === 'static') this.style.position = 'relative';
      this.style.display = 'block'; if (!this.style.height) { this.style.width = '100%'; this.style.height = '100%'; }
      this.setAttribute('role', 'img'); this.setAttribute('aria-label', this.getAttribute('label') || 'Orbit figure');
      const cv = this.cv = document.createElement('canvas'); cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block'; this.appendChild(cv); this.ctx = cv.getContext('2d');
      cv.addEventListener('pointermove', (e) => { const h = this.hit(e); this.hover = h; cv.style.cursor = h ? 'pointer' : (this.f[0] >= 0 ? 'zoom-out' : 'default'); });
      cv.addEventListener('pointerleave', () => { this.hover = null; });
      cv.addEventListener('click', (e) => {
        const h = this.hit(e);
        if (!h) { if (this.f[2] >= 0) this.f = [this.f[0], this.f[1], -1]; else if (this.f[1] >= 0) this.f = [this.f[0], -1, -1]; else this.f = [-1, -1, -1]; this.focusAt = performance.now(); }
        else { this.f = h.slice(); this.focusAt = performance.now(); }
        this.dispatchEvent(new CustomEvent('pick', { detail: { planet: this.f[0], moon: this.f[1], sat: this.f[2] }, bubbles: true }));
      });
      const loop = () => { this.raf = requestAnimationFrame(loop); this._ra = performance.now(); this.frame(); }; this.raf = requestAnimationFrame(loop);
      this.fb = setInterval(() => { if (performance.now() - (this._ra || 0) > 150) this.frame(); }, 50);
      this.colT = setInterval(() => this.readColors(), 500); this.readColors();
    }
    disconnectedCallback() { cancelAnimationFrame(this.raf); clearInterval(this.fb); clearInterval(this.colT); this.cv = null; }
    readColors() { const cs = getComputedStyle(this), g = (n, f) => { const v = cs.getPropertyValue(n); return v && v.trim() ? rgbOf(v) : f; }; const C = this.C; this.C = { fg: g('--fg', C.fg), mut: g('--mut', C.mut), dim: g('--dim', C.dim), acc: g('--acc', C.acc), cy: g('--cy', C.cy) }; }
    setData(d) { this.d = Object.assign({ planets: [] }, d); this.P = null; }
    focus(pi, mi, si) { const n = [pi == null ? -1 : pi, mi == null ? -1 : mi, si == null ? -1 : si]; if (n.join() === this.f.join()) return; this.f = n; this.focusAt = performance.now(); }
    layout(t) {
      const B = this.frameBox, W = B ? B.w : this.W, H = B ? B.h : this.H, R = Math.min(W * 0.46, H * 0.95), n = this.d.planets.length, out = [];
      this.d.planets.forEach((p, i) => {
        const rx = R * (0.32 + 0.68 * (i + 1) / n), ry = rx * 0.34, sp = RM.matches ? 0 : 0.035 / (1 + i * 0.4), a = (p.phase != null ? p.phase : i * 1.9 + 0.6) + t * sp;
        const x = Math.cos(a) * rx, y = Math.sin(a) * ry, depth = (Math.sin(a) + 1) / 2;
        const mN = p.moons.length, mR = Math.max(18, Math.min(R * 0.2, 10 + mN * 2.4)), moons = p.moons.map((m, j) => { const b = j / mN * 6.283 + (RM.matches ? 0 : t * 0.12) + i; return { x: x + Math.cos(b) * mR, y: y + Math.sin(b) * mR * 0.42, b, m, mR }; });
        out.push({ x, y, rx, ry, depth, p, moons, mR });
      });
      return out;
    }
    toScreen(x, y) { const B = this.frameBox; return [(x - this.cam.x) * this.cam.z + (B ? B.x + B.w / 2 : this.W / 2), (y - this.cam.y) * this.cam.z + (B ? B.y + B.h / 2 : this.H / 2)]; }
    hit(e) {
      if (!this.L) return null; const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, f = this.f; let best = null, bd = 1e9;
      const test = (x, y, rad, val) => { const s = this.toScreen(x, y), d = Math.hypot(s[0] - mx, s[1] - my); if (d < rad && d < bd) { bd = d; best = val; } };
      this.L.forEach((pl, i) => {
        test(pl.x, pl.y, 26, [i, -1, -1]);
        if (f[0] === i) (this.ML || pl.moons).forEach((m, j) => {
          test(m.x, m.y, 16, [i, j, -1]);
          if (f[1] === j && this.SL) this.SL.forEach((s, k) => test(s.x, s.y, 14, [i, j, k]));
        });
      });
      return best;
    }
    frame() {
      const cv = this.cv; if (!cv) return; const r = this.getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight || !r.width) return;
      const dpr = Math.min(2, devicePixelRatio || 1); this.W = r.width; this.H = r.height;
      if (cv.width !== Math.round(this.W * dpr) || cv.height !== Math.round(this.H * dpr)) { cv.width = Math.round(this.W * dpr); cv.height = Math.round(this.H * dpr); }
      const ctx = this.ctx, C = this.C, W = this.W, H = this.H, now = performance.now(), t = (now - this.t0) / 1000, rm = RM.matches, f = this.f;
      const A = (c, a) => 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + cl(a, 0, 1).toFixed(3) + ')', ga = 1; // dark only (rule 4): the day theme's halved glow is gone
      const glow = (x, y, rad, c, a) => { if (a <= 0.004 || rad <= 0) return; const g = ctx.createRadialGradient(x, y, 0, x, y, rad); g.addColorStop(0, A(c, a)); g.addColorStop(0.25, A(c, a * 0.35)); g.addColorStop(1, A(c, 0)); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.283); ctx.fill(); };
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const L = this.L = this.layout(t);
      // spread focused planet's moons into a readable ring
      const fp = L[f[0]], k = rm ? 1 : ease((now - (this.focusAt || 0)) / 1100);
      let ML = null, SL = null;
      if (fp) {
        const n = fp.moons.length, RR = Math.min(this.frameBox ? this.frameBox.w : W, this.frameBox ? this.frameBox.h : H) * 0.34 / (f[1] >= 0 ? 2.6 : 1.25);
        ML = fp.moons.map((m, j) => { const b = -Math.PI / 2 + j / n * 6.283 + (rm ? 0 : t * 0.03); const tx = fp.x + Math.cos(b) * RR, ty = fp.y + Math.sin(b) * RR * 0.62; return { x: lerp(m.x, tx, k), y: lerp(m.y, ty, k), m: m.m, b }; });
        const fm = ML[f[1]];
        if (fm) { const sats = fm.m.sats || [], ns = sats.length, SR = RR * 0.62; SL = sats.map((s, q) => { const b = -Math.PI / 2 + q / Math.max(1, ns) * 6.283 + (rm ? 0 : t * 0.06); return { x: lerp(fm.x, fm.x + Math.cos(b) * SR, k), y: lerp(fm.y, fm.y + Math.sin(b) * SR * 0.7, k), s, b }; }); }
      }
      this.ML = ML; this.SL = SL;
      // camera
      let tx = 0, ty = 0, tz = 1;
      if (fp) { tx = fp.x; ty = fp.y; tz = 1.25; if (ML && ML[f[1]]) { tx = ML[f[1]].x; ty = ML[f[1]].y; tz = 2.6; } }
      const cs = rm ? 1 : 0.07; this.cam.x += (tx - this.cam.x) * cs; this.cam.y += (ty - this.cam.y) * cs; this.cam.z += (tz - this.cam.z) * cs;
      const S = (x, y) => this.toScreen(x, y), z = this.cam.z;
      // canvas text through one door: kept inside the free area (slid, or faded whole) and recorded
      const CL = this.clip, TA = this.textAlpha == null ? 1 : this.textAlpha, TR = this.textRects = [], INK = this.inkPts = [], OC = [];
      const text = (str, x, y, align) => {
        const w = ctx.measureText(str).width; let l = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
        if (CL) { const lo = CL.L + 58; if (l < lo) l = lo; for (const r of CL.rects) if (r[0] > -1e8 && y > r[1] - 20 && y - 12 < r[3] + 20 && l + w > r[0] - 58 && l < r[0]) l = Math.max(lo, r[0] - 58 - w); }
        { const ed = this.W <= 600 ? 24 : 40; if (l + w > this.W - ed) l = this.W - ed - w; if (l < ed) l = ed; } // (G7) a label near an edge slides inside the HUD frame, whole
        let f = TA; if (CL) { f = Math.min(f, cl((l - CL.L - 28) / 28, 0, 1)); for (const r of CL.rects) { const dx = Math.max(r[0] - (l + w), l - r[2], 0), dy = Math.max(r[1] - (y + 3), (y - 11) - r[3], 0); f = Math.min(f, Math.max(cl((dx - 28) / 28, 0, 1), cl((dy - 20) / 20, 0, 1))); } }
        if (f <= 0.01) return;
        // a label that would land on one already drawn moves up or down a line (none is dropped)
        const over = (yy) => OC.some(o => l < o[2] + 2 && l + w > o[0] - 2 && yy - 11 < o[3] && yy + 3 > o[1]);
        if (over(y)) { const alt = [-14, 14, -28, 28].map(d => y + d).find(yy => !over(yy) && yy > 14 && yy < this.H - 8); if (alt != null) y = alt; }
        const ga0 = ctx.globalAlpha; ctx.globalAlpha = ga0 * f; ctx.textAlign = 'left'; ctx.fillText(str, l, y); ctx.globalAlpha = ga0; ctx.textAlign = align;
        OC.push([l, y - 11, l + w, y + 3]); if (f > 0.2) TR.push([l, y - 11, l + w, y + 3]);
      };
      // nebula
      const o = S(0, 0); glow(o[0], o[1], Math.max(W, H) * 0.5 * z, C.acc, 0.07 * ga);
      // orbits (planets)
      L.forEach((pl, i) => { const on = f[0] === i; ctx.strokeStyle = A(on ? C.acc : C.acc, (on ? 0.45 : f[0] >= 0 ? 0.08 : 0.2)); ctx.lineWidth = 1; ctx.beginPath(); ctx.ellipse(o[0], o[1], pl.rx * z, pl.ry * z, 0, 0, 6.283); ctx.stroke(); });
      // Vega
      glow(o[0], o[1], 46 * Math.sqrt(z), C.acc, 0.8 * ga); ctx.fillStyle = A(C.fg, 1); ctx.beginPath(); ctx.arc(o[0], o[1], 4.5, 0, 6.283); ctx.fill();
      const Ls = 22 * Math.sqrt(z); [[1, 0], [0, 1]].forEach(([dx, dy]) => { const g = ctx.createLinearGradient(o[0] - dx * Ls, o[1] - dy * Ls, o[0] + dx * Ls, o[1] + dy * Ls); g.addColorStop(0, A(C.fg, 0)); g.addColorStop(0.5, A(C.fg, 0.9)); g.addColorStop(1, A(C.fg, 0)); ctx.strokeStyle = g; ctx.beginPath(); ctx.moveTo(o[0] - dx * Ls, o[1] - dy * Ls); ctx.lineTo(o[0] + dx * Ls, o[1] + dy * Ls); ctx.stroke(); });
      if (this.d.center) { ctx.font = '500 10px ui-monospace,Menlo,Consolas,monospace'; ctx.textAlign = 'center'; ctx.fillStyle = A(C.dim, 1); text(this.d.center, o[0], o[1] + 34, 'center'); }
      // planets back to front
      const order = L.map((p, i) => i).sort((a, b) => L[a].depth - L[b].depth);
      const mono = 'ui-monospace,Menlo,Consolas,monospace';
      order.forEach(i => {
        const pl = L[i], on = f[0] === i, other = f[0] >= 0 && !on, hideLbl = other && z > 1.2, s = S(pl.x, pl.y), hov = this.hover && this.hover[0] === i && this.hover[1] < 0, sc = 0.75 + pl.depth * 0.5;
        // moons (compact)
        if (!on) pl.moons.forEach(m => { const q = S(m.x, m.y); ctx.fillStyle = A(m.m.ring === 'cy' ? C.cy : C.fg, other ? 0.25 : 0.55); ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(0.8, (this.d.moonSize || 1.4) * Math.sqrt(z)), 0, 6.283); ctx.fill(); });
        glow(s[0], s[1], (on ? 34 : 22) * sc * Math.sqrt(z), C.acc, (other ? 0.25 : on || hov ? 0.85 : 0.55) * ga);
        ctx.fillStyle = A(C.fg, other ? 0.5 : 1); ctx.beginPath(); ctx.arc(s[0], s[1], (on ? 6 : 4.4) * sc, 0, 6.283); ctx.fill(); INK.push([s[0], s[1], 6]);
        if (hov || on) { ctx.strokeStyle = A(C.acc, 0.7); ctx.setLineDash([3, 4]); ctx.lineDashOffset = rm ? 0 : -t * 8; ctx.beginPath(); ctx.arc(s[0], s[1], 14 * sc, 0, 6.283); ctx.stroke(); ctx.setLineDash([]); }
        ctx.textAlign = 'center'; ctx.font = (on ? '500 ' : '400 ') + (on ? 14 : 12.5) + 'px "Helvetica Neue",Helvetica,Arial,sans-serif'; ctx.fillStyle = A(on || hov ? C.fg : C.mut, other ? 0.45 : 1);
        const ly = s[1] - (on ? 26 : 18) * sc; if (hideLbl) return; text(pl.p.label, cl(s[0], 60, W - 60), cl(ly, 16, H - 48), 'center');
        if (pl.p.sub) { ctx.font = '500 10px ' + mono; ctx.fillStyle = A(C.dim, other ? 0.45 : 1); text(pl.p.sub, cl(s[0], 60, W - 60), cl(ly + 14, 30, H - 34), 'center'); }
      });
      // focused system
      if (fp && ML) {
        const ps = S(fp.x, fp.y);
        ctx.strokeStyle = A(C.acc, 0.22 * k); ctx.lineWidth = 1; const r0 = Math.hypot(...[0].map(() => 0)); 
        ML.forEach((m, j) => {
          const q = S(m.x, m.y), mon = f[1] === j, mo = f[1] >= 0 && !mon, hv = this.hover && this.hover[0] === f[0] && this.hover[1] === j && this.hover[2] < 0;
          ctx.strokeStyle = A(C.acc, (mon ? 0.6 : 0.16) * k); ctx.beginPath(); ctx.moveTo(ps[0], ps[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
          glow(q[0], q[1], (mon ? 22 : 12) * Math.sqrt(z / 1.25), m.m.ring === 'cy' ? C.cy : C.acc, (mo ? 0.12 : 0.7) * ga);
          ctx.fillStyle = A(C.fg, mo ? 0.3 : 1); ctx.beginPath(); ctx.arc(q[0], q[1], mon ? 4 : 2.6, 0, 6.283); ctx.fill();
          if (m.m.ring) { ctx.strokeStyle = A(m.m.ring === 'cy' ? C.cy : C.acc, mo ? 0.35 : 0.9); ctx.setLineDash(m.m.dash ? [2, 2] : []); ctx.beginPath(); ctx.arc(q[0], q[1], mon ? 9 : 7, 0, 6.283); ctx.stroke(); ctx.setLineDash([]); }
          if (hv) { ctx.strokeStyle = A(C.acc, 0.8); ctx.beginPath(); ctx.arc(q[0], q[1], 12, 0, 6.283); ctx.stroke(); }
          if (m.m.label && k > 0.3 && (this.d.moonLabels !== false)) {
            const out = Math.cos(m.b) >= 0; ctx.textAlign = out ? 'left' : 'right'; ctx.font = (mon ? '500 13px ' : '12px ') + mono; ctx.fillStyle = A(mon || hv ? C.fg : C.mut, (mo ? 0.22 : 1) * k);
            text(m.m.label, cl(q[0] + (out ? 12 : -12), 8, W - 8), cl(q[1] + 4, 14, H - 36), out ? 'left' : 'right');
          }
        });
        if (SL) SL.forEach((s, q2) => {
          const p2 = S(s.x, s.y), fm = S(ML[f[1]].x, ML[f[1]].y), son = f[2] === q2, hv = this.hover && this.hover[2] === q2;
          ctx.strokeStyle = A(C.cy, (son ? 0.8 : 0.3) * k); ctx.beginPath(); ctx.moveTo(fm[0], fm[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
          glow(p2[0], p2[1], son ? 18 : 10, s.s.ring === 'cy' ? C.cy : C.acc, 0.8 * ga); ctx.fillStyle = A(C.fg, 1); ctx.beginPath(); ctx.arc(p2[0], p2[1], son ? 3.6 : 2.4, 0, 6.283); ctx.fill();
          if (s.s.ring) { ctx.strokeStyle = A(s.s.ring === 'cy' ? C.cy : C.acc, 0.9); ctx.setLineDash(s.s.dash ? [2, 2] : []); ctx.beginPath(); ctx.arc(p2[0], p2[1], son ? 9 : 7, 0, 6.283); ctx.stroke(); ctx.setLineDash([]); }
          if (son || hv) { ctx.strokeStyle = A(C.acc, 0.8); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(p2[0], p2[1], 13, 0, 6.283); ctx.stroke(); ctx.setLineDash([]); }
          const out = Math.cos(s.b) >= 0; ctx.textAlign = out ? 'left' : 'right'; ctx.font = (son ? '500 12.5px ' : '12px ') + mono; ctx.fillStyle = A(son || hv ? C.fg : C.mut, k);
          text(s.s.label, cl(p2[0] + (out ? 12 : -12), 8, W - 8), cl(p2[1] + 4, 14, H - 36), out ? 'left' : 'right');
        });
      }
    }
  }
  customElements.define('vega-orbits', VegaOrbits);
})();
