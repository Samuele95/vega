// <vega-sky>: a photographic-feeling night sky. Static layer (Milky Way band with dust lanes, colour-graded field stars,
// bright stars with diffraction spikes) is pre-rendered once per size/theme; only scintillation and parallax animate.
// Attributes: seed, band ("on"|"off").
(function () {
  if (customElements.get('vega-sky')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const TEMP = [[155, 176, 255, 0.06], [170, 191, 255, 0.10], [202, 215, 255, 0.16], [248, 247, 255, 0.24], [255, 244, 234, 0.18], [255, 226, 196, 0.14], [255, 204, 160, 0.08], [255, 180, 130, 0.04]];
  class VegaSky extends HTMLElement {
    connectedCallback() {
      if (this.cv) return;
      Object.assign(this.style, { display: 'block', pointerEvents: 'none', position: getComputedStyle(this).position === 'static' ? 'absolute' : this.style.position, inset: '0', width: '100%', height: '100%' });
      const cv = this.cv = document.createElement('canvas'); cv.setAttribute('aria-hidden', 'true'); cv.style.cssText = 'position:absolute;inset:-14px;width:calc(100% + 28px);height:calc(100% + 28px);display:block'; this.appendChild(cv);
      this.mx = 0; this.my = 0; this.tx = 0; this.ty = 0;
      this.onMove = (e) => { if (e.pointerType === 'touch') return; this.tx = e.clientX / innerWidth - 0.5; this.ty = e.clientY / innerHeight - 0.5; };
      addEventListener('pointermove', this.onMove, { passive: true });
      const loop = () => { this.raf = requestAnimationFrame(loop); this._ra = performance.now(); this.frame(); }; this.raf = requestAnimationFrame(loop);
      this.fb = setInterval(() => { if (performance.now() - (this._ra || 0) > 150) this.frame(); }, 60);
    }
    disconnectedCallback() { cancelAnimationFrame(this.raf); clearInterval(this.fb); removeEventListener('pointermove', this.onMove); this.cv = null; }
    rng() { let s = +(this.getAttribute('seed') || 17) * 7919 + 1; return () => { s = (s * 16807) % 2147483647; return s / 2147483647; }; }
    build(W, H, dpr, day) {
      const mode = this.mode = day ? ((this.getAttribute('day-style') || this.getAttribute('daystyle')) || 'plate') : 'night';
      if (mode === 'plate' || mode === 'atlas' || mode === 'twilight') return this.buildDay(W, H, dpr, mode);
      const r = this.rng(), g = () => { let u = 0; for (let k = 0; k < 6; k++) u += r(); return (u - 3) / 1.5; };
      const pick = () => { let x = r(), acc = 0; for (const t of TEMP) { acc += t[3]; if (x <= acc) return t; } return TEMP[3]; };
      const off = document.createElement('canvas'); off.width = Math.round(W * dpr); off.height = Math.round(H * dpr);
      const c = off.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const k = day ? 0.32 : 1, A = (col, a) => 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + Math.max(0, Math.min(1, a * k)).toFixed(3) + ')';
      const blob = (x, y, rad, col, a) => { const gr = c.createRadialGradient(x, y, 0, x, y, rad); gr.addColorStop(0, A(col, a)); gr.addColorStop(1, A(col, 0)); c.fillStyle = gr; c.beginPath(); c.arc(x, y, rad, 0, 6.283); c.fill(); };
      const D = Math.hypot(W, H), x0 = -0.08 * W, y0 = 0.92 * H, x1 = 1.08 * W, y1 = 0.02 * H, ux = (x1 - x0) / Math.hypot(x1 - x0, y1 - y0), uy = (y1 - y0) / Math.hypot(x1 - x0, y1 - y0), nx = -uy, ny = ux, BW = D * 0.11;
      const band = this.getAttribute('band') !== 'off';
      if (band) {
        // diffuse glow, warm core / cool edges, uneven along its length
        for (let i = 0; i < 260; i++) { const t = r(), o = g() * BW * 0.9, len = Math.hypot(x1 - x0, y1 - y0), x = x0 + ux * len * t + nx * o, y = y0 + uy * len * t + ny * o, warm = Math.abs(o) < BW * 0.35 && r() < 0.6;
          blob(x, y, BW * (0.25 + r() * 0.55), warm ? [255, 232, 205] : [175, 200, 255], (0.018 + r() * 0.03) * (0.6 + 0.4 * Math.sin(t * 9 + 1.3))); }
        // unresolved star clouds
        for (let i = 0; i < Math.round(W * H / 55); i++) { const t = r(), o = g() * BW * 0.75, len = Math.hypot(x1 - x0, y1 - y0), x = x0 + ux * len * t + nx * o, y = y0 + uy * len * t + ny * o, col = pick(); c.fillStyle = A(col, 0.12 + r() * 0.35); const z = 0.35 + r() * 0.5; c.fillRect(x, y, z, z); }
        // dust lanes
        c.globalCompositeOperation = 'destination-out';
        for (let i = 0; i < 140; i++) { const t = r(), len = Math.hypot(x1 - x0, y1 - y0), wob = Math.sin(t * 14) * BW * 0.18 + Math.sin(t * 37) * BW * 0.06, o = wob + g() * BW * 0.12, x = x0 + ux * len * t + nx * o, y = y0 + uy * len * t + ny * o; blob(x, y, BW * (0.06 + r() * 0.16), [0, 0, 0], (0.25 + r() * 0.35) / k); }
        c.globalCompositeOperation = 'source-over';
      }
      // faint coloured emission patches
      blob(W * 0.18, H * 0.3, D * 0.22, [70, 140, 200], 0.035); blob(W * 0.82, H * 0.62, D * 0.2, [60, 170, 165], 0.03); blob(W * 0.62, H * 0.18, D * 0.12, [190, 120, 160], 0.012);
      // field stars — power-law brightness
      const N = Math.round(W * H / 420);
      for (let i = 0; i < N; i++) { const x = r() * W, y = r() * H, m = Math.pow(r(), 4.2), col = day ? [60, 80, 105] : pick(), z = 0.35 + m * 1.4; c.fillStyle = A(col, 0.18 + m * 0.75); c.beginPath(); c.arc(x, y, z * 0.62, 0, 6.283); c.fill(); if (m > 0.55) blob(x, y, 3 + m * 5, col, 0.12 * m); }
      // twinklers + bright stars
      this.tw = []; this.bright = [];
      for (let i = 0; i < Math.round(W * H / 9000); i++) this.tw.push({ x: r() * W, y: r() * H, m: 0.35 + r() * 0.5, col: pick(), f: 1.2 + r() * 3.5, p: r() * 6.283 });
      for (let i = 0; i < Math.max(6, Math.round(W * H / 90000)); i++) this.bright.push({ x: r() * W, y: r() * H, m: 0.6 + r() * 0.4, col: pick(), spikes: i < 5, f: 0.8 + r() * 1.4, p: r() * 6.283 });
      this.off = off; this.k = k;
    }
    buildDay(W, H, dpr, mode) {
      const r = this.rng(), g = () => { let u = 0; for (let k = 0; k < 6; k++) u += r(); return (u - 3) / 1.5; };
      const off = document.createElement('canvas'); off.width = Math.round(W * dpr); off.height = Math.round(H * dpr);
      const c = off.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const A = (col, a) => 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + Math.max(0, Math.min(1, a)).toFixed(3) + ')';
      const blob = (cx, x, y, rad, col, a) => { const gr = cx.createRadialGradient(x, y, 0, x, y, rad); gr.addColorStop(0, A(col, a)); gr.addColorStop(1, A(col, 0)); cx.fillStyle = gr; cx.beginPath(); cx.arc(x, y, rad, 0, 6.283); cx.fill(); };
      const D = Math.hypot(W, H), x0 = -0.08 * W, y0 = 0.92 * H, x1 = 1.08 * W, y1 = 0.02 * H, len = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / len, uy = (y1 - y0) / len, nx = -uy, ny = ux, BW = D * 0.11;
      const along = (t, o) => [x0 + ux * len * t + nx * o, y0 + uy * len * t + ny * o];
      let ink, base;
      if (mode === 'plate') {
        ink = [22, 26, 34];
        const gr = c.createRadialGradient(W / 2, H * 0.45, 0, W / 2, H * 0.45, D * 0.7); gr.addColorStop(0, '#F3EFE6'); gr.addColorStop(0.7, '#ECE6DA'); gr.addColorStop(1, '#DFD7C8'); c.fillStyle = gr; c.fillRect(0, 0, W, H);
        // emulsion grain
        for (let i = 0; i < W * H / 40; i++) { c.fillStyle = A([60, 55, 45], 0.025 + r() * 0.03); c.fillRect(r() * W, r() * H, 1, 1); }
        // band on its own layer so dust lanes can lighten it
        const L = document.createElement('canvas'); L.width = off.width; L.height = off.height; const l = L.getContext('2d'); l.setTransform(dpr, 0, 0, dpr, 0, 0);
        for (let i = 0; i < 240; i++) { const t = r(), o = g() * BW * 0.9, p = along(t, o); blob(l, p[0], p[1], BW * (0.25 + r() * 0.55), [40, 44, 54], (0.02 + r() * 0.03) * (0.6 + 0.4 * Math.sin(t * 9 + 1.3))); }
        for (let i = 0; i < Math.round(W * H / 60); i++) { const t = r(), o = g() * BW * 0.75, p = along(t, o); l.fillStyle = A(ink, 0.1 + r() * 0.3); const z = 0.4 + r() * 0.5; l.fillRect(p[0], p[1], z, z); }
        l.globalCompositeOperation = 'destination-out';
        for (let i = 0; i < 140; i++) { const t = r(), wob = Math.sin(t * 14) * BW * 0.18 + Math.sin(t * 37) * BW * 0.06, p = along(t, wob + g() * BW * 0.12); blob(l, p[0], p[1], BW * (0.06 + r() * 0.16), [0, 0, 0], 0.25 + r() * 0.35); }
        c.drawImage(L, 0, 0, W, H);
        // plate edge + reference marks
        c.strokeStyle = A(ink, 0.12); c.lineWidth = 1; c.strokeRect(14.5, 14.5, W - 29, H - 29);
        for (let i = 1; i < 10; i++) { c.beginPath(); c.moveTo(14 + (W - 28) * i / 10, 14); c.lineTo(14 + (W - 28) * i / 10, 22); c.moveTo(14 + (W - 28) * i / 10, H - 14); c.lineTo(14 + (W - 28) * i / 10, H - 22); c.stroke(); }
        for (let i = 1; i < 8; i++) { c.beginPath(); c.moveTo(14, 14 + (H - 28) * i / 8); c.lineTo(22, 14 + (H - 28) * i / 8); c.moveTo(W - 14, 14 + (H - 28) * i / 8); c.lineTo(W - 22, 14 + (H - 28) * i / 8); c.stroke(); }
      } else if (mode === 'twilight') {
        ink = [255, 255, 255];
        const gr = c.createLinearGradient(0, 0, 0, H); gr.addColorStop(0, '#8FA8CC'); gr.addColorStop(0.45, '#BCCBDE'); gr.addColorStop(0.8, '#E8DFD6'); gr.addColorStop(1, '#F2E2D0'); c.fillStyle = gr; c.fillRect(0, 0, W, H);
        blob(c, W * 0.5, H * 1.05, D * 0.5, [255, 214, 180], 0.35);
        for (let i = 0; i < 120; i++) { const t = r(), o = g() * BW * 0.9, p = along(t, o); if (p[1] > H * 0.55) continue; blob(c, p[0], p[1], BW * (0.25 + r() * 0.5), [240, 246, 255], 0.03 * (1 - p[1] / (H * 0.55))); }
      } else {
        ink = [11, 27, 43];
        c.fillStyle = '#F4F6F9'; c.fillRect(0, 0, W, H);
        for (let i = 0; i < 80; i++) { const t = r(), o = g() * BW * 0.8, p = along(t, o); blob(c, p[0], p[1], BW * (0.3 + r() * 0.5), [29, 88, 184], 0.012); }
        c.strokeStyle = A([29, 88, 184], 0.08); c.lineWidth = 1;
        const cxp = W * 0.5, cyp = H * 1.6;
        for (let i = 1; i <= 9; i++) { c.beginPath(); c.arc(cxp, cyp, H * 0.55 + i * H * 0.16, Math.PI * 1.05, Math.PI * 1.95); c.stroke(); }
        for (let i = -8; i <= 8; i++) { const a = -Math.PI / 2 + i * 0.09; c.beginPath(); c.moveTo(cxp + Math.cos(a) * H * 0.55, cyp + Math.sin(a) * H * 0.55); c.lineTo(cxp + Math.cos(a) * H * 2.2, cyp + Math.sin(a) * H * 2.2); c.stroke(); }
        c.setLineDash([2, 4]); c.strokeStyle = A([29, 88, 184], 0.2); [-1, 1].forEach(sg => { c.beginPath(); for (let t = 0; t <= 1.001; t += 0.02) { const p = along(t, sg * BW * (0.75 + 0.15 * Math.sin(t * 11))); t ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]); } c.stroke(); }); c.setLineDash([]);
      }
      const N = mode === 'twilight' ? 0 : Math.round(W * H / (mode === 'atlas' ? 3200 : 520)), pts = [];
      for (let i = 0; i < N; i++) { const x = r() * W, y = r() * H, m = Math.pow(r(), mode === 'atlas' ? 3 : 4.2);
        if (mode === 'atlas') { const rad = 0.5 + m * 2.6; c.fillStyle = A(ink, 0.32 + m * 0.4); c.beginPath(); c.arc(x, y, rad, 0, 6.283); c.fill(); if (m > 0.5) { c.strokeStyle = A([244, 246, 249], 1); c.lineWidth = 1; c.beginPath(); c.arc(x, y, rad + 0.8, 0, 6.283); c.stroke(); pts.push([x, y]); } }
        else { c.fillStyle = A(ink, 0.2 + m * 0.7); c.beginPath(); c.arc(x, y, (0.4 + m * 1.5) * 0.65, 0, 6.283); c.fill(); if (m > 0.6) blob(c, x, y, 2.5 + m * 4, ink, 0.1 * m); } }
      if (mode === 'atlas') { c.strokeStyle = A([29, 88, 184], 0.35); c.lineWidth = 0.8; c.strokeStyle = A([29, 88, 184], 0.22); pts.forEach((a, i) => { let bj = -1, bd = 1e9; pts.forEach((b, j) => { if (j === i) return; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d < bd) { bd = d; bj = j; } }); if (bj > i || (bj >= 0 && bd < Math.min(W, H) * 0.14)) { if (bd < Math.min(W, H) * 0.18) { const b = pts[bj]; c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); } } }); }
      this.tw = []; this.bright = [];
      if (mode !== 'atlas') for (let i = 0; i < Math.round(W * H / (mode === 'twilight' ? 40000 : 12000)); i++) { const y = mode === 'twilight' ? Math.pow(r(), 1.6) * H * 0.55 : r() * H; this.tw.push({ x: r() * W, y, m: 0.35 + r() * 0.5, col: ink, f: 1.2 + r() * 3.5, p: r() * 6.283 }); }
      for (let i = 0; i < Math.max(5, Math.round(W * H / 110000)); i++) { const y = mode === 'twilight' ? r() * H * 0.45 : r() * H; this.bright.push({ x: r() * W, y, m: 0.6 + r() * 0.4, col: ink, spikes: i < 5, f: 0.8 + r() * 1.4, p: r() * 6.283 }); }
      if (mode === 'atlas') this.bright = [];
      this.off = off; this.k = mode === 'twilight' ? 0.85 : 1;
    }
    frame() {
      const cv = this.cv; if (!cv || document.hidden) return;
      const rect = this.getBoundingClientRect(); if (!rect.width || rect.bottom < 0 || rect.top > innerHeight) return;
      const W = rect.width + 28, H = rect.height + 28, dpr = Math.min(2, devicePixelRatio || 1), day = document.documentElement.dataset.theme === 'day', key = Math.round(W) + 'x' + Math.round(H) + '@' + dpr + (day ? 'd' + ((this.getAttribute('day-style') || this.getAttribute('daystyle')) || 'plate') : 'n');
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      if (key !== this._key) { this._key = key; this.build(W, H, dpr, day); }
      const ctx = cv.getContext('2d'), rm = RM.matches, t = performance.now() / 1000, k = this.k;
      this.mx += (this.tx - this.mx) * 0.04; this.my += (this.ty - this.my) * 0.04;
      const ox = rm ? 0 : -this.mx * 12, oy = rm ? 0 : -this.my * 8;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(this.off, ox * dpr, oy * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const A = (col, a) => 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + Math.max(0, Math.min(1, a * k)).toFixed(3) + ')';
      const px = ox * 1.6, py = oy * 1.6;
      for (const s of this.tw) { const sc = rm ? 0.8 : 0.55 + 0.45 * Math.pow(0.5 + 0.5 * Math.sin(t * s.f + s.p) * Math.sin(t * s.f * 0.37 + s.p * 2), 1.4); ctx.fillStyle = A(day && this.mode === 'night' ? [60, 80, 105] : s.col, s.m * sc * (this.mode === 'twilight' ? Math.max(0, 1 - s.y / (H * 0.55)) : 1)); ctx.beginPath(); ctx.arc(s.x + px, s.y + py, 0.55 + s.m * 0.6, 0, 6.283); ctx.fill(); }
      for (const s of this.bright) {
        const x = s.x + px * 1.3, y = s.y + py * 1.3, sc = rm ? 1 : 0.82 + 0.18 * Math.sin(t * s.f + s.p), col = day && this.mode === 'night' ? [40, 60, 90] : s.col, tf = this.mode === 'twilight' ? Math.max(0.15, 1 - s.y / (H * 0.5)) : 1;
        const gr = ctx.createRadialGradient(x, y, 0, x, y, 9 * s.m); gr.addColorStop(0, A(col, 0.55 * sc * tf)); gr.addColorStop(0.3, A(col, 0.16 * sc * tf)); gr.addColorStop(1, A(col, 0)); ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(x, y, 9 * s.m, 0, 6.283); ctx.fill();
        ctx.fillStyle = A(this.mode === 'plate' ? [10, 12, 18] : [255, 255, 255], 0.95 * sc * tf); ctx.beginPath(); ctx.arc(x, y, 0.9 + s.m * 0.7, 0, 6.283); ctx.fill();
        if (s.spikes) { const L = 9 + s.m * 10; ctx.lineWidth = 0.7; [[1, 0], [0, 1]].forEach(([dx, dy]) => { const lg = ctx.createLinearGradient(x - dx * L, y - dy * L, x + dx * L, y + dy * L); lg.addColorStop(0, A(col, 0)); lg.addColorStop(0.5, A(col, 0.7 * sc * tf)); lg.addColorStop(1, A(col, 0)); ctx.strokeStyle = lg; ctx.beginPath(); ctx.moveTo(x - dx * L, y - dy * L); ctx.lineTo(x + dx * L, y + dy * L); ctx.stroke(); }); }
      }
    }
  }
  customElements.define('vega-sky', VegaSky);
})();
