// <vega-constellation>: a small hand-written 3D star-field renderer (canvas 2D, perspective projection).
// API: el.setData({nodes, edges, guides, labels, radius, pulse, autoRotate, yaw, pitch}); el.setSelected(i, fly); event 'pick' {detail:{index}}.
// d.nebula: optional nebula intensity (default 1; 0 skips it). el.ox / el.oy: optional screen offset in px (default 0) — vega-camera uses it to put its focus point where a stop frames it.
(function () {
  if (customElements.get('vega-constellation')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
  const wrap = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
  const rgbOf = (c) => {
    c = (c || '').trim();
    if (c[0] === '#') { const n = parseInt(c.length === 4 ? c.slice(1).split('').map(x => x + x).join('') : c.slice(1, 7), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [230, 240, 250];
  };
  const spriteCache = {};
  const sprite = (rgb) => {
    const k = rgb.join(','); if (spriteCache[k]) return spriteCache[k];
    const c = document.createElement('canvas'); c.width = c.height = 96; const g = c.getContext('2d');
    const gr = g.createRadialGradient(48, 48, 0, 48, 48, 48);
    gr.addColorStop(0, 'rgba(' + k + ',1)'); gr.addColorStop(0.12, 'rgba(' + k + ',.6)'); gr.addColorStop(0.35, 'rgba(' + k + ',.16)'); gr.addColorStop(1, 'rgba(' + k + ',0)');
    g.fillStyle = gr; g.fillRect(0, 0, 96, 96); return (spriteCache[k] = c);
  };
  let rs = 7; const rnd = () => { rs = (rs * 16807) % 2147483647; return rs / 2147483647; };

  class VegaConstellation extends HTMLElement {
    constructor() {
      super();
      this.d = { nodes: [], edges: [], guides: [], labels: [], radius: 300 };
      this.sel = -1; this.hover = -1; this._sh = 0; this.yaw = 0.6; this.pitch = -0.3; this.vy = 0;
      this.tYaw = null; this.tPitch = null; this.idle = 0; this.t0 = 0; this.started = false; this.visible = true;
      this.col = { fg: [233, 240, 248], mut: [163, 180, 199], acc: [156, 200, 255], cy: [111, 224, 220], dim: [140, 158, 179] };
      this.dust = []; for (let i = 0; i < 320; i++) { const u = rnd() * 2 - 1, th = rnd() * 6.283, r = Math.cbrt(rnd()) * 1.9 + 0.35; const s = Math.sqrt(1 - u * u); this.dust.push([s * Math.cos(th) * r, u * r, s * Math.sin(th) * r, rnd(), rnd() * 6.283]); }
    }
    connectedCallback() {
      if (this.cv) return;
      if (getComputedStyle(this).position === 'static') this.style.position = 'relative';
      this.style.display = 'block'; if (!this.style.height) { this.style.width = '100%'; this.style.height = '100%'; this.style.minHeight = '320px'; } this.style.touchAction = 'pan-y'; this.style.userSelect = 'none';
      if (!this.getAttribute('role')) { this.setAttribute('role', 'img'); this.setAttribute('aria-label', this.getAttribute('label') || 'Interactive 3D figure'); }
      const cv = this.cv = document.createElement('canvas');
      cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab';
      this.appendChild(cv); this.ctx = cv.getContext('2d');
      this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this); this.resize();
      this.io = null;
      let down = null, lastX = 0, lastY = 0, moved = 0;
      cv.addEventListener('pointerdown', (e) => { if (this._tour) this.dispatchEvent(new CustomEvent('tourstop', { bubbles: true })); down = e.pointerId; lastX = e.clientX; lastY = e.clientY; moved = 0; this.vy = 0; this.tYaw = null; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
      cv.addEventListener('pointermove', (e) => {
        if (down === e.pointerId) {
          const dx = e.clientX - lastX, dy = e.clientY - lastY; lastX = e.clientX; lastY = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
          this.yaw += dx * 0.0065; this.pitch = clamp(this.pitch - dy * 0.005, -1.25, 1.25); this.vy = dx * 0.0065; this.idle = performance.now() + 3500;
        } else { const h = this.pick(e); if (h !== this.hover) { this.hover = h; cv.style.cursor = h >= 0 ? 'pointer' : 'grab'; } }
      });
      const up = (e) => {
        if (down !== e.pointerId) return; down = null; cv.style.cursor = 'grab';
        if (moved < 6) { const h = this.pick(e); if (h >= 0) this.dispatchEvent(new CustomEvent('pick', { detail: { index: h }, bubbles: true })); }
      };
      cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
      cv.addEventListener('pointerleave', () => { if (down === null) this.hover = -1; });
      cv.addEventListener('dblclick', () => { this.tYaw = this.d.yaw != null ? this.d.yaw : 0.6; this.tPitch = this.d.pitch != null ? this.d.pitch : -0.3; });
      this.loop = this.loop.bind(this); this.raf = requestAnimationFrame(this.loop);
      this.fb = setInterval(() => { const n = performance.now(); if (n - (this._rafAt || 0) > 120) this.frame(n); }, 40);
      this.readColors(); this.colT = setInterval(() => this.readColors(), 500);
    }
    disconnectedCallback() { cancelAnimationFrame(this.raf); clearInterval(this.fb); clearInterval(this.colT); this.ro && this.ro.disconnect(); this.io && this.io.disconnect(); this.cv = null; }
    readColors() {
      const cs = getComputedStyle(this);
      const g = (n, f) => { const v = cs.getPropertyValue(n); return v && v.trim() ? rgbOf(v) : f; };
      this.col = { fg: g('--fg', this.col.fg), mut: g('--mut', this.col.mut), acc: g('--acc', this.col.acc), cy: g('--cy', this.col.cy), dim: g('--dim', this.col.dim) };
    }
    resize() {
      const r = this.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
      this.W = Math.max(1, r.width); this.H = Math.max(1, r.height);
      this.cv.width = Math.round(this.W * dpr); this.cv.height = Math.round(this.H * dpr); this.dpr = dpr;
    }
    setData(d) {
      const first = !this.d.nodes.length;
      if (!first && d.morph && !RM.matches && this.d.nodes.length === (d.nodes || []).length) { // reduced motion: a new state cuts in
        const now = performance.now(), old = this.d.nodes, k = this._mk != null ? this._mk(now) : 1;
        d.nodes.forEach((n, i) => { const o = old[i], f = o._f && k < 1 ? o : null; n._f = { x: f ? o._f.x + (o.x - o._f.x) * k : o.x, y: f ? o._f.y + (o.y - o._f.y) * k : o.y, z: f ? o._f.z + (o.z - o._f.z) * k : o.z, a: o.alpha != null ? o.alpha : 1, s: o.size || 3 }; n.tw = o.tw; n.dl = o.dl; });
        const t0 = now, dur = d.morph; this._mk = (t) => ease((t - t0) / dur);
      } else if (!first) this._mk = null;
      this.d = Object.assign({ nodes: [], edges: [], guides: [], labels: [], radius: 300 }, d);
      if (first) { if (d.yaw != null) this.yaw = d.yaw; if (d.pitch != null) this.pitch = d.pitch; }
      this.d.nodes.forEach((n, i) => { if (n.tw == null) { n.tw = rnd() * 6.283; n.dl = n.delay != null ? n.delay : i * 28; } });
    }
    tour(list, ms, onStep) {
      clearInterval(this._tour); if (!list) { this._tour = null; return; }
      let k = 0; const go = () => { if (k >= list.length) { clearInterval(this._tour); this._tour = null; onStep && onStep(-1); return; } onStep && onStep(list[k]); k++; };
      go(); this._tour = setInterval(go, ms || 2600);
    }
    setSelected(i, fly) {
      if (i === this.sel) return; this.sel = i;
      const n = this.d.nodes[i]; if (!n || fly === false) return;
      if (Math.hypot(n.x, n.z) > this.d.radius * 0.15) { this.tYaw = this.yaw + wrap(Math.atan2(n.x, n.z) + Math.PI - this.yaw); }
      this.tPitch = clamp(-0.22 - (n.y / this.d.radius) * 0.35, -0.9, 0.5); this.idle = performance.now() + 4500;
    }
    proj(x, y, z) {
      const cy = this._cy, sy = this._sy, cp = this._cp, sp = this._sp;
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const s = this.F / (z2 + this.dist);
      this._vx = x1; this._vy = y1; // view space (y up), for a renderer that lights bodies (vega-galaxy's planets)
      this._sh += ((this.d.shift || 0) - this._sh) * 0.06; return [this.W * (0.5 + this._sh) + x1 * s + (this.ox || 0), this.H / 2 - y1 * s + (this.oy || 0), z2, s / this.s0];
    }
    pick(e) {
      if (!this.P) return -1;
      const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = -1, bd = 1e9;
      this.P.forEach((p, i) => { const n = this.d.nodes[i]; if (!p || n.pick === false) return; const dd = Math.hypot(p[0] - mx, p[1] - my); const lim = Math.max(12, (n.size || 3) * 3 * p[3]); if (dd < lim && dd - p[3] * 4 < bd) { bd = dd - p[3] * 4; best = i; } });
      return best;
    }
    loop(now) { this.raf = requestAnimationFrame(this.loop); this._rafAt = performance.now(); this.frame(performance.now()); }
    frame(now) {
      if (!this.cv) return;
      if (!(this._vc = ((this._vc || 0) + 1) % 20)) { const r = this.getBoundingClientRect(); this.visible = r.bottom > 0 && r.top < innerHeight && r.height > 2; if (r.width && (Math.abs(r.width - this.W) > 1 || Math.abs(r.height - this.H) > 1)) this.resize(); }
      if (!this.visible) { this.last = now; return; }
      const dt = Math.min(0.05, ((now - (this.last || now)) / 1000)); this.last = now;
      const rm = RM.matches, d = this.d, R = d.radius || 300;
      if (!this.started) { this.started = true; this.t0 = now; }
      const intro = rm ? 1 : ease((now - this.t0) / 1900);
      if (this.tYaw != null) { const df = this.tYaw - this.yaw; this.yaw += rm ? df : df * Math.min(1, dt * 3.2); if (rm || Math.abs(df) < 0.002) this.tYaw = null; }
      else { if (rm) this.vy = 0; this.yaw += this.vy; this.vy *= 0.93; if (!rm && Math.abs(this.vy) < 1e-4 && now > this.idle && d.autoRotate !== false) this.yaw += (d.autoRotate || 0.07) * dt; }
      if (this.tPitch != null) { const dp = this.tPitch - this.pitch; this.pitch += rm ? dp : dp * Math.min(1, dt * 3); if (rm || Math.abs(dp) < 0.002) this.tPitch = null; }
      const yaw = this.yaw + (1 - intro) * 2.2;
      this._cy = Math.cos(yaw); this._sy = Math.sin(yaw); this._cp = Math.cos(this.pitch); this._sp = Math.sin(this.pitch);
      this.dist = R * 3.6; this.F = (this.d.zoom || 0.4) * Math.min(this.W < 500 ? this.W * 0.8 : this.W, this.H) * this.dist / R; this.s0 = this.F / this.dist;
      const ctx = this.ctx, C = this.col, W = this.W, H = this.H, t = now / 1000;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const rgba = (c, a) => 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + clamp(a, 0, 1).toFixed(3) + ')';
      const fog = (z) => clamp(1 - (z + R) / (2.4 * R), 0.18, 1);
      // nebula
      const nb = d.nebula == null ? 1 : d.nebula; // optional intensity; 0 skips the full-canvas fill
      if (nb > 0) {
        const nx = W / 2 + (this.ox || 0), ny = H / 2 + (this.oy || 0), neb = ctx.createRadialGradient(nx, ny, 0, nx, ny, Math.max(W, H) * 0.55);
        neb.addColorStop(0, rgba(C.acc, 0.09 * intro * nb)); neb.addColorStop(0.45, rgba(C.cy, 0.025 * intro * nb)); neb.addColorStop(1, rgba(C.acc, 0));
        ctx.fillStyle = neb; ctx.fillRect(0, 0, W, H);
      }
      // dust (depth parallax)
      for (const p of this.dust) {
        const q = this.proj(p[0] * R, p[1] * R, p[2] * R); if (q[2] + this.dist < 10) continue;
        const a = (0.1 + p[3] * 0.4) * fog(q[2]) * intro * (rm ? 1 : 0.75 + 0.25 * Math.sin(t * 1.3 + p[4]));
        ctx.fillStyle = rgba(C.fg, a); const s = (0.5 + p[3] * 0.9) * q[3]; ctx.fillRect(q[0] - s / 2, q[1] - s / 2, s, s);
      }
      // nodes → projected
      const TR = this.textRects = []; // canvas text boxes drawn this frame (element px) — pinned readouts keep clear of them
      const TF = this.faintRects = []; // faint labels (6–20% alpha): readouts keep off them too, at a lower cost
      const TT = this.textTried = []; // every text it meant to draw: { node (-1: annotation), text, a: alpha drawn, f: the free-area fade, ax/ay: its anchor }
      // text that would cross the edge of the free stage area (el.clip, set by vega-camera) fades whole, never cut mid-word;
      // el.textAlpha scales all canvas text (a backdrop stop, the loader gate)
      const CL = this.clip, TA = this.textAlpha == null ? 1 : this.textAlpha;
      // a label's centre x, kept inside the canvas and, with a clip, inside the free area (labels slide in, as at the canvas edge)
      const FE = (innerWidth <= 600 ? 12 : 24) + 8; // text stays inside the HUD frame (its inset + a little air)
      const cx = (x, w, y0, y1) => { let lo = w / 2 + FE, hi = W - w / 2 - FE;
        if (CL) { lo = Math.max(lo, CL.L + 58 + w / 2); for (const r of CL.rects) if (r[0] > -1e8 && y1 > r[1] - 20 && y0 < r[3] + 20 && r[0] > x - w) hi = Math.min(hi, r[0] - 58 - w / 2); }
        return lo <= hi ? clamp(x, lo, hi) : (lo + hi) / 2; };
      const tf = (l, t, r, b) => { let f = TA; if (!CL || f <= 0) return f;
        f = Math.min(f, clamp((l - CL.L - 28) / 28, 0, 1));
        for (const h of CL.rects) { const dx = Math.max(h[0] - r, l - h[2], 0), dy = Math.max(h[1] - b, t - h[3], 0); f = Math.min(f, Math.max(clamp((dx - 28) / 28, 0, 1), clamp((dy - 20) / 20, 0, 1))); }
        return f; };
      const P = this.P = d.nodes.map((n) => {
        const e = rm ? 1 : ease((now - this.t0 - n.dl) / 1500);
        let x = n.x, y = n.y, z = n.z;
        if (n._f && this._mk) { const k = this._mk(now), kk = clamp(k * (1 + (n.lag || 0)) - (n.lag || 0), 0, 1); x = n._f.x + (x - n._f.x) * kk; y = n._f.y + (y - n._f.y) * kk; z = n._f.z + (z - n._f.z) * kk; n._ka = n._f.a + ((n.alpha != null ? n.alpha : 1) - n._f.a) * kk; n._ks = n._f.s + ((n.size || 3) - n._f.s) * kk; } else { n._ka = null; n._ks = null; }
        const q = this.proj(x * e, y * e, z * e); q.push(e, this._vx, this._vy); return q; // [sx, sy, depth, scale, intro, vx, vy]
      });
      const WL = d.wave ? (d.wave.layers || 4) : 0, W8 = d.wave && d.wave.at != null ? d.wave.at : d.wave && !rm ? ((t / (d.wave.period || 0.9)) % (WL + 1.8)) - 0.4 : -9; // wave.at pins the wave to a layer (a page's scroll picks it)
      // guides
      ctx.lineWidth = 1;
      for (const g of d.guides || []) {
        ctx.beginPath(); let za = 0;
        g.pts.forEach((pt, k) => { const q = this.proj(pt[0], pt[1], pt[2]); za += q[2]; k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
        if (g.close) ctx.closePath();
        ctx.lineWidth = g.w || 1; ctx.strokeStyle = rgba(C[g.color || 'acc'], (g.alpha || 0.15) * fog(za / g.pts.length) * intro); ctx.setLineDash(g.dash || []); ctx.stroke(); ctx.lineWidth = 1;
        if (g.fill) { ctx.fillStyle = rgba(C[g.color || 'acc'], g.fill * intro); ctx.fill(); }
      }
      ctx.setLineDash([]);
      // strong ink drawn this frame (links at ≥25% alpha every 8px, star dots weighted 6), in an 8px grid of points: labels are placed
      // where the least of it lies under them; exposed as el.inkPts for the overlap checks
      const IG = new Map(), INK = this.inkPts = [];
      const ink = (x, y, w) => { const p = [x, y, w]; INK.push(p); const k = (Math.floor(x / 8) + 4096) * 8192 + Math.floor(y / 8) + 4096; const c = IG.get(k); if (c) c.push(p); else IG.set(k, [p]); };
      const inkIn = (r) => { let n = 0; for (let gx = Math.floor(r[0] / 8); gx <= Math.floor(r[2] / 8); gx++) for (let gy = Math.floor(r[1] / 8); gy <= Math.floor(r[3] / 8); gy++) { const c = IG.get((gx + 4096) * 8192 + gy + 4096); if (c) for (const p of c) if (p[0] > r[0] && p[0] < r[2] && p[1] > r[1] && p[1] < r[3]) n += p[2]; } return n; };
      // edges
      const sel = this.sel, hov = this.hover;
      const E = (d.edges || []).map((ed, k) => {
        const a = P[ed.a], b = P[ed.b]; if (!a || !b) return null;
        let c = null; if (ed.mid) c = this.proj(ed.mid[0], ed.mid[1], ed.mid[2]);
        const on = ed.a === sel || ed.b === sel || ed.a === hov || ed.b === hov;
        const base = ed.alpha != null ? ed.alpha : 0.4;
        const al = (on ? Math.max(0.9, base) : (sel >= 0 ? base * 0.7 : base)) * fog((a[2] + b[2]) / 2) * Math.min(a[4], b[4]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); c ? ctx.quadraticCurveTo(c[0], c[1], b[0], b[1]) : ctx.lineTo(b[0], b[1]);
        ctx.setLineDash(ed.dash || []); ctx.lineWidth = (ed.w || 1) * (on ? 1.5 : 1); ctx.strokeStyle = rgba(C[ed.color || 'acc'], al); ctx.stroke();
        if (al >= 0.25) { const m = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 8)); for (let t = 1; t < m; t++) { const u = t / m, v = 1 - u; ink(c ? v * v * a[0] + 2 * v * u * c[0] + u * u * b[0] : a[0] + (b[0] - a[0]) * u, c ? v * v * a[1] + 2 * v * u * c[1] + u * u * b[1] : a[1] + (b[1] - a[1]) * u, 1); } }
        return { a, b, c, ed };
      });
      ctx.setLineDash([]);
      if (W8 > -1) {
        const sp = sprite(C.acc);
        for (const x of E) { if (!x) continue; const la = d.nodes[x.ed.a].layer; if (la == null) continue; const u = W8 - la; if (u <= 0 || u >= 1) continue;
          const px = x.a[0] + (x.b[0] - x.a[0]) * u, py = x.a[1] + (x.b[1] - x.a[1]) * u, sc = x.a[3] + (x.b[3] - x.a[3]) * u;
          ctx.strokeStyle = rgba(C.acc, 0.35 * Math.sin(u * Math.PI)); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x.a[0], x.a[1]); ctx.lineTo(px, py); ctx.stroke();
          ctx.globalAlpha = 0.8; ctx.drawImage(sp, px - 7 * sc, py - 7 * sc, 14 * sc, 14 * sc); ctx.globalAlpha = 1; ctx.fillStyle = rgba(C.fg, 0.9); ctx.fillRect(px - 0.8, py - 0.8, 1.6, 1.6); }
      }
      // pulses
      if (!rm && d.pulse && intro > 0.6) {
        const seq = E.filter(x => x && x.ed.pulse !== false);
        const pt = (x, u) => { const { a, b, c } = x; if (!c) return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[3] + (b[3] - a[3]) * u]; const v = 1 - u; return [v * v * a[0] + 2 * v * u * c[0] + u * u * b[0], v * v * a[1] + 2 * v * u * c[1] + u * u * b[1], a[3] + (b[3] - a[3]) * u]; };
        const tokens = d.pulse.tokens || 2, per = d.pulse.period || 1.1;
        for (let k = 0; k < tokens; k++) {
          const g = ((t / per + k * seq.length / tokens) % seq.length + seq.length) % seq.length; const x = seq[Math.floor(g)]; if (!x) continue; const u = g % 1;
          for (let j = 7; j >= 0; j--) { const uu = u - j * 0.035; if (uu < 0) continue; const p = pt(x, uu); const a = (1 - j / 8); ctx.fillStyle = rgba(C.fg, a * 0.75); const s = (1.1 + (7 - j) * 0.16) * p[2]; ctx.beginPath(); ctx.arc(p[0], p[1], s, 0, 6.283); ctx.fill(); }
          const p = pt(x, u); const sp = sprite(C.acc); const s = 34 * p[2]; ctx.globalAlpha = 0.9; ctx.drawImage(sp, p[0] - s / 2, p[1] - s / 2, s, s); ctx.globalAlpha = 1;
        }
      }
      // nodes, back to front
      const order = d.nodes.map((n, i) => i).sort((i, j) => P[j][2] - P[i][2]);
      const LB = []; // labels to place once every star is drawn
      for (const i of order) {
        const n = d.nodes[i], q = P[i]; if (q[2] + this.dist < 10) continue;
        const on = i === sel, ho = i === hov, f = fog(q[2]), e = q[4];
        const cc = C[n.color || 'fg'];
        const tw = rm ? 1 : 0.86 + 0.14 * Math.sin(t * 1.7 + n.tw);
        const a = (n._ka != null ? n._ka : (n.alpha != null ? n.alpha : 1)) * f * e * tw;
        const wb = n.layer != null && W8 > -1 ? Math.exp(-Math.pow(W8 - n.layer, 2) * 6) : 0;
        const sz = (n._ks != null ? n._ks : (n.size || 3)) * q[3] * (on ? 1.45 : ho ? 1.2 : 1) * (1 + wb * 0.7);
        if (a < 0.01) continue;
        // a planet drawn by another layer as a lit 3D sphere (el.externalBodies, index on WebGL): only its rings and label here
        const ext = n.kind === 'planet' && this.externalBodies;
        if (ext) { ink(q[0], q[1], 6); if (on || ho) { ctx.strokeStyle = rgba(C.acc, (on ? 0.9 : 0.5) * e); ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.lineDashOffset = rm ? 0 : -t * 8; ctx.beginPath(); ctx.arc(q[0], q[1], sz * 4.6 + 7, 0, 6.283); ctx.stroke(); ctx.setLineDash([]); }
          if (n.label && (d.showLabels || on || ho) && !(CL && (q[0] < 0 || q[0] > W || q[1] < 0 || q[1] > H))) LB.push({ i, n, q, sz: sz * 3.2, on, ho, f, e }); continue; }
        const sp = sprite(n.glow ? C[n.glow] : C.acc); const gs = sz * (n.kind === 'core' ? 13 : 8);
        ctx.globalAlpha = Math.min(1, a * 0.85 * (on ? 1.3 : 1) * (1 + wb * 1.5)); ctx.drawImage(sp, q[0] - gs / 2, q[1] - gs / 2, gs, gs); ctx.globalAlpha = 1;
        if (n.kind === 'core' || on) {
          const L = sz * (n.kind === 'core' ? 7 : 5.5) * (rm ? 1 : 0.92 + 0.08 * Math.sin(t * 2));
          const gr = (x1, y1, x2, y2) => { const g = ctx.createLinearGradient(x1, y1, x2, y2); g.addColorStop(0, rgba(cc, 0)); g.addColorStop(0.5, rgba(cc, a * 0.9)); g.addColorStop(1, rgba(cc, 0)); return g; };
          ctx.lineWidth = 1.1; ctx.strokeStyle = gr(q[0], q[1] - L, q[0], q[1] + L); ctx.beginPath(); ctx.moveTo(q[0], q[1] - L); ctx.lineTo(q[0], q[1] + L); ctx.stroke();
          ctx.strokeStyle = gr(q[0] - L * 0.8, q[1], q[0] + L * 0.8, q[1]); ctx.beginPath(); ctx.moveTo(q[0] - L * 0.8, q[1]); ctx.lineTo(q[0] + L * 0.8, q[1]); ctx.stroke();
        }
        ctx.fillStyle = rgba(cc, a); ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(0.6, sz), 0, 6.283); ctx.fill();
        if (a >= 0.3) ink(q[0], q[1], sz >= 1.6 ? 6 : 1); // a star weighs 6; a tiny dim dot (coverage's not-admitted cells) 1
        if (n.ring) { ctx.strokeStyle = rgba(C[n.ring], a * 0.95); ctx.lineWidth = 1.1; ctx.beginPath(); ctx.arc(q[0], q[1], sz * 2.4 + 2, 0, 6.283); ctx.stroke(); }
        if (on || ho) {
          ctx.strokeStyle = rgba(C.acc, (on ? 0.9 : 0.5) * e); ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.lineDashOffset = rm ? 0 : -t * 8;
          ctx.beginPath(); ctx.arc(q[0], q[1], sz * 3.4 + 6, 0, 6.283); ctx.stroke(); ctx.setLineDash([]);
        }
        // with a clip (inner pages), a star outside the canvas doesn't pin its label to the edge
        if (n.label && (d.showLabels || on || ho) && !(CL && (q[0] < 0 || q[0] > W || q[1] < 0 || q[1] > H))) LB.push({ i, n, q, sz, on, ho, f, e });
      }
      // annotations first (they sit where the data put them), then the stars' labels, the selected/hovered and the
      // nearest first: a label that would land on one already placed tries its other side, then a step further out,
      // so labels sit apart instead of stacking (none is dropped)
      const OC = [], over = (r) => OC.some(o => r[0] < o[2] + 2 && r[2] > o[0] - 2 && r[1] < o[3] + 1 && r[3] > o[1] - 1);
      for (const l of d.labels || []) {
        const q = this.proj(l.x, l.y, l.z); const f = fog(q[2]);
        if (CL && (q[0] < 0 || q[0] > W || q[1] < 0 || q[1] > H)) continue; // an anchor off the canvas doesn't pin its text to the edge (inner pages)
        ctx.textAlign = l.align || 'center'; ctx.font = l.font || '500 10.5px ui-monospace,Menlo,Consolas,monospace';
        const w = ctx.measureText(l.text).width, al = l.align || 'center';
        const yy = clamp(q[1], 14, H - 8), x = al === 'right' ? cx(q[0] - w / 2, w, yy - 11, yy + 3) + w / 2 : al === 'left' ? cx(q[0] + w / 2, w, yy - 11, yy + 3) - w / 2 : cx(q[0], w, yy - 11, yy + 3);
        // an annotation keeps its anchor, shifted up to 42px up or down (and a little sideways) if the scene's own ink would cross it there
        const tries = []; [0, -14, 14, -28, 28, -42, 42].forEach(dy => [0, -24, 24].forEach(dx => tries.push([dx, dy])));
        const best = tries.map(([dx, dy], k) => { const y = clamp(q[1] + dy, 14, H - 8), l0 = (al === 'right' ? x - w : al === 'left' ? x : x - w / 2) + dx, r = [l0, y - 11, l0 + w, y + 3]; return { y, dx, s: (l0 < FE || l0 + w > W - FE ? 3000 : 0) + (over(r) ? 1000 : 0) + (tf(...r) < TA * 0.5 ? 500 : 0) + inkIn(r) + k * 0.15 }; }).sort((u, v) => u.s - v.s)[0];
        const ty = best.y, tx = (al === 'right' ? x - w : al === 'left' ? x : x - w / 2) + best.dx, ko = tf(tx, ty - 11, tx + w, ty + 3);
        ctx.fillStyle = rgba(C[l.color || 'dim'], Math.max(0.55, f) * intro * ko); ctx.fillText(l.text, x + best.dx, ty); if (Math.max(0.55, f) * intro * ko > 0.2) TR.push([tx, ty - 11, tx + w, ty + 3]); TT.push({ node: -1, text: l.text, a: Math.max(0.55, f) * intro * ko, f: ko, ax: q[0], ay: q[1] });
        if (Math.max(0.55, f) * intro * ko > 0.06) OC.push([tx - 6, ty - 11, tx + w + 6, ty + 3]); // annotations keep 6px either side
      }
      LB.sort((a, b) => (b.on || b.ho) - (a.on || a.ho) || a.q[2] - b.q[2]);
      for (const { i, n, q, sz, on, ho, f, e } of LB) {
        ctx.textAlign = 'center'; ctx.font = '500 10.5px ui-monospace,Menlo,Consolas,monospace';
        const sw = n.sub ? ctx.measureText(n.sub).width : 0;
        ctx.font = (on ? '500 ' : '400 ') + Math.round(13 + (on ? 2 : 0)) + 'px "Helvetica Neue",Helvetica,Arial,sans-serif';
        const lw = Math.max(ctx.measureText(n.label).width, sw), la = on || ho ? 1 : clamp((f - 0.42) / 0.38, 0.12, 1);
        const place = (below, step, dx = 0) => { const ly = clamp(q[1] + (below ? sz * 2.4 + 18 + step : -(sz * 2.4 + 12 + step)), 26, H - 22), lx = cx(q[0] + dx, lw, ly - 13, ly + 19);
          return { below, ly, lx, r: [lx - lw / 2, ly - 13 - (n.sub && !below ? 16 : 0), lx + lw / 2, ly + 4 + (n.sub && below ? 15 : 0)] }; };
        const b0 = n.labelPos !== 'above', tries = [place(b0, 0), place(!b0, 0), place(b0, 0, -24), place(b0, 0, 24), place(!b0, 0, -24), place(!b0, 0, 24), place(b0, 16), place(!b0, 16), place(b0, 32), place(!b0, 32)];
        // the spot with no label under it, not faded by the free area, and the least ink; earlier spots preferred
        const pl = tries.map((t, k) => ({ t, s: (over(t.r) ? 1000 : 0) + (tf(...t.r) < TA * 0.5 ? 500 : 0) + inkIn(t.r) + k * 0.4 })).sort((u, v) => u.s - v.s)[0].t, ko = tf(...pl.r), al = la * e * ko;
        ctx.fillStyle = rgba(on || ho ? C.fg : C.mut, al); ctx.fillText(n.label, pl.lx, pl.ly);
        if (al > 0.2) TR.push(pl.r); else if (al > 0.06) TF.push(pl.r); if (al > 0.06) OC.push(pl.r); TT.push({ node: i, text: n.label, a: al, f: ko, ax: q[0], ay: q[1] }); // faint labels keep their space too
        if (n.sub) { ctx.font = '500 10.5px ui-monospace,Menlo,Consolas,monospace'; ctx.fillStyle = rgba(C.dim, al); ctx.fillText(n.sub, pl.lx, pl.ly + (pl.below ? 15 : -16)); }
      }
    }
  }
  customElements.define('vega-constellation', VegaConstellation);
})();
