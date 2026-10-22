/* vega-readouts — pinned readouts (spec §2.6). Martian Mono labels anchored to constellation nodes,
   placed every frame from the scene's own projected coordinates (scene.P), so they track their stars
   while the camera moves. The active stop's readout is bright with a leader line; the others are dim;
   at ≤600px only the active one shows (spec §5). Text must be existing page copy (CLAUDE.md rule 5).
   Usage: readouts.configure({ scene, activeOnly, items: [{ node, text, stop } | { el, node }], showAt: (activeStop) => bool })
     node — index into scene.d.nodes, or (activeStop) => index (-1 hides it); stop — the stop it belongs to
     el   — an existing page element (e.g. a figure HUD bound to page state) pinned instead of a text label;
            it is always the active readout and keeps its own content
   Decorative (aria-hidden): the same copy is in the page. */
(() => {
  if (customElements.get('vega-readouts')) return;
  const MONO = "'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace";
  const CSS = `
vega-readouts{position:absolute;inset:0;pointer-events:none;display:block;z-index:2;font:500 11px/1 ${MONO};letter-spacing:.22em;color:var(--dim,#8C9EB3)}
vega-readouts .vro{position:absolute;left:0;top:0;display:flex;align-items:center;gap:8px;white-space:nowrap;will-change:transform;transition:opacity .35s,color .35s}
vega-readouts .vro i{display:block;width:18px;height:1px;background:currentColor;opacity:.6;transform-origin:left;transform:scaleX(.45);transition:transform .35s}
vega-readouts .vro.on{color:var(--fg,#E9F0F8)}
vega-readouts .vro.on i{transform:none;background:var(--acc,#9CC8FF);opacity:1}
@media (max-width:600px){vega-readouts .vro{white-space:normal;max-width:calc(100vw - 64px);line-height:1.5}}
@media (prefers-reduced-motion: reduce){vega-readouts .vro,vega-readouts .vro i{transition:none}}
`;
  class VegaReadouts extends HTMLElement {
    connectedCallback() {
      if (!document.getElementById('vega-readouts-css')) { const s = document.createElement('style'); s.id = 'vega-readouts-css'; s.textContent = CSS; document.head.appendChild(s); }
      this.setAttribute('aria-hidden', 'true'); this.active = 0;
      this.onStop = (e) => { this.active = e.detail.index; this.kick(); };
      addEventListener('vegastop', this.onStop);
    }
    disconnectedCallback() { removeEventListener('vegastop', this.onStop); cancelAnimationFrame(this.raf); }
    configure(o) {
      this.o = o; this.rail = document.querySelector('vega-rail');
      [...this.children].forEach(c => c.remove());
      this.els = o.items.map(it => { if (it.el) { it.el.classList.add('vro', 'on'); return it.el; } const d = document.createElement('span'); d.className = 'vro'; d.innerHTML = '<i></i>'; const t = document.createElement('span'); t.textContent = it.text; d.append(t); this.append(d); return d; });
      const cam = document.querySelector('vega-camera'); if (cam && cam.active >= 0) this.active = cam.active;
      this.loop = this.loop.bind(this); this.kick();
    }
    kick() { if (!this.raf && this.o && !document.hidden) this.raf = requestAnimationFrame(this.loop); }
    loop() {
      this.raf = 0;
      const sc = this.o.scene, P = sc && sc.P; if (!P) { this.kick(); return; }
      const small = innerWidth <= 600, show = this.o.showAt ? this.o.showAt(this.active) : true;
      const sop = parseFloat(sc.style.opacity || '1');
      // the content blocks on screen right now (readouts keep out of them)
      const own = new Set(this.els);
      // the blocks only move with the scroll (or a resize, or the active stop changing what is shown): measured again only then
      const bk = scrollY + ',' + innerWidth + ',' + innerHeight + ',' + this.active;
      if (bk !== this._bk) { this._bk = bk; this._blocks = [...document.querySelectorAll('[data-stop] .vp, [data-fighud]')].filter(e => !own.has(e) && +getComputedStyle(e).opacity > 0.05).map(e => e.getBoundingClientRect()).filter(r => r.bottom > 0 && r.top < innerHeight && r.width > 0); }
      const blocks = this._blocks;
      // every layout read happens here, before any write below (no forced style/layout flush per readout per frame)
      const shown = sop > 0.05 && getComputedStyle(sc).display !== 'none', R = sc.getBoundingClientRect(), sizes = this.els.map(e => [e.offsetWidth, e.offsetHeight]);
      const rr = this.rail && !this.rail.classList.contains('vr-compact') ? this.rail.getBoundingClientRect() : null;
      const texts = [];
      // and out of the scene's own canvas text (its node labels and annotations), when the scene is showing
      if (sc.textRects && shown) sc.textRects.forEach(t => texts.push({ left: t[0] + R.left, top: t[1] + R.top, right: t[2] + R.left, bottom: t[3] + R.top }));
      const faint = []; if (sc.faintRects && shown) sc.faintRects.forEach(t => faint.push({ left: t[0] + R.left, top: t[1] + R.top, right: t[2] + R.left, bottom: t[3] + R.top }));
      // the scene's links (edge samples; faint ones under a quarter alpha don't count) and stars, in page px, to keep off
      const links = [], stars = [], writes = [];
      if (shown) {
        P.forEach(p => { if (p && p[4] > 0.5) stars.push([p[0] + R.left, p[1] + R.top]); });
        (sc.d.edges || []).forEach(e => { const a = P[e.a], b = P[e.b], on = e.a === sc.sel || e.b === sc.sel; if (a && b && (on || (e.alpha != null ? e.alpha : 0.4) * (sc.sel >= 0 ? 0.7 : 1) >= 0.25)) { const m = Math.max(2, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 12)); for (let t = 1; t < m; t++) links.push([a[0] + (b[0] - a[0]) * t / m + R.left, a[1] + (b[1] - a[1]) * t / m + R.top]); } }); }
      this.pick = this.pick || [];
      this.o.items.forEach((it, k) => {
        const ni = typeof it.node === 'function' ? it.node(this.active) : it.node;
        const el = this.els[k], q = ni >= 0 ? P[ni] : null, on = it.el ? true : it.stop === this.active;
        const ins = small ? 12 : 24; // inside the HUD frame, below its top row, above its bottom row
        const vis = show && q && q[4] > 0.5 && (!(small || this.o.activeOnly) || on) && q[0] > ins + 16 && q[0] < innerWidth - ins - 70 && q[1] > ins + 64 && q[1] < innerHeight - ins - 56
          && !blocks.some(b => q[0] + R.left > b.left && q[0] + R.left < b.right && q[1] + R.top > b.top && q[1] + R.top < b.bottom); // its star isn't under the text
        const op = vis ? String((on ? 1 : 0.62) * Math.min(1, sop * 1.2)) : '0';
        writes.push(() => { if (!it.el) el.classList.toggle('on', on); el.style.opacity = op; });
        if (q) {
          const gap = Math.max(10, (sc.d.nodes[ni].size || 3) * q[3] * 3.2);
          const w = sizes[k][0], h = sizes[k][1], above = (sc.d.nodes[ni] || {}).labelPos === 'above';
          // candidate spots around the star, preferred first: a block readout (and any at ≤600px) centred above it —
          // below it when the canvas label sits above — a text readout to its right; then the other sides and the
          // diagonals, near and a step further out
          const spots = (g) => { const up = [q[0] - w / 2, q[1] - g - h - 6], down = [q[0] - w / 2, q[1] + g + 18], right = [q[0] + g, q[1] - h / 2], left = [q[0] - g - w, q[1] - h / 2];
            const ur = [q[0] + g * 0.7, q[1] - g * 0.7 - h], ul = [q[0] - g * 0.7 - w, q[1] - g * 0.7 - h], dr = [q[0] + g * 0.7, q[1] + g * 0.7], dl = [q[0] - g * 0.7 - w, q[1] + g * 0.7];
            // and above or below, slid along so the star sits off-centre (for narrow gaps, where the diagonals don't fit)
            const slid = [0.2, 0.35, 0.65, 0.8].flatMap(fr => [[q[0] - w * fr, up[1]], [q[0] - w * fr, down[1]]]);
            return ((it.el || small) ? (above ? [down, up, ur, ul, right, left, dr, dl] : [up, down, ur, ul, right, left, dr, dl]) : [right, ur, dr, up, down, left, ul, dl]).concat(slid); };
          const cands = spots(gap).concat(spots(gap + 36), spots(gap + 72));
          const fit = (c) => [Math.max(ins + 8, Math.min(innerWidth - ins - 8 - w, c[0])), Math.max(ins + 64, Math.min(innerHeight - ins - 60 - h, c[1]))];
          const hits = (c, b) => c[0] < b.right + 6 && c[0] + w > b.left - 6 && c[1] < b.bottom + 4 && c[1] + h > b.top - 4;
          const area = (c, b) => Math.max(0, Math.min(c[0] + w + 6, b.right) - Math.max(c[0] - 6, b.left)) * Math.max(0, Math.min(c[1] + h + 4, b.bottom) - Math.max(c[1] - 4, b.top));
          const inside = (c, p) => p[0] > c[0] - 4 && p[0] < c[0] + w + 4 && p[1] > c[1] - 4 && p[1] < c[1] + h + 4;
          // a spot's cost: canvas text it would cover — any overlap outweighs everything else (text is never hidden; the
          // readout moves instead) — then faint labels, then the scene's links
          // and stars under it, then how far down the preference list it is; the rail and the text blocks are hard limits
          const reach = gap + 72 + w + h + 20;
          const cost = (c, k, T, Ln, Sn, Fa) => (rr && hits(c, rr)) || blocks.some(b => hits(c, b)) ? 1e6
            : (T.some(b => area(c, b) > 0) ? 4 : 0) + T.reduce((a, b) => a + area(c, b), 0) / (w * h) * 30
            + (Fa.some(b => area(c, b) > 0) ? 1.5 : 0) + Fa.reduce((a, b) => a + area(c, b), 0) / (w * h) * 8 + Math.min(3, Ln.filter(p => inside(c, p)).length * 0.6) + Sn.filter(p => inside(c, p)).length * 0.3 + k * 0.04;
          // only what lies near this star can matter (keeps the search cheap with many readouts on screen)
          const near = (p) => Math.abs(p[0] - q[0]) < reach && Math.abs(p[1] - q[1]) < reach;
          const inR = (b) => b.right > q[0] - reach && b.left < q[0] + reach && b.bottom > q[1] - reach && b.top < q[1] + reach;
          const T = texts.filter(inR), Fa = faint.filter(inR), Ln = links.filter(near), Sn = stars.filter(near);
          const fits = cands.map(fit), clean = (c) => cost(c, 0, T, Ln, Sn, Fa) === 0;
          // keep the current spot while it covers nothing (no flicker); else the first spot in preference order that covers
          // nothing; else the cheapest
          const prev = this.pick[k]; let best = -1;
          if (prev != null && prev < fits.length && clean(fits[prev])) best = prev;
          if (best < 0) best = fits.findIndex(c => clean(c));
          if (best < 0) { const costs = fits.map((c, j) => cost(c, j, T, Ln, Sn, Fa)); best = 0; costs.forEach((v, j) => { if (v < costs[best]) best = j; }); }
          this.pick[k] = best;
          const x = fits[best][0], y = fits[best][1];
          const tr = 'translate3d(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px,0)'; writes.push(() => { el.style.transform = tr; });
        }
      });
      writes.forEach(f => f());
      this.kick();
    }
  }
  customElements.define('vega-readouts', VegaReadouts);
})();
