/* vega-camera — the scroll-driven camera (spec §2.3). Native document scroll is kept: the camera reads
   scroll position, maps it through the page's stops, and flies a vega-constellation between them by
   writing its yaw / pitch / d.zoom and the screen offset (ox / oy) that puts the stop's focus point
   where the stop frames it. The sky gets a slow parallax. Under prefers-reduced-motion the camera cuts
   to the active stop instead of flying, and a drag or double-click on the scene is held, not eased back.
   Usage: camera.configure({ scene, sky, stops: [{ el, focus, yaw, pitch, zoom, at, opacity, select }] })
     el      — the stop's section (its top is where the stop sits in the scroll)
     focus   — [x, y, z] world point, or () => [x, y, z]
     at      — (W, H, scene) => [x, y] screen point (scene px) where focus lands
     yaw / pitch / zoom / opacity — the pose; nebula — the scene's nebula intensity (default 1);
     select — node index to highlight (-1 none); cross — 0..1 share of the alternate scene (default 0)
     offset — px (or () => px) the stop's resting scroll sits above its element's top (default 0)
   configure({ free: true }) — inner pages: the scenes draw only in the stage area left free by the rail and the
     content blocks (.vp, footer, a fixed HUD text block); everything else fades out under a soft CSS mask, so no node, path or canvas
     label sits under the rail or the text. At ≥900px a side block (narrower than 60% of the frame) cuts the
     stage as a rectangle from its left edge to the frame's right edge; a full-width block (≤899px, or a wide
     stop) cuts a band (below 900px it leaves the scene at 15%). Both follow the blocks exactly as they scroll. The scenes get the same area as `clip`, so canvas text near an edge fades whole
     instead of being cut mid-word. camera.free(i) gives the area at stop i's resting scroll, camera.fitBoxes(i)
     the boxes a subject can fill there, camera.fitIn(i, …) the best of them.
     Per stop with free: fit — true (every node) or [node indices], plus optional fitPts [[x, y, z], …]: zoom and `at` are fitted so that subject and
     its labels sit whole inside the free area at rest; under — the stop's block dims the stage (to 40%) instead
     of cutting it, for a backdrop stop (at most 19% behind its text); text — 0..1 share of the scene's canvas text (default 1).
   configure({ scene: null, stops, free: true, mask: [els] }) — stops only (els get the free-area mask): the active stop, arrival, rail and reel, for a page that draws its own
     view from camera.progress() (promise's sky).
   configure({ home: k }) — the page is planet k of the VEGA system (INDEX order): the stage's vega-galaxy floats beside it.
   configure({ alt }) — an optional second scene that crossfades in as `cross` rises (chain's campaign);
     whichever scene is fully faded out is display:none, so it stops rendering
   Events out: window 'vegastop' { detail: { index, previous, count, source } } when the active stop changes;
     window 'vegaarrive' { detail: { index } } once the camera has settled on the active stop (headline decode).
   Read: camera.active, camera.count. camera.dive(ms): zoom into the current focus over ms (the fly-through, §2.8); dive(0) cancels. */
(() => {
  if (customElements.get('vega-camera')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (t) => t * t * (3 - 2 * t);
  const HOLD = 0.16; // share of each leg held at either end, so a stop settles before the next flight
  const hold = (f) => smooth(clamp((f - HOLD) / (1 - 2 * HOLD), 0, 1));
  const DOLLY = 0.14; /* G8: was 0.32 */ // share of zoom given up at mid-leg in a reel shot (vega-reel)
  const SWAY = 0.1, SWAY_W = 0.13; // idle sway (rad, rad/s): the slow drift Hubtown's set pieces have, bounded so a stop's framing holds

  class VegaCamera extends HTMLElement {
    connectedCallback() { this.style.display = 'none'; this.active = -1; this.count = 0; }
    disconnectedCallback() { this.stop(); }

    configure(o) {
      this.stop();
      this.o = o; this.count = o.stops.length; this.active = -1;
      this.cur = null; this.user = { yaw: 0, pitch: 0 }; this.lastApplied = null; this.t0 = performance.now(); this.last = 0;
      if (o.scene && o.scene.setData && o.scene.d) o.scene.d.autoRotate = false;
      this.onScroll = () => this.kick();
      this.onVis = () => { if (!document.hidden) this.kick(); };
      this.onPtr = (e) => {
        // reduced motion: a double-click returns to the stop's pose at once (no easing)
        if (e.type === 'dblclick' && RM.matches) { this.user = { yaw: 0, pitch: 0 }; this.lastApplied = null; o.scene.tYaw = null; o.scene.tPitch = null; }
        if (e.type !== 'pointermove' || e.buttons) this.kick();
      };
      if (o.scene) ['pointermove', 'pointerup', 'dblclick'].forEach(t => o.scene.addEventListener(t, this.onPtr));
      addEventListener('scroll', this.onScroll, { passive: true });
      addEventListener('resize', this.onScroll);
      document.addEventListener('visibilitychange', this.onVis);
      this.loop = this.loop.bind(this);
      // home: the page is planet o.home of the VEGA system — its galaxy (vega-galaxy in the stage) floats beside that planet
      // and orbits it as the page scrolls
      if (o.home != null) { const hook = () => { const g = document.querySelector('vega-galaxy'); if (!g) return; if (g.failed) return; if (!g.SYS) { this._gh = setTimeout(hook, 120); return; }
        g.home = o.home; g.system = true; g.exposure = o.homeExposure == null ? 0.7 : o.homeExposure; g.homeU = () => this.progress().lin / Math.max(1, this.count - 1); }; hook(); }
      this.kick();
    }
    stop() {
      cancelAnimationFrame(this.raf); this.raf = 0;
      if (this.onScroll) { removeEventListener('scroll', this.onScroll); removeEventListener('resize', this.onScroll); document.removeEventListener('visibilitychange', this.onVis); if (this.o.scene) ['pointermove', 'pointerup', 'dblclick'].forEach(t => this.o.scene.removeEventListener(t, this.onPtr)); }
    }
    kick() { if (!this.raf && !document.hidden) this.raf = requestAnimationFrame(this.loop); }
    dive(ms) { this.diveT0 = ms ? performance.now() : 0; this.diveMs = ms || 1; this.kick(); }

    // a stop's resting scroll sits `offset` px above its element's top (number or function; a figure comes to rest below the HUD)
    off(s) { return typeof s.offset === 'function' ? s.offset() : (s.offset || 0); }
    // fractional stop index from the scroll position; `lin` is the unheld value used for the active stop
    // the free-area mask (see configure): applied to the scenes, or to a stops-only page's own drawing (o.mask)
    maskTo(f, opacity, els, boxes) {
        const E = 56, F = 40, W = innerWidth;
      const op = Math.max(0.01, opacity), L = Math.round(f.left), C = Math.round(f.col);
      const sd = f.side.map(h => h.map(Math.round)), hs = f.holes.map(h => [Math.round(h[0]), Math.round(h[1]), h[2] ? Math.min(1, Math.round(19 / op) / 100) : W < 900 ? 0.15 : 0]);
      const key = L + '|' + C + '|' + sd.join(';') + '|' + hs.join(';');
      if (key !== this._maskKey) {
        this._maskKey = key;
        // an under band keeps the backdrop at ≤19% behind its text; below 900px every text band leaves the scene at 15%,
        // so the flight between stops stays visible while a block scrolls over it (its canvas text still fades whole)
        const bands = ['#000 0px'];
        hs.forEach(([t, b, a]) => { const cc = a ? 'rgba(0,0,0,' + a + ')' : 'transparent'; bands.push('#000 ' + (t - F) + 'px', cc + ' ' + t + 'px', cc + ' ' + b + 'px', '#000 ' + (b + F) + 'px'); });
        const rows = ['transparent 0px']; sd.forEach(([t, b]) => rows.push('transparent ' + (t - F) + 'px', '#000 ' + t + 'px', '#000 ' + b + 'px', 'transparent ' + (b + F) + 'px'));
        const layers = ['linear-gradient(to bottom,' + bands.join(',') + ')', 'linear-gradient(to right,' + (L > 0 ? 'transparent ' + L + 'px,#000 ' + (L + E) + 'px' : '#000 0px,#000 100%') + ')'];
        if (sd.length && C < W) layers.push('linear-gradient(to right,transparent ' + (C - E) + 'px,#000 ' + C + 'px)', 'linear-gradient(to bottom,' + rows.join(',') + ')');
        const m = layers.join(','), comp = sd.length && C < W ? ['intersect', 'subtract', 'intersect', 'add'] : ['intersect', 'add'], wk = sd.length && C < W ? ['source-in', 'source-out', 'source-in', 'source-over'] : ['source-in', 'source-over'];
        els.forEach((el, n) => {
          if (!el) return; el.style.webkitMaskImage = el.style.maskImage = m; el.style.webkitMaskComposite = wk.join(','); el.style.maskComposite = comp.join(',');
          const r = boxes && boxes[n]; if (!r) return;
          el.clip = { L: L > 0 ? L - r.left : -1e9, rects: sd.map(([t, b]) => [C - r.left, t - r.top, 1e9, b - r.top]).concat(hs.map(([t, b]) => [-1e9, t - r.top, 1e9, b - r.top])) };
        });
      }
    }
    progress() {
      const st = this.o.stops, max = Math.max(0, document.documentElement.scrollHeight - innerHeight), y = scrollY;
      const tops = st.map(s => Math.min(max, s.el.getBoundingClientRect().top + y - this.off(s)));
      if (y <= tops[0]) return { p: 0, lin: 0 };
      for (let i = 0; i < tops.length - 1; i++) {
        if (y < tops[i + 1]) { const f = (y - tops[i]) / Math.max(1, tops[i + 1] - tops[i]); return { p: i + hold(f), lin: i + f }; }
      }
      return { p: tops.length - 1, lin: tops.length - 1 };
    }
    pose(i) {
      const s = this.o.stops[i], sc = this.o.scene;
      const W = sc.W || innerWidth, H = sc.H || innerHeight;
      const q = { yaw: s.yaw, pitch: s.pitch, zoom: s.zoom, opacity: s.opacity == null ? 1 : s.opacity, nebula: s.nebula == null ? 1 : s.nebula, cross: s.cross || 0, text: s.text == null ? 1 : s.text,
        focus: typeof s.focus === 'function' ? s.focus() : s.focus, at: s.at(W, H, sc) };
      if (s.fit && this.o.free) this.fit(i, q);
      return q;
    }
    mix(a, b, t) {
      return { yaw: lerp(a.yaw, b.yaw, t), pitch: lerp(a.pitch, b.pitch, t), zoom: lerp(a.zoom, b.zoom, t), opacity: lerp(a.opacity, b.opacity, t), nebula: lerp(a.nebula, b.nebula, t), cross: lerp(a.cross, b.cross, t), text: lerp(a.text, b.text, t),
        focus: a.focus.map((v, k) => lerp(v, b.focus[k], t)), at: a.at.map((v, k) => lerp(v, b.at[k], t)) };
    }
    // the constellation's own projection, so the focus point can be pinned to its screen spot
    project(sc, p, yaw, pitch, zoom) {
      const R = (sc.d && sc.d.radius) || 300, W = sc.W || innerWidth, H = sc.H || innerHeight;
      const dist = R * 3.6, F = zoom * Math.min(W < 500 ? W * 0.8 : W, H) * dist / R;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = p[0] * cy - p[2] * sy, z1 = p[0] * sy + p[2] * cy, y1 = p[1] * cp - z1 * sp, z2 = p[1] * sp + z1 * cp;
      const s = F / (z2 + dist);
      return [W * 0.5 + x1 * s, H / 2 - y1 * s];
    }

    // the stage area left free by the rail and the content blocks (see configure({ free })); with a stop index,
    // the layout as it will be when that stop is at rest. Returns { left: the rail's cut, col: the side blocks' left
    // edge, side: [top, bottom] rows of the side blocks (each is a cut-out rectangle from col to the right edge),
    // holes: [top, bottom, alpha] full-width bands }.
    free(i) {
      const W = innerWidth, H = innerHeight, r = document.querySelector('vega-rail'), st = this.o.stops;
      const left = W >= 900 && r && !r.classList.contains('vr-compact') ? r.getBoundingClientRect().right + 8 : 0;
      let dy = 0;
      if (i != null) { const max = Math.max(0, document.documentElement.scrollHeight - H); dy = scrollY - Math.min(max, st[i].el.getBoundingClientRect().top + scrollY - this.off(st[i])); }
      let col = W; const side = [], holes = [];
      // below 900px a stop may name its text band (`band: () => el`): inside its block only that element cuts the stage
      const bs = st[i != null ? i : Math.max(0, this.active)], bandEl = W < 900 && bs && bs.band ? bs.band() : null;
      // text blocks scroll with the page; a fixed HUD text block (data's corner HUD; not a pinned readout) stays put
      document.querySelectorAll('.vp, footer, [data-fighud]:not(.vro)' + (W < 900 ? ', [data-fighud-narrow]' : '')) /* -narrow: a fixed readout that is text only on phones (install's step readout in the top band) */.forEach(e => {
        const fixed = e.matches('[data-fighud]'), cs = getComputedStyle(e); if (fixed && +cs.opacity < 0.05) return;
        const b = (bandEl && e.contains(bandEl) ? bandEl : e).getBoundingClientRect(); let top = b.top + (fixed ? 0 : dy), bottom = b.bottom + (fixed ? 0 : dy);
        // a sticky block (a stop with sub-stops) doesn't move with the scroll while stuck: predict it from its parent
        if (dy && cs.position === 'sticky') { const pr = e.parentElement.getBoundingClientRect(), pc = getComputedStyle(e.parentElement), h = b.height;
          const nat = pr.top + parseFloat(pc.paddingTop) + dy, lim = pr.bottom - parseFloat(pc.paddingBottom) - h + dy; top = Math.min(Math.max(nat, parseFloat(cs.top) || 0), lim); bottom = top + h; }
        if (!b.width || bottom < -80 || top > H + 80) return;
        if (W >= 900 && b.width < W * 0.6) { col = Math.min(col, b.left - 12); side.push([top - 8, bottom + 8]); return; }
        // a block belongs to the last stop at or before it; an `under` stop's block dims the stage instead of cutting it
        let k = -1; st.forEach((s, j) => { if (s.el === e || s.el.contains(e) || (s.el.compareDocumentPosition(e) & 4)) k = j; });
        holes.push([top - 8, bottom + 8, k >= 0 && st[k].under ? 1 : 0]); // under: dimmed in loop() to the scene's opacity
      });
      // G7: the HUD's top bands only appear under text, so the scene itself keeps clear of the nav row and the brand
      if (document.querySelector('vega-hud')) holes.push([-100, (W <= 600 ? 12 : 24) + 44, 0]);
      const merge = (a) => { a.sort((x, y) => x[0] - y[0]); const m = []; a.forEach(h => { const l = m[m.length - 1]; if (l && l[2] === h[2] && h[0] <= l[1] + 80) l[1] = Math.max(l[1], h[1]); else m.push(h.slice()); }); return m; };
      return { left, col, side: merge(side), holes: merge(holes) };
    }
    // the boxes a subject (nodes and their labels) can fill at stop i's rest, inside the mask's opaque part and the HUD
    // frame: the column left of the side blocks, and each full-width gap above, between or below them
    fitBoxes(i) {
      const W = innerWidth, H = innerHeight, f = this.free(i), E = 56, F = 40, edge = (W <= 600 ? 12 : 24) + 16;
      const L = f.left > 0 ? f.left + E : edge, R = W - edge, T = 76, B = H - 64;
      // vertical gaps in [T, B] once the given bands (grown by the mask's ramp) are taken out
      const gaps = (bl) => { const out = []; let y = T; bl.slice().sort((a, b) => a[0] - b[0]).forEach(([t, b]) => { if (t - F > y) out.push([y, Math.min(B, t - F)]); y = Math.max(y, b + F); }); if (B > y) out.push([y, B]); return out.filter(g => g[1] - g[0] > 40); };
      const cuts = f.holes.filter(h => h[2] < 0.5), boxes = [];
      if (f.col < W) gaps(cuts).forEach(g => boxes.push({ L, R: f.col - E, T: g[0], B: g[1] }));
      // below 900px the scene sits above the text it goes with (the top gap only), the way a phone reads
      gaps(cuts.concat(f.side)).slice(0, W < 900 ? 1 : 99).forEach(g => boxes.push({ L, R, T: g[0], B: g[1] }));
      return boxes.filter(b => b.R - b.L > 40);
    }
    // the fit of a subject with screen extents x0..x1, y0..y1 (relative to its anchor, at scale 1) and label room
    // px / pt / pb. Returns { k, at } — the scale (≤ 1) and the anchor's spot.
    fitIn(i, x0, x1, y0, y1, px, pt, pb, boxes) {
      // boxes come beside-the-text first, then top to bottom; the earliest one within 80% of the best scale wins, so
      // the subject keeps its place from stop to stop instead of jumping to whichever gap is a little larger
      const fits = (boxes || this.fitBoxes(i)).map(b => {
        const k = clamp(Math.min(1, (b.R - b.L - 2 * px) / Math.max(1, x1 - x0), (b.B - b.T - pt - pb) / Math.max(1, y1 - y0)), 0.3, 1);
        return { k, at: [(b.L + b.R) / 2 - (x0 + x1) / 2 * k, (b.T + b.B) / 2 - ((y0 * k - pt) + (y1 * k + pb)) / 2] };
      });
      const top = Math.max(0, ...fits.map(f => f.k));
      return fits.find(f => f.k >= top * 0.8) || null;
    }
    // fit a stop's subject (offsets from the focus scale with zoom, so the zoom that fits is exact), sampled across
    // the idle sway, so the framing holds while it drifts
    fit(i, q) {
      const s = this.o.stops[i], sc = this.o.scene, nodes = (sc.d && sc.d.nodes) || [];
      // keyed on the node positions (data moves its nodes in place between steps)
      let sig = 0; for (let k = 0; k < nodes.length; k += 7) sig += (nodes[k].x || 0) + 3 * (nodes[k].y || 0) + 7 * (nodes[k].z || 0);
      // and on the stop's resting layout as measured now (sections that reveal with a transform settle after load)
      const idx0 = s.fit === true ? null : s.fit; // which stars (a getter may follow the page's selection)
      const fb = this.fitBoxes(i), key = [innerWidth, innerHeight, nodes.length, Math.round(sig), Math.round(q.zoom * 1000), idx0 ? (idx0.length < 40 ? idx0.join('.') : idx0.length) : 'all'].join(',') + '|' + fb.map(b => [b.L, b.R, b.T, b.B].map(v => Math.round(v / 4)).join(',')).join(';');
      const c = (this._fit || (this._fit = {}))[i];
      if (c && c.key === key && c.d === sc.d) { q.zoom = c.zoom; q.at = c.at; return; } // a new setData refits too
      const idx = idx0 || nodes.map((n, k) => k), sh = { d: sc.d, W: innerWidth, H: innerHeight };
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      [-SWAY, 0, SWAY].forEach(dw => {
        const o = this.project(sh, q.focus, q.yaw + dw, q.pitch, q.zoom);
        const pts = idx.map(k => nodes[k]).filter(Boolean).map(n => [n.x, n.y, n.z]).concat(s.fitPts || []); // fitPts: world points to keep in frame too (annotations)
        pts.forEach(w => { const p = this.project(sh, w, q.yaw + dw, q.pitch, q.zoom), dx = p[0] - o[0], dy = p[1] - o[1]; x0 = Math.min(x0, dx); x1 = Math.max(x1, dx); y0 = Math.min(y0, dy); y1 = Math.max(y1, dy); });
      });
      if (x0 > x1) return;
      // room for a star's glow each side (labels slide inside the clip), and — when the subject's stars carry labels —
      // a label above and a label with its sub below
      const lab = sc.d.showLabels && idx.some(k => nodes[k] && nodes[k].label), b = this.fitIn(i, x0, x1, y0, y1, lab ? 14 : 8, lab ? 30 : 10, lab ? 38 : 10, fb); if (!b) return;
      q.zoom *= b.k; q.at = b.at;
      this._fit[i] = { key, d: sc.d, zoom: q.zoom, at: q.at };
    }

    loop(now) {
      this.raf = 0;
      const o = this.o, sc = o.scene;
      // stops only (scene: null): a page that draws its own view (promise's sky) gets the active stop, the arrival and
      // the rail / reel from the scroll, and reads camera.progress() for its own flight
      if (!sc) { const { lin } = this.progress(), act = clamp(Math.round(lin), 0, this.count - 1);
        if (act !== this.active) { const first = this.active < 0, previous = first ? null : this.active; this.active = act; this.arrived = -1; if (!first && window.VegaAudio) window.VegaAudio.tick();
          dispatchEvent(new CustomEvent('vegastop', { detail: { index: act, previous, count: this.count, source: first ? 'init' : 'scroll' } })); }
        if (Math.abs(lin - act) < HOLD && this.arrived !== act) { this.arrived = act; dispatchEvent(new CustomEvent('vegaarrive', { detail: { index: act } })); }
        if (o.free && o.mask) this.maskTo(this.free(), 1, o.mask, null); // the page's own drawing keeps out from under the text
        return; }
      if (!sc.d) return;
      const rm = RM.matches, dt = Math.min(0.05, this.last ? (now - this.last) / 1000 : 0.016); this.last = now;
      const pr = this.progress(), lin = pr.lin, n = this.count;
      // a reel shot (vega-reel) already eases the scroll, so the flight follows it unheld
      const p = window.VegaReel && window.VegaReel.playing ? lin : pr.p;
      const act = clamp(Math.round(lin), 0, n - 1);
      const i = Math.min(n - 2, Math.floor(p)), f = p - i;
      const target = rm ? this.pose(act) : (n < 2 ? this.pose(0) : this.mix(this.pose(Math.max(0, i)), this.pose(Math.max(0, i) + 1), n < 2 ? 0 : f));
      // a reel shot is a crane move: the camera pulls back mid-shot and pushes in on the stop it lands on
      if (!rm && window.VegaReel && window.VegaReel.playing && n > 1) target.zoom *= 1 - DOLLY * Math.sin(Math.PI * clamp(window.VegaReel.t || 0, 0, 1));

      // ease toward the target (a cut under reduced motion)
      const prev = this.cur;
      if (!prev || rm) this.cur = target;
      else {
        const k = 1 - Math.exp(-dt * 9), c = this.cur; /* G8: snappier (was 6.5) */
        c.yaw += (target.yaw - c.yaw) * k; c.pitch += (target.pitch - c.pitch) * k; c.zoom += (target.zoom - c.zoom) * k; c.opacity += (target.opacity - c.opacity) * k; c.nebula += (target.nebula - c.nebula) * k; c.cross += (target.cross - c.cross) * k; c.text += (target.text - c.text) * k;
        c.focus = c.focus.map((v, j) => v + (target.focus[j] - v) * k); c.at = c.at.map((v, j) => v + (target.at[j] - v) * k);
      }
      const c = this.cur;
      // layout reads before any write this frame (the free area and the scenes' boxes), so nothing forces a flush
      const fNow = o.free ? this.free() : null, boxes = o.free ? [sc, o.alt].map(el => el && el.getBoundingClientRect()) : null;

      // a drag or double-click on the scene becomes an offset that settles back after a few seconds
      if (this.lastApplied) { this.user.yaw += sc.yaw - this.lastApplied.yaw; this.user.pitch += sc.pitch - this.lastApplied.pitch; }
      if (!rm && now > (sc.idle || 0)) { const dk = Math.exp(-dt * 1.2); this.user.yaw *= dk; this.user.pitch *= dk; }
      const drift = rm ? 0 : Math.sin(((now - this.t0) / 1000) * SWAY_W) * SWAY;
      const yaw = c.yaw + drift + this.user.yaw, pitch = clamp(c.pitch + this.user.pitch, -1.25, 1.25);
      const dv = this.diveT0 ? clamp((now - this.diveT0) / this.diveMs, 0, 1) : 0, zoom = c.zoom * (1 + 2.4 * dv * dv);
      sc.yaw = yaw; sc.pitch = pitch; sc.d.zoom = zoom; sc.tYaw = null; sc.tPitch = null; sc.vy = 0;
      const q = this.project(sc, c.focus, yaw, pitch, zoom);
      sc.ox = c.at[0] - q[0]; sc.oy = c.at[1] - q[1];
      sc.style.opacity = (c.opacity * (1 - c.cross)).toFixed(3); sc.d.nebula = c.nebula < 0.02 ? 0 : c.nebula;
      if (o.alt) { // a scene's own display value is kept (vega-constellation sets display:block on itself)
        // a hidden scene also stops taking pointers in the wrappers around it (its container and the x-import's host),
        // or the scene stacked on top would swallow every drag and pick meant for the visible one
        const show = (el, on) => { if (el._camDisp == null) el._camDisp = el.style.display || 'block'; el.style.display = on ? el._camDisp : 'none';
          for (let n = el.parentElement; n && !n.hasAttribute('data-stage'); n = n.parentElement) n.style.pointerEvents = on ? '' : 'none'; };
        o.alt.style.opacity = c.cross.toFixed(3); show(o.alt, c.cross >= 0.01); show(sc, c.cross <= 0.99);
      }
      this.lastApplied = { yaw, pitch };

      // the scenes draw only in the free stage area (soft edges); their canvas text gets the same area (clip).
      // mask layers, composited bottom-up: side rows ∩ side column = the side blocks' rectangles; the rail's cut minus
      // those; ∩ the full-width bands
      if (o.free) this.maskTo(fNow, c.opacity * (1 - c.cross), [sc, o.alt], boxes);
      // canvas text: the stop's share, and none while the loader gate holds the page
      const g = document.documentElement.classList.contains('vl-holding') ? 0 : 1;
      this._gate = rm || this._gate == null ? g : this._gate + (g - this._gate) * (1 - Math.exp(-dt * 4));
      sc.textAlpha = c.text * this._gate; if (o.alt) o.alt.textAlpha = this._gate;
      // sky parallax: the far field moves a little against the scroll
      if (o.sky) { const travel = rm ? 0 : innerHeight * 0.1; o.sky.style.transform = 'translate3d(0,' + (-(n > 1 ? p / (n - 1) - 0.5 : 0) * travel).toFixed(1) + 'px,0)'; }

      // camera speed → the drone (spec §2.7)
      if (prev && window.VegaAudio) {
        const sp = (Math.abs(target.yaw - c.yaw) + Math.abs(target.zoom - c.zoom) * 3 + Math.abs(target.pitch - c.pitch)) * 2.5;
        if (!this._mt || now - this._mt > 120) { this._mt = now; window.VegaAudio.motion(clamp(sp, 0, 1)); }
      }

      if (act !== this.active) {
        const first = this.active < 0, previous = first ? null : this.active; this.active = act; this.arrived = -1;
        if (rm) this.user = { yaw: 0, pitch: 0 };
        const s = o.stops[act]; if (sc.setSelected) sc.setSelected(s.select == null ? -1 : s.select, false);
        if (!first && window.VegaAudio) window.VegaAudio.tick();
        dispatchEvent(new CustomEvent('vegastop', { detail: { index: act, previous, count: n, source: first ? 'init' : 'scroll' } }));
      }
      // arrival: the scroll sits at the stop (inside its hold) and the eased pose has caught up
      const near = Math.abs(lin - act) < HOLD && (rm || Math.abs(target.zoom - c.zoom) + Math.abs(target.yaw - c.yaw) + Math.abs(target.at[1] - c.at[1]) / 200 < 0.03);
      if (near && this.arrived !== act) { this.arrived = act; dispatchEvent(new CustomEvent('vegaarrive', { detail: { index: act } })); }
      // keep running while the drift or an easing is live; under reduced motion only on scroll
      const settling = prev && !rm && (Math.abs(target.cross - c.cross) + Math.abs(target.zoom - c.zoom) + Math.abs(target.yaw - c.yaw) + Math.abs(target.at[1] - c.at[1]) / 100 > 1e-4);
      if (!rm || settling || this.diveT0 || Math.abs(this.user.yaw) + Math.abs(this.user.pitch) > 1e-4) this.kick();
    }
  }
  customElements.define('vega-camera', VegaCamera);
})();
