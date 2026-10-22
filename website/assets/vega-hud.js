/* vega-hud — the HUD frame and nav (spec §2.2). A fixed 1px frame in --line with a chamfered top-left
   corner, pointer-events: none except on its controls. Top-left: the star mark (link to the threshold)
   and V2.3.0. Top-right: the seven page links (Martian Mono, scramble on hover/focus, current in --fg),
   INDEX. Bottom-left: SOUND ON / OFF, FLY (the spacecraft, vega-flight.js; key F). The links fold into INDEX when they don't fit (always at 390).
   Attributes: page — the current page's file stem (index, chain, …).
   Events out: window 'vegaindex' (the page opens its INDEX dialog). Method: focusIndex(). */
(() => {
  if (customElements.get('vega-hud')) return;
  const PAGES = [
    ['index', 'Threshold'], ['chain', 'The chain'], ['data', 'The data lane'], ['promise', 'The rules'],
    ['proof', 'The proof'], ['coverage', 'Coverage & the showcase'], ['install', 'Install']
  ];
  const rm = matchMedia('(prefers-reduced-motion: reduce)');
  const MONO = "'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace";
  const CSS = `
.sc-host-x:has(> vega-hud){pointer-events:none}
vega-hud{position:fixed;inset:0;z-index:9;pointer-events:none;display:block;--in:24px;--ch:18px;font:500 11px/1 ${MONO};letter-spacing:.22em;color:var(--dim,#8C9EB3)}
@media (max-width:600px){vega-hud{--in:12px;--ch:12px}}
vega-hud::before,vega-hud::after{content:'';position:absolute;left:0;right:0;z-index:-1;pointer-events:none;opacity:0;transition:opacity .45s ease}
/* G7: the bands only darken the scene while text runs under the nav or the bottom row; otherwise the scene runs full-bleed */
vega-hud.vh-tb::before,vega-hud.vh-bb::after{opacity:1}
vega-hud::before{top:0;height:calc(var(--in) + 92px);background:linear-gradient(var(--bg,#04080E) 58%,rgba(4,8,14,0))} /* opaque through the nav row: text scrolling up goes under it */
vega-hud::after{bottom:0;height:calc(var(--in) + 104px);background:linear-gradient(rgba(4,8,14,0),var(--bg,#04080E) 42%)} /* opaque through the SOUND / FLY / counter row: text scrolling down goes under it (as the nav row above) */
@media (max-width:600px){vega-hud::after{height:calc(var(--in) + 112px)}}
vega-hud .vh-frame{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
vega-hud .vh-frame path{fill:none;stroke:var(--line,rgba(156,200,255,.18));stroke-width:1}
vega-hud .vh-frame path.vh-edge{stroke:rgba(156,200,255,.055)}
vega-hud a,vega-hud button{pointer-events:auto;appearance:none;background:none;border:0;margin:0;cursor:pointer;color:inherit;font:inherit;letter-spacing:inherit;text-decoration:none;text-transform:uppercase;white-space:nowrap;transition:color .25s}
vega-hud a:hover,vega-hud button:hover,vega-hud a:focus-visible,vega-hud button:focus-visible{color:var(--fg,#E9F0F8)}
vega-hud :focus-visible{outline:1.5px solid var(--acc,#9CC8FF);outline-offset:5px;border-radius:2px}
vega-hud .vh-top{position:absolute;top:calc(var(--in) + 12px);left:calc(var(--in) + var(--ch) + 6px);right:calc(var(--in) + 14px);display:flex;align-items:center;justify-content:space-between;gap:24px}
vega-hud .vh-tl{display:flex;align-items:center;gap:16px;flex:none}
vega-hud .vh-brand{display:block;padding:4px;color:var(--fg,#E9F0F8)}
vega-hud .vh-brand svg{display:block;width:22px;height:22px;overflow:visible;filter:drop-shadow(0 0 8px rgba(156,200,255,.3))}
vega-hud .vh-ver{letter-spacing:.22em}
vega-hud nav{display:flex;align-items:center;gap:20px;min-width:0}
vega-hud .vh-links{display:flex;align-items:center;gap:20px;margin:0;padding:0;list-style:none}
vega-hud .vh-links a{display:block;padding:8px 2px}
vega-hud .vh-links a[aria-current="page"]{color:var(--fg,#E9F0F8)}
vega-hud .vh-sep{width:1px;height:16px;background:var(--line,rgba(156,200,255,.18));flex:none}
vega-hud .vh-index{display:flex;align-items:center;gap:10px;padding:8px 4px}
vega-hud .vh-burger{display:flex;flex-direction:column;gap:3px}
vega-hud .vh-burger i{display:block;width:12px;height:1px;background:currentColor}
vega-hud .vh-burger i:last-child{width:8px}
vega-hud .vh-bl{position:absolute;left:calc(var(--in) + 12px);bottom:calc(var(--in) + 10px)}
vega-hud .vh-sound{display:flex;align-items:center;gap:10px;padding:8px 4px}
vega-hud .vh-bl{display:flex;align-items:center;gap:22px}
vega-hud .vh-fly{display:flex;align-items:center;gap:10px;padding:8px 4px}
vega-hud .vh-fly svg{width:12px;height:12px;display:block;overflow:visible}
vega-hud .vh-fly[aria-pressed="true"]{color:var(--fg,#E9F0F8)}
vega-hud .vh-eq{display:flex;align-items:flex-end;gap:2px;height:10px}
vega-hud .vh-eq i{display:block;width:2px;height:10px;background:currentColor;transform-origin:bottom;transform:scaleY(.2);transition:transform .3s}
vega-hud .vh-sound[aria-pressed="true"]{color:var(--fg,#E9F0F8)}
vega-hud .vh-sound[aria-pressed="true"] .vh-eq i{animation:vh-eq 1.1s ease-in-out infinite alternate}
vega-hud .vh-sound[aria-pressed="true"] .vh-eq i:nth-child(2){animation-delay:-.4s}
vega-hud .vh-sound[aria-pressed="true"] .vh-eq i:nth-child(3){animation-delay:-.75s}
vega-hud .vh-sound[aria-pressed="true"] .vh-eq i:nth-child(4){animation-delay:-.2s}
@keyframes vh-eq{0%{transform:scaleY(.25)}100%{transform:scaleY(1)}}
vega-hud.vh-fold .vh-links,vega-hud.vh-fold .vh-sep{display:none}
/* G7 one navigation: the page links are the menu; INDEX (the same pages as a dialog) only stands in for them when they fold */
vega-hud:not(.vh-fold) .vh-index,vega-hud:not(.vh-fold) .vh-sep{display:none}
[data-fighud].vh-clash{opacity:0!important;transition:opacity .25s ease!important}
/* G8: a page's calm opening (html.vg-calm, index's hero): no frame, no chapter counter, the page links folded into INDEX */
vega-hud .vh-frame{transition:opacity .4s ease}
html.vg-calm vega-hud .vh-frame{opacity:0}
html.vg-calm vega-hud .vh-links,html.vg-calm vega-hud .vh-sep{display:none}
html.vg-calm vega-hud:not(.vh-fold) .vh-index{display:flex}
vega-hud .vh-links{animation:vh-in .35s ease both}@keyframes vh-in{from{opacity:0}to{opacity:1}}
html.vg-calm vega-rail{opacity:0!important;visibility:hidden!important;pointer-events:none!important;transition:opacity .35s ease,visibility 0s linear .35s} /* hidden, not just transparent: out of the tab order */
vega-rail{transition:opacity .35s ease,visibility 0s}
@media (prefers-reduced-motion: reduce){vega-hud .vh-sound[aria-pressed="true"] .vh-eq i{animation:none;transform:scaleY(.7)}vega-hud *{transition:none!important}}
`;
  const MARK = '<svg viewBox="0 0 64 64" aria-hidden="true"><polygon fill="currentColor" points="32,5 33.5,32 32,59 30.5,32"/><polygon fill="currentColor" points="9,32 32,30.6 55,32 32,33.4"/><g stroke="var(--acc,#9CC8FF)" stroke-width="1.4"><line x1="27.8" y1="27.8" x2="18" y2="18"/><line x1="36.2" y1="27.8" x2="46" y2="18"/><line x1="27.8" y1="36.2" x2="18" y2="46"/><line x1="36.2" y1="36.2" x2="46" y2="46"/></g><g fill="var(--acc,#9CC8FF)"><circle cx="18" cy="18" r="2.4"/><circle cx="46" cy="18" r="2.4"/><circle cx="18" cy="46" r="2.4"/><circle cx="46" cy="46" r="2.4"/></g><g fill="currentColor"><circle cx="32" cy="5" r="3.2"/><circle cx="32" cy="59" r="3.2"/><circle cx="9" cy="32" r="3.2"/><circle cx="55" cy="32" r="3.2"/></g><circle cx="32" cy="32" r="6" fill="currentColor"/></svg>';

  const el = (tag, cls, attrs) => { const e = document.createElement(tag); if (cls) e.className = cls; if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  class VegaHud extends HTMLElement {
    connectedCallback() {
      if (this._built) return;
      this._built = true;
      if (!document.getElementById('vega-hud-css')) { const s = el('style'); s.id = 'vega-hud-css'; s.textContent = CSS; document.head.appendChild(s); }
      const page = this.getAttribute('page') || 'index';

      this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      this.svg.setAttribute('class', 'vh-frame'); this.svg.setAttribute('aria-hidden', 'true');
      this.path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      this.edge = document.createElementNS('http://www.w3.org/2000/svg', 'path'); this.edge.setAttribute('class', 'vh-edge');
      this.svg.append(this.edge, this.path);

      const top = el('div', 'vh-top'), tl = el('div', 'vh-tl');
      const brand = el('a', 'vh-brand', { href: 'index.dc.html', 'aria-label': 'Vega — threshold' }); brand.innerHTML = MARK;
      const ver = el('span', 'vh-ver'); ver.textContent = 'V2.3.0';
      tl.append(brand, ver);

      const nav = el('nav', '', { 'aria-label': 'Site' }), ul = el('ul', 'vh-links');
      for (const [stem, label] of PAGES) {
        const li = el('li'), a = el('a', '', { href: stem + '.dc.html', 'aria-label': label }), t = el('span');
        t.textContent = label; a.appendChild(t);
        if (stem === page) a.setAttribute('aria-current', 'page');
        li.appendChild(a); ul.appendChild(li);
      }
      const idx = el('button', 'vh-index', { type: 'button', 'aria-haspopup': 'dialog', 'aria-label': 'INDEX' });
      idx.innerHTML = '<span class="vh-burger" aria-hidden="true"><i></i><i></i><i></i></span>';
      const idxT = el('span'); idxT.textContent = 'INDEX'; idx.appendChild(idxT);
      idx.addEventListener('click', () => dispatchEvent(new CustomEvent('vegaindex')));
      nav.append(ul, el('span', 'vh-sep', { 'aria-hidden': 'true' }), idx);
      top.append(tl, nav);

      const bl = el('div', 'vh-bl');
      const snd = this.snd = el('button', 'vh-sound', { type: 'button' });
      snd.innerHTML = '<span class="vh-eq" aria-hidden="true"><i></i><i></i><i></i><i></i></span>';
      this.sndT = el('span'); snd.appendChild(this.sndT);
      snd.addEventListener('click', () => { if (window.VegaAudio) window.VegaAudio.toggle(); });
      // FLY: the spacecraft (vega-flight.js, loaded on first use): free flight through the galaxy to VEGA's planets
      const fly = this.fly = el('button', 'vh-fly', { type: 'button', 'aria-label': 'FLY', 'aria-pressed': 'false', 'aria-keyshortcuts': 'F' });
      fly.innerHTML = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 .5 L11 11 L6 8.4 L1 11 Z" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linejoin="round"/></svg>';
      const flyT = el('span'); flyT.textContent = 'FLY'; fly.appendChild(flyT);
      fly.addEventListener('click', () => this.toggleFlight());
      bl.append(snd, fly);
      this.onFlight = (e) => fly.setAttribute('aria-pressed', String(!!e.detail.on));
      addEventListener('vegaflight', this.onFlight);
      this.onKeyF = (e) => { if ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey && !e.altKey && !e.defaultPrevented && !(window.VegaLoader && window.VegaLoader.active)) {
        const t = e.target; if (t && t.closest && t.closest('input, textarea, select, [contenteditable="true"], [role="dialog"]:not(.vfl)')) return; e.preventDefault(); this.toggleFlight(); } };
      addEventListener('keydown', this.onKeyF);

      // inner pages keep their own header brand link (VEGA / <PAGE>) in the HUD's top-left: brand="page"
      if (this.getAttribute('brand') === 'page') tl.style.visibility = 'hidden';
      this.append(this.svg, top, bl);
      this.idx = idx; this.nav = nav; this.tl = tl;
      this.syncSound(false);
      this.onSound = () => this.syncSound(true);
      addEventListener('vegasound', this.onSound);

      // hover / focus: scramble the label, a very quiet click
      for (const c of this.querySelectorAll('a,button')) {
        const t = c.classList.contains('vh-brand') ? null : c.querySelector('span:last-child');
        const go = () => { if (t && t.dataset.final && window.VegaScramble) window.VegaScramble.run(t, t.dataset.final, { duration: 380 }); if (window.VegaAudio) window.VegaAudio.click(); };
        c.addEventListener('pointerenter', go);
        c.addEventListener('focus', go);
      }
      for (const t of this.querySelectorAll('.vh-links span, .vh-index span:last-child, .vh-fly span')) t.dataset.final = t.textContent;

      // a fixed readout block (data's corner HUD, chain's on phones) steps aside while a text block scrolls across it, so
      // the two never print over each other between stops (verifier G4 r5); readouts pinned to stars (.vro) and promise's
      // guarantees, which sit under their own block, are not affected
      this.onClash = () => { if (this._cr) return; this._cr = requestAnimationFrame(() => { this._cr = 0;
        const vps = [...document.querySelectorAll('.vp > *')].map(v => v.getBoundingClientRect()).filter(b => b.width && b.bottom > 0 && b.top < innerHeight); // the block's content, not its box
        const inn = innerWidth <= 600 ? 12 : 24, H = innerHeight, vis = (v) => { const o = getComputedStyle(v.parentElement.closest('.vp') || v).opacity; return +o > 0.05; };
        const els = [...document.querySelectorAll('.vp > *')].filter(v => { const b = v.getBoundingClientRect(); return b.width && b.bottom > 0 && b.top < H; });
        this.classList.toggle('vh-tb', els.some(v => v.getBoundingClientRect().top < inn + 92 && vis(v)));
        this.classList.toggle('vh-bb', els.some(v => v.getBoundingClientRect().bottom > H - inn - (innerWidth <= 600 ? 112 : 104) && vis(v)));
        const fx = [...document.querySelectorAll('[data-fighud]:not(.vro):not([data-gcards])')]; if (!fx.length) return;
        fx.forEach(el => { const r = el.getBoundingClientRect(); if (!r.width) return;
          el.classList.toggle('vh-clash', vps.some(b => b.left < r.right + 4 && b.right > r.left - 4 && b.top < r.bottom + 4 && b.bottom > r.top - 4)); }); }); };
      addEventListener('scroll', this.onClash, { passive: true }); addEventListener('resize', this.onClash); setTimeout(this.onClash, 800);
      // a reel shot's scroll can end before its arriving text has faded in: check again when the shot ends and on arrival
      this.onReelEnd = (e) => { if (e.detail && e.detail.state === 'end') { this.onClash(); setTimeout(this.onClash, 150); } };
      addEventListener('vegareel', this.onReelEnd); addEventListener('vegaarrive', this.onClash);
      this.ro = new ResizeObserver(() => this.layout());
      this.ro.observe(this);
      this.layout();
    }
    disconnectedCallback() { removeEventListener('vegareel', this.onReelEnd); removeEventListener('vegaarrive', this.onClash); removeEventListener('scroll', this.onClash); removeEventListener('resize', this.onClash); this.ro && this.ro.disconnect(); removeEventListener('vegasound', this.onSound); removeEventListener('vegaflight', this.onFlight); removeEventListener('keydown', this.onKeyF); }

    syncSound(animate) {
      const on = !!(window.VegaAudio && window.VegaAudio.on), text = on ? 'SOUND ON' : 'SOUND OFF';
      this.snd.setAttribute('aria-pressed', String(on)); this.snd.setAttribute('aria-label', text);
      this.sndT.dataset.final = text;
      if (animate && window.VegaScramble) window.VegaScramble.run(this.sndT, text, { duration: 380 }); else this.sndT.textContent = text;
    }

    layout() {
      const W = this.clientWidth, H = this.clientHeight; if (!W || !H) return;
      const cs = getComputedStyle(this), inn = parseFloat(cs.getPropertyValue('--in')) || 24, ch = parseFloat(cs.getPropertyValue('--ch')) || 18;
      const x0 = inn + 0.5, y0 = inn + 0.5, x1 = W - inn - 0.5, y1 = H - inn - 0.5;
      this.svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      // G7: a viewfinder — corner marks in --line (the chamfer stays top-left), the edges between them barely there
      const k = Math.min(34, (x1 - x0) / 6);
      this.path.setAttribute('d', 'M' + x0 + ',' + (y0 + ch + k) + 'V' + (y0 + ch) + 'L' + (x0 + ch) + ',' + y0 + 'H' + (x0 + ch + k)
        + 'M' + (x1 - k) + ',' + y0 + 'H' + x1 + 'V' + (y0 + k) + 'M' + x1 + ',' + (y1 - k) + 'V' + y1 + 'H' + (x1 - k) + 'M' + (x0 + k) + ',' + y1 + 'H' + x0 + 'V' + (y1 - k));
      this.edge.setAttribute('d', 'M' + (x0 + ch) + ',' + y0 + 'H' + x1 + 'V' + y1 + 'H' + x0 + 'V' + (y0 + ch) + 'Z');
      // fold the page links into INDEX when they would crowd the brand
      this.classList.remove('vh-fold');
      const fold = W <= 600 || this.nav.getBoundingClientRect().left < this.tl.getBoundingClientRect().right + 32;
      this.classList.toggle('vh-fold', fold);
    }

    focusIndex() { this.idx && this.idx.focus(); }
    focusFly() { this.fly && this.fly.focus(); }
    toggleFlight() {
      if (window.VegaFlight) { window.VegaFlight.toggle(); return; }
      if (this._flyLoading) return; this._flyLoading = true;
      const sc = document.createElement('script'); sc.src = window.VegaFly && window.VegaFly.asset ? window.VegaFly.asset('vega-flight.js') : '../assets/vega-flight.js';
      sc.onload = () => { this._flyLoading = false; if (window.VegaFlight) window.VegaFlight.toggle(); }; sc.onerror = () => { this._flyLoading = false; };
      document.head.appendChild(sc);
    }
  }
  customElements.define('vega-hud', VegaHud);
})();
