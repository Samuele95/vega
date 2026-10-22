/* vega-rail — the chapter counter (G7: one navigation; it replaces the chapter list of spec §2.4). Bottom-right inside
   the HUD frame: `02 / 06`, the chapter's name and a thin progress line, with previous / next buttons. The page links
   live in the HUD only. Stops come from the page: [data-stop] in order, [data-rail] the chapter name, [data-rail-full]
   its full text for aria-label; a [data-stop][data-sub] element is a camera stop that belongs to the chapter before it.
   Listens to window 'vegastop' from vega-camera. It always carries `vr-compact`, which pages and the camera read as
   "the rail takes no column of the stage". API: go(i) — play to chapter i (the reel's shot; scroll otherwise). */
(() => {
  if (customElements.get('vega-rail')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const MONO = "'Martian Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace";
  const CSS = `
.sc-host-x:has(> vega-rail){pointer-events:none}
vega-rail{position:fixed;z-index:9;pointer-events:none;--in:24px;right:calc(var(--in) + 16px);bottom:calc(var(--in) + 12px);font:500 11px/1 ${MONO};letter-spacing:.22em;color:var(--dim,#8C9EB3);text-transform:uppercase}
@media (max-width:600px){vega-rail{--in:12px}}
vega-rail nav{display:flex;align-items:center;gap:14px}
vega-rail button{pointer-events:auto;appearance:none;background:none;border:0;margin:0;padding:8px 4px;font:inherit;letter-spacing:inherit;color:inherit;cursor:pointer;line-height:0}
vega-rail button:hover,vega-rail button:focus-visible{color:var(--fg,#E9F0F8)}
vega-rail button[disabled]{opacity:.25;cursor:default}
vega-rail button i{display:block;width:6px;height:6px;border-left:1px solid currentColor;border-top:1px solid currentColor}
vega-rail .vr-up i{transform:rotate(45deg) translate(1px,1px)}vega-rail .vr-dn i{transform:rotate(225deg) translate(1px,1px)}
vega-rail .vr-now{display:flex;flex-direction:column;align-items:flex-end;gap:7px;min-width:0}
vega-rail .vr-n{color:var(--fg,#E9F0F8)}vega-rail .vr-n b{font-weight:500;color:var(--dim,#8C9EB3)}
vega-rail .vr-name{max-width:28vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim,#8C9EB3);font-size:10px;letter-spacing:.18em}
vega-rail .vr-bar{position:relative;width:96px;height:1px;background:var(--line,rgba(156,200,255,.18))}
vega-rail .vr-bar span{position:absolute;left:0;top:0;bottom:0;background:var(--acc,#9CC8FF);transition:width .6s cubic-bezier(.2,.8,.2,1)}
vega-rail :focus-visible{outline:1.5px solid var(--acc,#9CC8FF);outline-offset:4px;border-radius:2px}
@media (max-width:600px){vega-rail .vr-name{display:none}vega-rail .vr-bar{width:64px}}
@media (prefers-reduced-motion: reduce){vega-rail *{transition:none!important}}
`;
  class VegaRail extends HTMLElement {
    connectedCallback() {
      if (this._built) return;
      this._built = true;
      if (!document.getElementById('vega-rail-css')) { const s = document.createElement('style'); s.id = 'vega-rail-css'; s.textContent = CSS; document.head.appendChild(s); }
      this.onStop = (e) => this.mark(e.detail.index);
      addEventListener('vegastop', this.onStop);
      this.tryBuild(0);
    }
    disconnectedCallback() { removeEventListener('vegastop', this.onStop); this.mo && this.mo.disconnect(); }

    tryBuild(n) {
      const all = [...document.querySelectorAll('[data-stop]')].sort((a, b) => +a.dataset.stop - +b.dataset.stop), stops = all.filter(s => !s.hasAttribute('data-sub'));
      if (!stops.length || stops.some(s => !s.dataset.rail)) { if (n < 100) setTimeout(() => this.tryBuild(n + 1), 50); return; }
      this.stops = stops; this.all = all;
      // camera stop index → rail entry index (a sub-stop belongs to the last full stop before it)
      let e = -1; this.entryOf = all.map(s => { if (!s.hasAttribute('data-sub')) e++; return Math.max(0, e); });
      this.classList.add('vr-compact'); // no column of the stage (pages and the camera read this)
      const nav = document.createElement('nav'); nav.setAttribute('aria-label', this.getAttribute('label') || 'Chapters');
      const btn = (cls, label) => { const b = document.createElement('button'); b.type = 'button'; b.className = cls; b.setAttribute('aria-label', label); b.innerHTML = '<i aria-hidden="true"></i>'; return b; };
      this.up = btn('vr-up', 'Previous chapter'); this.dn = btn('vr-dn', 'Next chapter');
      this.up.addEventListener('click', () => this.go(Math.max(0, (this.cur || 0) - 1)));
      this.dn.addEventListener('click', () => this.go(Math.min(this.stops.length - 1, (this.cur || 0) + 1)));
      [this.up, this.dn].forEach(b => { b.addEventListener('pointerenter', () => window.VegaAudio && window.VegaAudio.click()); });
      const now = document.createElement('div'); now.className = 'vr-now'; now.setAttribute('aria-live', 'polite');
      this.chT = document.createElement('span'); this.chT.className = 'vr-n'; this.nameT = document.createElement('span'); this.nameT.className = 'vr-name';
      const bar = document.createElement('span'); bar.className = 'vr-bar'; bar.setAttribute('aria-hidden', 'true'); this.barT = document.createElement('span'); bar.append(this.barT);
      now.append(this.chT, this.nameT, bar);
      nav.append(this.up, now, this.dn); this.append(nav);
      this.links = stops; // one per chapter (kept for callers that count them)
      // a chapter's name can follow page state (data's hand-off stop follows the USE IT mode)
      this.mo = new MutationObserver(() => this.mark(this.lastSi || 0));
      stops.forEach(s => this.mo.observe(s, { attributes: true, attributeFilter: ['data-rail', 'data-rail-full'] }));
      const cam = document.querySelector('vega-camera');
      this.mark(cam && cam.active >= 0 ? cam.active : 0);
    }
    layout() {} // (the counter has one form at every width)
    go(i) {
      const s = this.stops[i]; if (!s) return; this.goTo = i;
      // a page in film mode plays the jump as one shot (vega-reel); otherwise a smooth scroll
      const si = +s.dataset.stop;
      if (!(window.VegaReel && s.dataset.stop != null && window.VegaReel.go(si))) {
        const y = s.getBoundingClientRect().top + scrollY;
        scrollTo({ top: y, behavior: RM.matches ? 'auto' : 'smooth' });
      }
    }
    mark(si) {
      if (!this.stops) return; this.lastSi = si;
      const i = this.entryOf ? (this.entryOf[si] != null ? this.entryOf[si] : this.entryOf[this.entryOf.length - 1]) : si;
      this.cur = i; if (i === this.goTo) setTimeout(() => { if (this.cur === i) this.goTo = null; }, 0);
      const n = this.stops.length, pad = (v) => String(v).padStart(2, '0'), s = this.stops[i];
      this.chT.innerHTML = pad(i + 1) + ' <b>/ ' + pad(n) + '</b>';
      this.nameT.textContent = s ? s.dataset.rail : '';
      this.barT.style.width = (n > 1 ? i / (n - 1) * 100 : 100) + '%';
      this.up.disabled = i <= 0; this.dn.disabled = i >= n - 1;
      this.setAttribute('aria-label', 'Chapter ' + pad(i + 1) + ' of ' + pad(n) + (s ? ': ' + (s.dataset.railFull || s.dataset.rail) : ''));
    }
  }
  customElements.define('vega-rail', VegaRail);
})();
