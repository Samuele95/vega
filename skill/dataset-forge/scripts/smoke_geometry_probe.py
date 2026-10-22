"""
smoke_geometry_probe.py — the probe `smoke_geometry.py` appends to a copy of the layered HTML:
it drives the Geometry tab the way an analyst would (tab, PCA toggle, hover, brush, a node's
card, "make this the basis member" and back, a heatmap cell, a candidate card) and writes what it
saw, as JSON, into a <pre id="geometry-smoke"> the DOM dump carries back. `__TARGET__` is
replaced by {"derived": <a rule head, in a cycle when there is one>, "candidate": <a label>}.

Imported by `smoke_geometry.py`; not a command.
"""

PROBE = r"""
<script id="geometry-smoke-probe">
(function(){
  var R = { errors: [], c: {} }, T = __TARGET__;
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' '));
    if (origErr) origErr.apply(console, arguments); };
  function q(s, r){ return (r || document).querySelector(s); }
  function qa(s, r){ return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function mouse(node, type, x, y){
    node.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, view: window }));
  }
  function nodeEl(name){ return qa('[data-view="graph"] g.g-node').filter(function(g){ return g.getAttribute('data-node') === name; })[0]; }
  function run(){
    var c = R.c;
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      var btn = nav && nav.querySelector('button[data-tab="geometry"]');
      c.tab_button = !!btn;
      var pane = q('section.tab-pane[data-layer="geometry"]');
      c.pane = !!pane;
      c.hidden_by_default = !!pane && !pane.classList.contains('active');
      if (btn) btn.click();
      c.active_after_click = !!pane && pane.classList.contains('active');
      c.views = {};
      ['space','graph','orthogonality','partition'].forEach(function(v){ c.views[v] = !!(pane && pane.querySelector('[data-view="' + v + '"]')); });
      c.hook = !!(window.__geometry && typeof window.__geometry.basis === 'function');
      c.empty_space = qa('[data-view="space"] .g-empty').length;
      c.empty_ortho = qa('[data-view="orthogonality"] .g-empty').length;
      /* ---- View A */
      c.scree_bars = qa('.g-scree-svg rect.g-bar').length;
      c.points = qa('[data-view="space"] circle.g-pt').length;
      var pcaBtn = q('[data-view="space"] .g-pca');
      if (pcaBtn) {
        pcaBtn.click();
        var xl = q('[data-view="space"] .g-xlabel');
        c.pca_label = xl ? xl.textContent : null;
        c.pca_loadings = !!q('[data-panel="loadings"]');
        c.pca_points = qa('[data-view="space"] circle.g-pt').length;
        var back = q('[data-view="space"] .g-pca'); if (back) back.click();
        var xl2 = q('[data-view="space"] .g-xlabel');
        var xs = q('[data-view="space"] select[data-axis="x"]');
        c.named_back = !!(xl2 && xs && xl2.textContent.indexOf(xs.value) === 0 && !q('[data-panel="loadings"]'));
      }
      var pt = q('[data-view="space"] circle.g-pt');
      if (pt) {
        mouse(pt, 'mousemove', 0, 0);
        var rp = q('[data-view="space"] .g-rowpanel');
        c.row_values = rp ? qa('dt', rp).length : 0;
        c.row_rules = rp ? qa('.g-rule', rp).length : 0;
      }
      var plot = q('[data-view="space"] .g-plot');
      if (plot) {
        var b = plot.getBoundingClientRect();
        var svg = plot.ownerSVGElement;
        mouse(svg, 'mousedown', b.left + b.width * 0.15, b.top + b.height * 0.15);
        mouse(svg, 'mousemove', b.left + b.width * 0.6, b.top + b.height * 0.6);
        mouse(svg, 'mouseup', b.left + b.width * 0.6, b.top + b.height * 0.6);
        var st = window.__geometry.state();
        c.brushed = st.selectedRows ? Object.keys(st.selectedRows).length : 0;
        var ban = q('.layer-geometry .g-banner');
        c.banner = !!(ban && ban.style.display !== 'none' && ban.textContent.length > 10);
        c.dimmed = qa('[data-view="space"] circle.g-pt-dim').length;
        var clr = q('.layer-geometry .g-banner .g-clear'); if (clr) clr.click();
        c.cleared = !window.__geometry.state().selectedRows;
      }
      /* ---- View B */
      c.nodes = qa('[data-view="graph"] g.g-node').length;
      c.basis_nodes = qa('[data-view="graph"] g.g-node.g-basis').length;
      c.edges = qa('[data-view="graph"] path.g-edge').length;
      c.fd_edges = qa('[data-view="graph"] path.g-fd').length;
      c.cycles = qa('[data-view="graph"] path.g-cycle').length;
      c.basis_before = window.__geometry.basis();
      c.others_before = q('[data-view="partition"]').getAttribute('data-others');
      if (T.derived) {
        var n = nodeEl(T.derived);
        if (n) {
          n.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          var cd = q('[data-view="graph"] .g-card[data-card="' + T.derived + '"]');
          c.card = !!cd;
          c.card_prov = cd ? qa('.g-prov .g-chip', cd).length : 0;
          var mk = cd ? q('.g-make-basis', cd) : null;
          c.make_basis = !!mk;
          if (mk) {
            c.orientation = +mk.getAttribute('data-orientation');
            c.orient_cycle = mk.getAttribute('data-cycle');
            mk.click();
            c.basis_after = window.__geometry.basis();
            var line = q('[data-view="graph"] .g-basis-line');
            c.basis_line_after = line ? line.getAttribute('data-basis') : null;
            var cons = q('[data-view="graph"] .g-orient-cycle[data-cycle="' + c.orient_cycle + '"] .g-orient-cons .g-decision');
            c.decision_after = cons ? cons.textContent : null;
            c.others_after = q('[data-view="partition"]').getAttribute('data-others');
            c.changed_chip = !!q('[data-view="graph"] .g-basis-line .g-changed');
            var rs = q('[data-view="graph"] .g-restore');
            if (rs) rs.click();
            c.basis_restored = window.__geometry.basis();
            c.others_restored = q('[data-view="partition"]').getAttribute('data-others');
          }
        }
      }
      /* ---- View C */
      c.cells = qa('[data-view="orthogonality"] g.g-cell').length;
      var cell = q('[data-view="orthogonality"] g.g-cell');
      if (cell) {
        c.cell_pair = cell.getAttribute('data-pair');
        cell.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        var sx = q('[data-view="space"] select[data-axis="x"]'), sy = q('[data-view="space"] select[data-axis="y"]');
        c.axes_after_cell = sx && sy ? sx.value + '|' + sy.value : null;
        c.pair_panel = !!q('[data-panel="pair"] h4');
      }
      /* ---- View D */
      c.cards = qa('[data-view="partition"] [data-candidate]').length;
      if (T.candidate) {
        var sel = function(){ return qa('[data-view="partition"] [data-candidate]').filter(function(a){
          return a.getAttribute('data-candidate') === T.candidate; })[0]; };
        var cardEl = sel();
        if (cardEl && cardEl.classList.contains('g-on')) { q('.g-cand-btn', cardEl).click(); cardEl = sel(); }
        if (cardEl) {
          q('.g-cand-btn', cardEl).click();
          cardEl = sel();
          c.cand_on = cardEl.classList.contains('g-on');
          c.leak_nodes = qa('[data-view="graph"] g.g-node.g-leak').map(function(g){ return g.getAttribute('data-node'); }).sort();
          c.feature_nodes = qa('[data-view="graph"] g.g-node.g-feature').map(function(g){ return g.getAttribute('data-node'); }).sort();
          c.label_nodes = qa('[data-view="graph"] g.g-node.g-label').map(function(g){ return g.getAttribute('data-node'); });
          var dim = q('[data-input-dim]', cardEl);
          c.input_dim = dim ? +dim.getAttribute('data-input-dim') : null;
          var col = q('[data-view="space"] select[data-colour]');
          c.colour_after = col ? col.value : null;
        }
      }
    } catch (e) { R.errors.push('probe: ' + (e && e.stack || e)); }
    var pre = document.createElement('pre'); pre.id = 'geometry-smoke';
    pre.textContent = JSON.stringify(R); document.body.appendChild(pre);
  }
  function go(){ setTimeout(run, 300); }
  if (document.readyState === 'complete') go(); else window.addEventListener('load', go);
})();
</script>
"""
