/* dataset-forge — `geometry` layer render script (contract: references/report-contract.md §6).
   Vanilla JS, hand-written inline SVG, no libraries, no network. Reads ONLY #layer-geometry-data
   and mounts one tab (data-layer="geometry") holding the four linked views of the explorer:
     A  Space          — scree, a scatter of the sample on two named basis members (or the two
                         leading PCA components, labelled as rotated), colour by a nominal or a
                         label candidate, hover a row, brush a rectangle to select rows;
     B  Derivations    — columns as nodes, kept rules as edges (body → head), exact functional
                         dependencies dashed, cycles drawn; a node's card; inside a cycle,
                         "make this the basis member" re-orients it and recomputes the basis;
     C  Orthogonality  — the residual dependence among the current basis members (NMI, the
                         common scale), a cell opens the pair as View A's axes;
     D  Partition      — the label candidates; selecting one marks the features that remain,
                         strikes the leakage set, colours View A by the label.
   One state object drives all four. The script never recomputes rank, PCA or dependencies:
   it filters, projects, joins and redraws the numbers geometry.py shipped. The only thing it
   derives is the basis under a re-orientation — the columns that are the head of no active
   rule — which is the definition, not a statistic. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="geometry"]')) return;

  var VERSION = 'dataset-forge/geometry-render@1';
  var NS = 'http://www.w3.org/2000/svg';
  var PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#8a6420'];
  var OTHER = '#9a9280';
  var OUTSIDE = { identity: 1, degenerate: 1, 'foreign-key': 1 };
  var CATEGORICAL = { nominal: 1, boolean: 1, id: 1, text: 1 };
  var STATUS_MARK = { confirmed: 'confirmed', refuted: 'refuted', untested: 'untested' };

  /* ------------------------------------------------------------------ small helpers */
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    setAttrs(n, attrs);
    (kids || []).forEach(function (c) {
      if (c !== null && c !== undefined && c !== false)
        n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  function sv(tag, attrs, kids) {
    var n = document.createElementNS(NS, tag);
    setAttrs(n, attrs);
    (kids || []).forEach(function (c) {
      if (c) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  function setAttrs(n, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') n.textContent = String(v);
      else if (k === 'class') n.setAttribute('class', v);
      else if (k.indexOf('on') === 0 && typeof v === 'function') n[k] = v;
      else n.setAttribute(k, String(v));
    });
  }
  function clear(n) { while (n && n.firstChild) n.removeChild(n.firstChild); }
  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function arr(v) { return Array.isArray(v) ? v : []; }
  function has(a, x) { return a.indexOf(x) >= 0; }
  function uniq(a) { var o = []; a.forEach(function (x) { if (!has(o, x)) o.push(x); }); return o; }
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (!isNum(v)) return String(v);
    if (v === Math.round(v) && Math.abs(v) < 1e15) return String(v);
    var a = Math.abs(v);
    if (a < 1e-3 || a >= 1e6) return v.toExponential(2);
    return String(Math.round(v * 1e4) / 1e4);
  }
  function chip(text, cls, title) {
    return el('span', { class: 'g-chip ' + (cls || ''), text: text, title: title || null });
  }
  function para(text, cls) { return el('p', { class: cls || 'g-prose', text: text }); }
  function empty(text) { return el('p', { class: 'g-empty', text: text }); }
  function keyed(fn) {
    return function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(e); } };
  }
  function button(text, cls, fn, extra) {
    var b = el('button', { type: 'button', class: 'g-btn ' + (cls || ''), text: text });
    setAttrs(b, extra);
    b.onclick = fn;
    return b;
  }
  function details(summary, cls) {
    var d = el('details', { class: 'g-det ' + (cls || '') });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }
  function jitter(i) { return ((i * 9301 + 49297) % 233280) / 233280 - 0.5; }
  function escRe(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
  function mentions(text, col) {
    return new RegExp('(^|[^A-Za-z0-9_])' + escRe(col) + '([^A-Za-z0-9_]|$)').test(String(text));
  }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  /* ------------------------------------------------------------------ host plumbing */
  function switchTab(name) {
    document.querySelectorAll('nav.tabs button[data-tab], nav[role="tablist"] button[data-tab]')
      .forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-tab') === name); });
    document.querySelectorAll('section.tab-pane')
      .forEach(function (p) { p.classList.toggle('active', p.getAttribute('data-tab') === name); });
  }
  function mount() {
    var pane = el('section', { class: 'layer-geometry tab-pane', 'data-layer': 'geometry',
                               'data-tab': 'geometry', id: 'tab-geometry', role: 'tabpanel',
                               'aria-label': 'Geometry' });
    var panes = document.querySelectorAll('section.tab-pane');
    var last = panes.length ? panes[panes.length - 1] : null;
    if (last && last.parentNode) last.parentNode.insertBefore(pane, last.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = document.querySelector('nav.tabs') || document.querySelector('nav[role="tablist"]');
    if (tabs) {
      var b = el('button', { 'data-tab': 'geometry', 'data-layer': 'geometry', role: 'tab',
                             type: 'button', text: 'Geometry' });
      b.onclick = function () { switchTab('geometry'); };
      tabs.appendChild(b);
    }
    return pane;
  }
  function readData() {
    var s = document.getElementById('layer-geometry-data');
    if (!s) return { error: 'missing' };
    try { return { data: JSON.parse(s.textContent) }; }
    catch (e) { return { error: 'unparseable', detail: String(e) }; }
  }

  /* ------------------------------------------------------------------ the model */
  function status(d, ch) {
    var p = isObj(d) && isObj(d.provenance) ? d.provenance[ch] : null;
    return isObj(p) ? p.status : null;
  }
  function kept(d) { return isObj(d) && status(d, 'semantic') === 'confirmed' && d.kept !== false; }

  function buildModel(M) {
    var src = M.source || {};
    var E = isObj(M.explore) ? M.explore : null;
    var typing = {};
    arr(M.typing).forEach(function (t) { if (isObj(t) && t.column) typing[t.column] = t; });
    var script = isObj(M.columns) ? M.columns : {};
    var derivs = arr(M.derivations).filter(isObj);
    var reportBasis = arr((M.basis || {}).members).slice();
    var cols = arr(src.column_names).length ? arr(src.column_names).slice()
      : (Array.isArray(src.columns) ? src.columns.slice() : []);
    function add(c) { if (typeof c === 'string' && !has(cols, c)) cols.push(c); }
    Object.keys(typing).forEach(add);
    reportBasis.forEach(add);
    derivs.forEach(function (d) { add(d.column); arr(d.body).forEach(add); });
    if (E && isObj(E.columns)) Object.keys(E.columns).forEach(add);
    var reportHeads = {};
    derivs.filter(kept).forEach(function (d) { if (!reportHeads[d.column]) reportHeads[d.column] = d; });
    var cycles = arr(M.cycles).filter(function (c) {
      return isObj(c) && c.id && arr(c.members).length > 1 && arr(c.orientations).length > 1;
    });
    cycles.forEach(function (cy) {
      var rep = cy.members.filter(function (m) { return !!reportHeads[m]; }).sort().join('|');
      cy._report = 0;
      cy.orientations.forEach(function (o, i) {
        if (arr(o.derived).slice().sort().join('|') === rep) cy._report = i;
      });
    });
    function stype(c) { return (script[c] || {}).type || (typing[c] || {}).script_type || null; }
    function ftype(c) { return (typing[c] || {}).final_type || stype(c) || 'unknown'; }
    /* The stated role, else the one the analyst's final type implies (the script's type only
       when the analyst typed nothing): the script's types are heuristics. */
    function role(c) {
      if (typing[c] && typing[c].role) return typing[c].role;
      var t = ftype(c);
      if (t === 'id') return 'identity';
      if (t === 'constant') return 'degenerate';
      if (reportHeads[c]) return 'derived';
      if (t === 'text' && !has(reportBasis, c)) return 'text';
      return 'dimension';
    }
    function cycleOf(c) {
      for (var i = 0; i < cycles.length; i++) if (has(cycles[i].members, c)) return cycles[i];
      return null;
    }
    var analysed = cols.filter(function (c) {
      var r = role(c);
      return !OUTSIDE[r] && !(r === 'text' && !has(reportBasis, c)) || !!cycleOf(c);
    });
    var fds = arr(M.functional_dependencies).filter(function (f) {
      return isObj(f) && f.exact && Array.isArray(f.lhs) && f.rhs;
    });
    var posOf = {};
    if (E) arr(E.index).forEach(function (ix, i) { posOf[ix] = i; });
    return { M: M, E: E, src: src, cols: cols, typing: typing, derivs: derivs,
             reportHeads: reportHeads, reportBasis: reportBasis, cycles: cycles, fds: fds,
             stype: stype, ftype: ftype, role: role, cycleOf: cycleOf, analysed: analysed,
             posOf: posOf, P: isObj(M.partitions) ? M.partitions : {},
             deps: isObj(M.dependencies) ? M.dependencies : {} };
  }

  /* The active rules under the chosen orientations, and the basis they leave. */
  function activeRules(G, orient) {
    var heads = {};
    Object.keys(G.reportHeads).forEach(function (h) {
      var d = G.reportHeads[h];
      heads[h] = { column: h, body: arr(d.body), formula: d.formula, rule: d.rule_id, d: d };
    });
    G.cycles.forEach(function (cy) {
      var oi = orient[cy.id];
      if (oi === undefined || oi === cy._report) return;
      var o = cy.orientations[oi];
      cy.members.forEach(function (m) { delete heads[m]; });
      var stay = cy.members.filter(function (m) { return !has(arr(o.derived), m); });
      arr(o.derived).forEach(function (m) {
        var od = arr(o.derivations).concat(arr(o.rules))
          .filter(function (x) { return isObj(x) && x.column === m; })[0];
        heads[m] = { column: m, body: od ? arr(od.body) : stay,
                     formula: od && od.formula ? od.formula : m + ' from ' + stay.join(', '),
                     rule: od && od.rule_id ? od.rule_id : null,
                     orientation: o.id || cy.id + '#' + oi, d: null };
      });
    });
    var basis = G.reportBasis.filter(function (c) { return !heads[c]; });
    G.analysed.forEach(function (c) { if (!heads[c] && !has(basis, c)) basis.push(c); });
    return { heads: heads, basis: basis };
  }

  /* The candidates as the active orientation leaves them. */
  function candidateView(G, act) {
    var reportBasis = G.reportBasis;
    var newlyBasis = act.basis.filter(function (c) { return !has(reportBasis, c); });
    var newlyDerived = reportBasis.filter(function (c) { return !!act.heads[c]; });
    var fdHeads = G.fds.map(function (f) { return f.rhs; });
    var authored = arr(G.P.candidates).filter(isObj).map(function (c) {
      var leak = arr(c.dropped_for_leakage), lab = c.label;
      var feats = arr(c.features).filter(function (f) { return !has(newlyDerived, f); });
      newlyBasis.forEach(function (b) {
        if (b !== lab && !has(leak, b) && !has(feats, b)) feats.push(b);
      });
      var h = act.heads[lab];
      var notes = [];
      if (G.reportHeads[lab] && !h) {
        notes.push(has(fdHeads, lab)
          ? 'under this orientation ' + lab + ' is a basis member; it stays a candidate only through its functional dependency'
          : 'under this orientation ' + lab + ' is a basis member: no rule derives it any more');
      }
      if (h) {
        var bodyKept = h.body.filter(function (b) { return has(feats, b); });
        if (bodyKept.length)
          notes.push('under this orientation the rule for ' + lab + ' reads ' + bodyKept.join(', ') +
                     ', which the leakage set written for the report does not drop');
      }
      return { c: c, label: lab, features: feats, leak: leak, input_dim: feats.length,
               changed: newlyBasis.length > 0 || newlyDerived.length > 0, notes: notes };
    });
    var labels = authored.map(function (x) { return x.label; });
    var others = Object.keys(act.heads).filter(function (h) {
      return !has(labels, h) && act.heads[h].body.every(function (b) { return has(act.basis, b); });
    });
    return { authored: authored, others: others };
  }

  /* ------------------------------------------------------------------ scales */
  function kindOf(G, col) {
    var t = G.ftype(col);
    if (CATEGORICAL[t]) return 'cat';
    if (t === 'datetime') return 'date';
    return 'num';
  }
  function value(G, col, i, kind) {
    var v = G.E && G.E.columns && G.E.columns[col] ? G.E.columns[col][i] : null;
    if (v === null || v === undefined || v === '') return null;
    if (kind === 'date') { var t = Date.parse(v); return isNaN(t) ? null : t; }
    if (kind === 'num') return isNum(v) ? v : (isNum(Number(v)) ? Number(v) : null);
    return String(v);
  }
  function scaleFor(values, kind) {
    if (kind === 'cat') {
      var cats = uniq(values.filter(function (v) { return v !== null; }).map(String)).sort();
      return { kind: 'cat', cats: cats, pos: function (v, i) {
        var k = cats.indexOf(String(v));
        return k < 0 ? null : (k + 0.5 + 0.55 * jitter(i)) / cats.length; } };
    }
    var nums = values.filter(isNum);
    var lo = Math.min.apply(null, nums.length ? nums : [0]);
    var hi = Math.max.apply(null, nums.length ? nums : [1]);
    if (hi === lo) { hi = lo + 1; }
    return { kind: kind, lo: lo, hi: hi, pos: function (v) {
      return isNum(v) ? (v - lo) / (hi - lo) : null; } };
  }
  function tickText(s, which) {
    if (s.kind === 'cat') return which === 0 ? trunc(s.cats[0] || '', 12) : trunc(s.cats[s.cats.length - 1] || '', 12);
    var v = which === 0 ? s.lo : s.hi;
    if (s.kind === 'date') return new Date(v).toISOString().slice(0, 10);
    return fmt(v);
  }

  /* ------------------------------------------------------------------ the page */
  ready(function () {
    var read = readData();
    var pane = mount();
    pane.appendChild(el('header', { class: 'g-head' }, [el('h2', { text: 'Geometry' })]));
    if (read.error) {
      pane.appendChild(empty(read.error === 'missing'
        ? 'This file carries a geometry layer marker but no data script: nothing to show.'
        : 'The geometry layer data did not parse: ' + (read.detail || '')));
      return;
    }
    var M = read.data || {};
    var G = buildModel(M);
    var E = G.E;
    var exploreOk = !!(E && isObj(E.columns) && isNum(E.n) && E.n >= 10);
    var src = G.src;
    pane.firstChild.appendChild(el('p', { class: 'g-sub', text:
      (src.path || 'dataset') + ' — ' + (src.rows !== undefined ? src.rows + ' rows, ' : '') +
      G.cols.length + ' columns' + (exploreOk ? ', explorer sample of ' + E.n + ' rows' : '') +
      (src.seed !== undefined && src.seed !== null ? ', seed ' + src.seed : '') }));

    var state = {
      selectedRows: null, focus: null,
      axes: { x: null, y: null, mode: 'named' },
      colourBy: null, activeOrientation: {}, activeCandidate: null, pair: null
    };
    G.cycles.forEach(function (cy) { state.activeOrientation[cy.id] = cy._report; });
    if (G.P.chosen) state.activeCandidate = G.P.chosen;
    var act = activeRules(G, state.activeOrientation);

    var views = {};
    ['space', 'graph', 'orthogonality', 'partition'].forEach(function (v) {
      views[v] = el('div', { class: 'g-view g-view-' + v, 'data-view': v });
    });
    var banner = el('div', { class: 'g-banner', role: 'status' });
    pane.appendChild(banner);
    pane.appendChild(views.space);
    pane.appendChild(views.graph);
    pane.appendChild(views.orthogonality);
    pane.appendChild(views.partition);

    function axisColumns() {
      if (!exploreOk) return [];
      return act.basis.filter(function (c) { return E.columns[c] !== undefined; });
    }
    function defaultAxes() {
      var ok = axisColumns();
      var pairs = arr((M.orthogonality || {}).pairs).filter(function (p) {
        return isObj(p) && has(ok, p.a) && has(ok, p.b);
      });
      if (pairs.length) return { x: pairs[0].a, y: pairs[0].b };
      var nums = ok.filter(function (c) { return kindOf(G, c) !== 'cat'; });
      var pool = nums.length >= 2 ? nums : ok;
      return { x: pool[0] || null, y: pool[1] || pool[0] || null };
    }
    var ax0 = defaultAxes();
    state.axes.x = ax0.x; state.axes.y = ax0.y;
    if (exploreOk && G.P.chosen && E.columns[G.P.chosen] !== undefined) state.colourBy = G.P.chosen;

    function selectionCount() {
      return state.selectedRows ? Object.keys(state.selectedRows).length : 0;
    }
    function redraw() {
      act = activeRules(G, state.activeOrientation);
      var ok = axisColumns();
      if (state.axes.mode === 'named' && ok.length &&
          (!has(ok, state.axes.x) || !has(ok, state.axes.y))) {
        var d = defaultAxes();
        if (!has(ok, state.axes.x)) state.axes.x = d.x;
        if (!has(ok, state.axes.y)) state.axes.y = d.y === state.axes.x ? d.x : d.y;
      }
      drawBanner();
      drawSpace();
      drawGraph();
      drawOrtho();
      drawPartition();
    }
    function drawBanner() {
      clear(banner);
      var n = selectionCount();
      banner.style.display = n ? '' : 'none';
      if (!n) return;
      banner.appendChild(el('span', { text: n + ' of ' + E.n + ' sample rows selected in the space view; ' +
        'the graph cards and the partition cards read them, the orthogonality measures stay the full-data ones.' }));
      banner.appendChild(button('clear the selection', 'g-clear', function () {
        state.selectedRows = null; redraw(); }));
    }

    /* ================================================================ View A — Space */
    function drawSpace() {
      var V = views.space;
      clear(V);
      V.appendChild(el('h3', { text: 'The space' }));
      var S = M.space || {};
      if (S.reading) V.appendChild(para(S.reading));
      var nums = el('div', { class: 'g-chips' });
      [['ambient', S.ambient_dim], ['exact rank', S.exact_rank], ['near rank', S.near_rank],
       ['95 % variance in', S.dims_95], ['99 % variance in', S.dims_99],
       ['participation ratio', S.participation_ratio], ['TwoNN intrinsic', S.intrinsic_dim_twonn]]
        .forEach(function (p) { if (p[1] !== undefined && p[1] !== null) nums.appendChild(chip(p[0] + ' ' + fmt(p[1]), 'g-num')); });
      V.appendChild(nums);
      if (!exploreOk) {
        V.appendChild(empty('No explorer sample ships with this layer (or it holds fewer than ten rows): ' +
          'the scree and the scatter need the script\'s explore payload. The derivation graph and ' +
          'the partition cards below work without it.'));
        return;
      }
      var row = el('div', { class: 'g-row2' });
      row.appendChild(scree(S));
      var right = el('div', { class: 'g-scatterbox' });
      right.appendChild(controls());
      var sc = scatter();
      right.appendChild(sc.svg);
      right.appendChild(sc.legend);
      row.appendChild(right);
      V.appendChild(row);
      if (state.axes.mode === 'pca') V.appendChild(loadings());
      V.appendChild(sc.rowPanel);
    }
    function scree(S) {
      var box = el('figure', { class: 'g-scree' });
      var sv0 = arr(S.singular_values), ev = arr(S.explained_variance_ratio);
      if (!sv0.length) {
        box.appendChild(empty('No singular values: the numeric space is empty.'));
        return box;
      }
      var W = 220, H = 150, pad = 22, bw = (W - pad - 6) / sv0.length;
      var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'g-svg g-scree-svg', role: 'img',
                            'aria-label': 'scree of ' + sv0.length + ' singular values' });
      var mx = Math.max.apply(null, sv0);
      sv0.forEach(function (v, i) {
        var h = (H - pad - 10) * v / (mx || 1);
        var r = sv('rect', { x: pad + i * bw + 1, y: H - pad - h, width: Math.max(bw - 2, 1), height: h,
                             class: 'g-bar' + (isNum(S.dims_95) && i < S.dims_95 ? ' g-bar-in' : '') });
        r.appendChild(sv('title', { text: 'σ' + (i + 1) + ' = ' + fmt(v) +
          (ev[i] !== undefined ? ' (' + fmt(Math.round(ev[i] * 1000) / 10) + ' % of the variance)' : '') }));
        svg.appendChild(r);
      });
      svg.appendChild(sv('line', { x1: pad, y1: H - pad, x2: W - 4, y2: H - pad, class: 'g-axis' }));
      svg.appendChild(sv('text', { x: pad, y: H - 6, class: 'g-tick', text: 'σ1' }));
      svg.appendChild(sv('text', { x: W - 4, y: H - 6, class: 'g-tick', 'text-anchor': 'end', text: 'σ' + sv0.length }));
      box.appendChild(svg);
      box.appendChild(el('figcaption', { text: 'Scree: singular values of the standardized numeric columns' +
        (isNum(S.dims_95) ? '; the dark bars hold 95 % of the variance (' + S.dims_95 + ')' : '') + '.' }));
      return box;
    }
    function controls() {
      var bar = el('div', { class: 'g-controls' });
      var ok = axisColumns();
      function select(label, value, opts, onchange, extra) {
        var s = el('select', { 'aria-label': label });
        setAttrs(s, extra);
        opts.forEach(function (o) {
          var op = el('option', { value: o[0], text: o[1] });
          if (o[0] === value) op.selected = true;
          s.appendChild(op);
        });
        s.onchange = function () { onchange(s.value); };
        var w = el('label', { class: 'g-sel', text: label + ' ' });
        w.appendChild(s);
        return w;
      }
      var named = state.axes.mode === 'named';
      var axisOpts = ok.map(function (c) { return [c, c]; });
      bar.appendChild(select('x', state.axes.x, axisOpts, function (v) {
        state.axes.x = v; state.axes.mode = 'named'; redraw(); }, { 'data-axis': 'x', disabled: named ? null : 'disabled' }));
      bar.appendChild(select('y', state.axes.y, axisOpts, function (v) {
        state.axes.y = v; state.axes.mode = 'named'; redraw(); }, { 'data-axis': 'y', disabled: named ? null : 'disabled' }));
      var labels = arr(G.P.candidates).filter(isObj).map(function (c) { return c.label; });
      var colOpts = [['', 'nothing']];
      G.cols.forEach(function (c) {
        if (E.columns[c] === undefined) return;
        var isLabel = has(labels, c);
        var cats = uniq(E.columns[c].map(String)).length;
        if (isLabel || (kindOf(G, c) === 'cat' && cats <= 20) || (G.ftype(c) === 'ordinal' && cats <= 20))
          colOpts.push([c, c + (isLabel ? ' (label candidate)' : '')]);
      });
      bar.appendChild(select('colour by', state.colourBy || '', colOpts, function (v) {
        state.colourBy = v || null; redraw(); }, { 'data-colour': '1' }));
      var hasPca = !!(E.pca && arr(E.pca.scores).length);
      if (hasPca) {
        bar.appendChild(button(named ? 'rotate to the PCA axes' : 'back to named axes', 'g-pca', function () {
          state.axes.mode = named ? 'pca' : 'named'; redraw();
        }, { 'aria-pressed': named ? 'false' : 'true' }));
      }
      return bar;
    }
    function colourer() {
      var c = state.colourBy;
      if (!c || !E.columns[c]) return { fill: function () { return null; }, legend: [] };
      var kind = kindOf(G, c) === 'cat' || G.ftype(c) === 'ordinal' ||
        uniq(E.columns[c].map(String)).length <= 8 ? 'cat' : 'num';
      if (kind === 'cat') {
        var counts = {};
        E.columns[c].forEach(function (v) { counts[String(v)] = (counts[String(v)] || 0) + 1; });
        var top = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); })
          .slice(0, PALETTE.length);
        return { fill: function (i) {
          var k = top.indexOf(String(E.columns[c][i]));
          return k < 0 ? OTHER : PALETTE[k]; },
          legend: top.map(function (t, k) { return [t, PALETTE[k]]; })
            .concat(Object.keys(counts).length > top.length ? [['other', OTHER]] : []) };
      }
      var vals = E.columns[c].map(function (v, i) { return value(G, c, i, 'num'); });
      var s = scaleFor(vals, 'num');
      return { fill: function (i) {
        var t = s.pos(vals[i]);
        if (t === null) return OTHER;
        var r = Math.round(207 + (11 - 207) * t), g = Math.round(224 + (61 - 224) * t), b = Math.round(243 + (145 - 243) * t);
        return 'rgb(' + r + ',' + g + ',' + b + ')'; },
        legend: [[fmt(s.lo), 'rgb(207,224,243)'], [fmt(s.hi), 'rgb(11,61,145)']] };
    }
    function scatter() {
      var W = 460, H = 320, L = 46, R = 10, T = 12, B = 34;
      var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, class: 'g-svg g-scatter', role: 'img' });
      var pts = [], px = [], py = [], xlab, ylab, sx, sy;
      if (state.axes.mode === 'pca') {
        var pca = E.pca;
        var xs = pca.scores.map(function (r) { return r[0]; });
        var ys = pca.scores.map(function (r) { return r.length > 1 ? r[1] : 0; });
        sx = scaleFor(xs, 'num'); sy = scaleFor(ys, 'num');
        pca.scores.forEach(function (r, k) {
          var i = G.posOf[pca.index[k]];
          if (i === undefined) return;
          pts.push(i); px.push(sx.pos(xs[k])); py.push(sy.pos(ys[k]));
        });
        var ex = arr(pca.explained);
        xlab = 'PC1' + (ex[0] !== undefined ? ' (' + fmt(Math.round(ex[0] * 1000) / 10) + ' %)' : '') + ' — rotated axis';
        ylab = 'PC2' + (ex[1] !== undefined ? ' (' + fmt(Math.round(ex[1] * 1000) / 10) + ' %)' : '') + ' — rotated axis';
      } else {
        var cx = state.axes.x, cy = state.axes.y;
        if (!cx || !cy) {
          return { svg: empty('Fewer than two basis members are in the explorer sample: nothing to plot.'),
                   legend: el('div'), rowPanel: el('div', { class: 'g-rowpanel' }) };
        }
        var kx = kindOf(G, cx), ky = kindOf(G, cy);
        var vx = [], vy = [];
        for (var i = 0; i < E.n; i++) { vx.push(value(G, cx, i, kx)); vy.push(value(G, cy, i, ky)); }
        sx = scaleFor(vx, kx); sy = scaleFor(vy, ky);
        for (var j = 0; j < E.n; j++) {
          var a = sx.pos(vx[j], j), b = sy.pos(vy[j], j + 7);
          if (a === null || b === null) continue;
          pts.push(j); px.push(a); py.push(b);
        }
        xlab = cx + (kx === 'cat' ? ' (categories, jittered)' : '');
        ylab = cy + (ky === 'cat' ? ' (categories, jittered)' : '');
      }
      svg.setAttribute('aria-label', 'scatter of ' + pts.length + ' sample rows: ' + xlab + ' against ' + ylab);
      svg.appendChild(sv('rect', { x: L, y: T, width: W - L - R, height: H - T - B, class: 'g-plot' }));
      function X(t) { return L + t * (W - L - R); }
      function Y(t) { return H - B - t * (H - T - B); }
      svg.appendChild(sv('text', { x: L, y: H - 18, class: 'g-tick', text: tickText(sx, 0) }));
      svg.appendChild(sv('text', { x: W - R, y: H - 18, class: 'g-tick', 'text-anchor': 'end', text: tickText(sx, 1) }));
      svg.appendChild(sv('text', { x: L - 4, y: H - B, class: 'g-tick', 'text-anchor': 'end', text: tickText(sy, 0) }));
      svg.appendChild(sv('text', { x: L - 4, y: T + 8, class: 'g-tick', 'text-anchor': 'end', text: tickText(sy, 1) }));
      svg.appendChild(sv('text', { x: (L + W - R) / 2, y: H - 4, class: 'g-axislabel g-xlabel', 'text-anchor': 'middle', text: xlab }));
      svg.appendChild(sv('text', { x: 12, y: (T + H - B) / 2, class: 'g-axislabel g-ylabel', 'text-anchor': 'middle',
                                   transform: 'rotate(-90 12 ' + (T + H - B) / 2 + ')', text: ylab }));
      var col = colourer();
      var sel = state.selectedRows;
      var layer = sv('g', { class: 'g-points' });
      var screen = [];
      pts.forEach(function (i, k) {
        var x = X(px[k]), y = Y(py[k]);
        screen.push([x, y, i]);
        var f = col.fill(i);
        layer.appendChild(sv('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: 2.8, 'data-i': i,
          class: 'g-pt' + (sel ? (sel[i] ? ' g-pt-sel' : ' g-pt-dim') : ''),
          style: f ? 'fill:' + f : null }));
      });
      svg.appendChild(layer);
      var brush = sv('rect', { class: 'g-brush', x: 0, y: 0, width: 0, height: 0, visibility: 'hidden' });
      svg.appendChild(brush);
      var rowPanel = el('div', { class: 'g-rowpanel', 'aria-live': 'polite' },
        [para('Hover a point to read its row; drag a rectangle to select rows.', 'g-muted')]);
      function toSvg(e) {
        var m = svg.getScreenCTM();
        if (!m) return null;
        var p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
        return p.matrixTransform(m.inverse());
      }
      var start = null;
      svg.addEventListener('mousedown', function (e) {
        var p = toSvg(e);
        if (!p) return;
        start = p;
        setAttrs(brush, { x: p.x, y: p.y, width: 0, height: 0, visibility: 'visible' });
        e.preventDefault();
      });
      svg.addEventListener('mousemove', function (e) {
        var p = toSvg(e);
        if (start && p) {
          setAttrs(brush, { x: Math.min(start.x, p.x), y: Math.min(start.y, p.y),
                            width: Math.abs(p.x - start.x), height: Math.abs(p.y - start.y) });
          return;
        }
        var i = e.target && e.target.getAttribute ? e.target.getAttribute('data-i') : null;
        if (i !== null && i !== undefined) showRow(rowPanel, +i);
      });
      svg.addEventListener('mouseup', function (e) {
        if (!start) return;
        var p = toSvg(e) || start;
        var x0 = Math.min(start.x, p.x), x1 = Math.max(start.x, p.x);
        var y0 = Math.min(start.y, p.y), y1 = Math.max(start.y, p.y);
        start = null;
        brush.setAttribute('visibility', 'hidden');
        if (x1 - x0 < 3 && y1 - y0 < 3) return;
        var chosen = {};
        screen.forEach(function (s) { if (s[0] >= x0 && s[0] <= x1 && s[1] >= y0 && s[1] <= y1) chosen[s[2]] = true; });
        state.selectedRows = Object.keys(chosen).length ? chosen : null;
        redraw();
      });
      svg.addEventListener('mouseleave', function () {
        if (start) { start = null; brush.setAttribute('visibility', 'hidden'); }
      });
      var legend = el('div', { class: 'g-legend' });
      if (state.colourBy) {
        legend.appendChild(el('span', { class: 'g-muted', text: 'colour: ' + state.colourBy + ' ' }));
        col.legend.forEach(function (l) {
          var sw = el('span', { class: 'g-swatch' }); sw.style.background = l[1];
          legend.appendChild(el('span', { class: 'g-legitem' }, [sw, trunc(l[0], 24)]));
        });
      }
      return { svg: svg, legend: legend, rowPanel: rowPanel };
    }
    function provChips(d) {
      var box = el('span', { class: 'g-prov' });
      ['semantic', 'symbolic', 'empirical'].forEach(function (ch) {
        var s = status(d, ch) || 'untested';
        var p = d.provenance[ch] || {};
        var txt = ch + ' ' + (STATUS_MARK[s] || s);
        if (ch === 'symbolic' && s !== 'untested' && isNum(p.verified_rows))
          txt += ' ' + p.verified_rows + '/' + p.of_rows;
        box.appendChild(chip(txt, 'g-st-' + s, p.note || p.evidence || null));
      });
      return box;
    }
    function showRow(panel, i) {
      clear(panel);
      panel.appendChild(el('h4', { text: 'row ' + (arr(E.index)[i] !== undefined ? E.index[i] : i) }));
      var dl = el('dl', { class: 'g-kv' });
      act.basis.forEach(function (c) {
        if (E.columns[c] === undefined) return;
        dl.appendChild(el('dt', { text: c }));
        dl.appendChild(el('dd', { text: fmt(E.columns[c][i]) }));
      });
      Object.keys(act.heads).forEach(function (h) {
        if (E.columns[h] === undefined) return;
        var r = act.heads[h];
        dl.appendChild(el('dt', { class: 'g-derived-k', text: h }));
        var dd = el('dd', null, [fmt(E.columns[h][i]) + ' ',
          chip(r.rule || r.orientation || 'rule', 'g-rule', r.formula)]);
        if (r.d) dd.appendChild(provChips(r.d));
        else dd.appendChild(chip('orientation rule — not verified', 'g-st-untested'));
        dl.appendChild(dd);
      });
      panel.appendChild(dl);
    }
    function loadings() {
      var pca = E.pca || {};
      var box = el('div', { class: 'g-loadings', 'data-panel': 'loadings' });
      box.appendChild(el('h4', { text: 'What the rotated axes are made of' }));
      arr(pca.loadings).slice(0, 2).forEach(function (vec, k) {
        var items = arr(pca.columns).map(function (c, j) { return [c, vec[j]]; })
          .sort(function (a, b) { return Math.abs(b[1]) - Math.abs(a[1]); }).slice(0, 5);
        box.appendChild(el('p', { class: 'g-prose', text: 'PC' + (k + 1) + ': ' + items.map(function (it) {
          return (it[1] >= 0 ? '+' : '−') + fmt(Math.abs(it[1])) + ' ' + it[0]; }).join(', ') +
          '. A rotated axis has no name of its own; the named basis stays the report\'s.' }));
      });
      return box;
    }

    /* ================================================================ View B — Derivations */
    function nodeClass(c) {
      var r = G.role(c), cls = 'g-node';
      if (OUTSIDE[r] || (r === 'text' && !has(act.basis, c))) cls += ' g-outside';
      else if (act.heads[c]) cls += ' g-derived';
      else cls += ' g-basis';
      if (G.cycleOf(c)) cls += ' g-cyc';
      var cand = currentCandidate();
      if (cand) {
        if (c === cand.label) cls += ' g-label';
        else if (has(cand.leak, c)) cls += ' g-leak';
        else if (has(cand.features, c)) cls += ' g-feature';
      }
      if (state.focus === c) cls += ' g-focus';
      return cls;
    }
    function currentCandidate() {
      if (!state.activeCandidate) return null;
      var cv = candidateView(G, act);
      return cv.authored.filter(function (x) { return x.label === state.activeCandidate; })[0] || null;
    }
    function layout() {
      var depth = {}, visiting = {};
      function dep(c) {
        if (depth[c] !== undefined) return depth[c];
        if (!act.heads[c] || visiting[c]) return 0;
        visiting[c] = true;
        var m = 0;
        act.heads[c].body.forEach(function (b) { m = Math.max(m, dep(b)); });
        visiting[c] = false;
        depth[c] = m + 1;
        return depth[c];
      }
      var colsBy = {}, maxD = 0, muted = [];
      G.cols.forEach(function (c) {
        var r = G.role(c);
        if (OUTSIDE[r] || (r === 'text' && !has(act.basis, c) && !act.heads[c])) { muted.push(c); return; }
        var d = dep(c);
        maxD = Math.max(maxD, d);
        (colsBy[d] = colsBy[d] || []).push(c);
      });
      if (muted.length) colsBy[maxD + 1] = muted;
      var pos = {}, rows = 0, CW = 160, RH = 30;
      Object.keys(colsBy).forEach(function (k) {
        colsBy[k].forEach(function (c, i) {
          var w = Math.min(136, 8 + 7 * Math.min(String(c).length, 18));
          pos[c] = { x: 14 + (+k) * CW, y: 30 + i * RH, w: w, h: 20 };
        });
        rows = Math.max(rows, colsBy[k].length);
      });
      var nCols = Object.keys(colsBy).length;
      return { pos: pos, W: Math.max(14 + nCols * CW, 320), H: 40 + rows * RH, CW: CW, mutedCol: muted.length ? maxD + 1 : null };
    }
    function drawGraph() {
      var V = views.graph;
      clear(V);
      V.appendChild(el('h3', { text: 'Derivations' }));
      var basisLine = el('p', { class: 'g-basis-line', 'data-basis': act.basis.join(',') });
      basisLine.appendChild(el('strong', { text: 'basis (' + act.basis.length + '): ' }));
      basisLine.appendChild(document.createTextNode(act.basis.join(', ')));
      var plus = act.basis.filter(function (c) { return !has(G.reportBasis, c); });
      var minus = G.reportBasis.filter(function (c) { return !has(act.basis, c); });
      if (plus.length || minus.length) {
        basisLine.appendChild(chip('re-oriented: ' + plus.map(function (c) { return '+' + c; })
          .concat(minus.map(function (c) { return '−' + c; })).join(' '), 'g-changed'));
      }
      V.appendChild(basisLine);
      var Lo = layout();
      var svg = sv('svg', { viewBox: '0 0 ' + Lo.W + ' ' + Lo.H, width: Lo.W, height: Lo.H, class: 'g-svg g-graph', role: 'group',
                            'aria-label': 'derivation graph: ' + G.cols.length + ' columns' });
      var defs = sv('defs');
      defs.appendChild(sv('marker', { id: 'lg-arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7,
        markerHeight: 7, orient: 'auto-start-reverse' }, [sv('path', { d: 'M0,0 L8,4 L0,8 z', class: 'g-arrowhead' })]));
      defs.appendChild(sv('marker', { id: 'lg-arrow-fd', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6,
        markerHeight: 6, orient: 'auto-start-reverse' }, [sv('path', { d: 'M0,0 L8,4 L0,8 z', class: 'g-arrowhead-fd' })]));
      svg.appendChild(defs);
      [['basis', 14], ['derived', 14 + Lo.CW]].forEach(function (h) {
        svg.appendChild(sv('text', { x: h[1], y: 14, class: 'g-colhead', text: h[0] }));
      });
      if (Lo.mutedCol !== null)
        svg.appendChild(sv('text', { x: 14 + Lo.mutedCol * Lo.CW, y: 14, class: 'g-colhead', text: 'outside the span' }));
      var P = Lo.pos;
      /* cycles, under everything */
      G.cycles.forEach(function (cy) {
        var ms = cy.members.filter(function (m) { return P[m]; });
        if (ms.length < 2) return;
        var d = ms.map(function (m, i) { return (i ? 'L' : 'M') + (P[m].x + P[m].w / 2) + ',' + (P[m].y + 10); }).join(' ') + ' Z';
        svg.appendChild(sv('path', { d: d, class: 'g-cycle', 'data-cycle': cy.id }));
      });
      /* active rules: body → head */
      var drawn = {};
      Object.keys(act.heads).forEach(function (h) {
        act.heads[h].body.forEach(function (b) {
          if (!P[b] || !P[h]) return;
          drawn[b + '>' + h] = true;
          var x1 = P[b].x + P[b].w, y1 = P[b].y + 10, x2 = P[h].x - 2, y2 = P[h].y + 10;
          var dx = Math.max(24, (x2 - x1) / 2);
          svg.appendChild(sv('path', { d: 'M' + x1 + ',' + y1 + ' C' + (x1 + dx) + ',' + y1 + ' ' + (x2 - dx) + ',' + y2 + ' ' + x2 + ',' + y2,
            class: 'g-edge' + (act.heads[h].d ? '' : ' g-edge-orient'), 'marker-end': 'url(#lg-arrow)',
            'data-edge': b + '>' + h }));
        });
      });
      /* exact functional dependencies the rules do not already draw */
      G.fds.forEach(function (f) {
        f.lhs.forEach(function (a) {
          var b = f.rhs;
          if (!P[a] || !P[b] || drawn[a + '>' + b]) return;
          drawn[a + '>' + b] = true;
          var x1 = P[a].x + P[a].w / 2, y1 = P[a].y + 10, x2 = P[b].x + P[b].w / 2, y2 = P[b].y + 10;
          var mx = (x1 + x2) / 2, my = (y1 + y2) / 2, nx = -(y2 - y1), ny = x2 - x1;
          var len = Math.sqrt(nx * nx + ny * ny) || 1, k = 18 / len;
          svg.appendChild(sv('path', { d: 'M' + x1 + ',' + y1 + ' Q' + (mx + nx * k) + ',' + (my + ny * k) + ' ' + x2 + ',' + y2,
            class: 'g-fd', 'marker-end': 'url(#lg-arrow-fd)', 'data-fd': a + '>' + b }));
        });
      });
      /* nodes */
      G.cols.forEach(function (c) {
        var p = P[c];
        if (!p) return;
        var g = sv('g', { class: nodeClass(c), 'data-node': c, tabindex: 0, role: 'button',
                          'aria-label': c + ', ' + (act.heads[c] ? 'derived' : G.role(c)) + ' — open its card',
                          transform: 'translate(' + p.x + ',' + p.y + ')' });
        g.appendChild(sv('rect', { width: p.w, height: p.h, rx: 10, ry: 10 }));
        g.appendChild(sv('text', { x: p.w / 2, y: 14, 'text-anchor': 'middle', text: trunc(c, 18) }));
        if (has(nodeClass(c).split(' '), 'g-leak'))
          g.appendChild(sv('line', { x1: 4, y1: 10, x2: p.w - 4, y2: 10, class: 'g-strike' }));
        g.appendChild(sv('title', { text: c }));
        var open = function () { state.focus = state.focus === c ? null : c; redraw(); };
        g.addEventListener('click', open);
        g.addEventListener('keydown', keyed(open));
        svg.appendChild(g);
      });
      var wrap = el('div', { class: 'g-graphwrap' });
      wrap.appendChild(svg);
      V.appendChild(wrap);
      V.appendChild(el('p', { class: 'g-muted', text: 'Filled: basis members. Hollow: derived by a kept rule ' +
        '(arrows run from the rule\'s body to its head). Dashed: exact functional dependencies the rules do ' +
        'not already draw. Dotted rings: cycles, whose direction is a choice. Click a column for its card.' }));
      if (state.focus) V.appendChild(card(state.focus));
      V.appendChild(orientPanel());
    }
    function consequenceBlock(cq, title) {
      var box = el('div', { class: 'g-cons' });
      if (!isObj(cq)) { box.appendChild(para('No consequence block was written for this decision.', 'g-muted')); return box; }
      if (title) box.appendChild(el('h5', { text: title }));
      if (cq.decision) box.appendChild(el('p', { class: 'g-decision', text: cq.decision }));
      if (arr(cq.alternatives).length)
        box.appendChild(el('p', { class: 'g-alt', text: 'weighed against: ' + cq.alternatives.join('; ') }));
      var ds = cq.downstream;
      if (typeof ds === 'string') box.appendChild(para(ds));
      else if (isObj(ds)) {
        var dl = el('dl', { class: 'g-kv' });
        Object.keys(ds).forEach(function (k) {
          dl.appendChild(el('dt', { text: k.replace(/_/g, ' ') }));
          dl.appendChild(el('dd', { text: String(ds[k]) }));
        });
        box.appendChild(dl);
      }
      return box;
    }
    function card(c) {
      var box = el('article', { class: 'g-card', 'data-card': c });
      var t = G.typing[c] || {};
      var head = el('div', { class: 'g-cardhead' }, [el('h4', { text: c })]);
      var st = G.stype(c), ft = G.ftype(c);
      head.appendChild(chip(act.heads[c] ? 'derived' : (has(act.basis, c) ? 'basis member' : G.role(c)),
                            act.heads[c] ? 'g-derived-chip' : 'g-basis-chip'));
      head.appendChild(chip(st && st !== ft ? 'typed ' + st + ' → ' + ft : 'type ' + ft, 'g-type'));
      box.appendChild(head);
      if (t.reason) box.appendChild(para(t.reason));
      if (t.consequences) box.appendChild(consequenceBlock(t.consequences, 'the typing decision'));
      var h = act.heads[c];
      if (h) {
        box.appendChild(el('p', { class: 'g-formula', text: (h.formula || c) + (h.rule ? '   [' + h.rule + ']' : '') }));
        if (h.d) {
          box.appendChild(provChips(h.d));
          ['semantic', 'symbolic', 'empirical'].forEach(function (ch) {
            var p = h.d.provenance[ch] || {};
            var txt = p.note || p.evidence;
            if (txt) box.appendChild(el('p', { class: 'g-muted', text: ch + ': ' + txt }));
          });
          box.appendChild(consequenceBlock(h.d.consequences, 'the derivation'));
        } else {
          box.appendChild(para('This rule belongs to the orientation chosen in the explorer; the report did ' +
            'not verify it, and its consequence block is the orientation\'s, below.', 'g-muted'));
        }
      }
      var rejected = G.derivs.filter(function (d) { return d.column === c && !kept(d); });
      rejected.forEach(function (d) {
        box.appendChild(el('p', { class: 'g-formula g-rejected', text: 'rejected: ' + d.formula + ' [' + d.rule_id + ']' }));
        box.appendChild(provChips(d));
        box.appendChild(consequenceBlock(d.consequences, 'why it was rejected'));
      });
      arr(M.disagreements).filter(function (x) { return isObj(x) && x.column === c; }).forEach(function (x) {
        box.appendChild(el('div', { class: 'g-disagree' }, [chip(x.kind || 'disagreement', 'g-warn'),
          para(x.reading || ''), x.evidence ? para('evidence: ' + x.evidence, 'g-muted') : null]));
      });
      arr(M.markers).filter(function (m) { return /^(WARN|FORK|ERROR):/.test(String(m)) && mentions(m, c); })
        .forEach(function (m) { box.appendChild(el('p', { class: 'g-marker', text: m })); });
      var cy = G.cycleOf(c);
      if (cy) {
        var sec = el('div', { class: 'g-cycard' });
        sec.appendChild(el('h5', { text: 'cycle ' + cy.id + ': ' + cy.members.join(', ') }));
        if (cy.reading) sec.appendChild(para(cy.reading));
        if (act.heads[c]) {
          var target = -1, cur = state.activeOrientation[cy.id];
          var order = [cy._report].concat(cy.orientations.map(function (o, i) { return i; }));
          order.forEach(function (i) {
            if (target < 0 && i !== cur && !has(arr(cy.orientations[i].derived), c)) target = i;
          });
          if (target >= 0) {
            sec.appendChild(button('make ' + c + ' the basis member', 'g-make-basis', function () {
              state.activeOrientation[cy.id] = target; redraw();
            }, { 'data-cycle': cy.id, 'data-orientation': target }));
          }
        } else {
          sec.appendChild(para(c + ' is this cycle\'s basis member under the active orientation.', 'g-muted'));
        }
        box.appendChild(sec);
      } else if (h) {
        box.appendChild(para('Not in a cycle: the direction of this rule is not a choice.', 'g-muted'));
      }
      if (state.selectedRows && exploreOk && E.columns[c] !== undefined) box.appendChild(selectionNote(c));
      box.appendChild(button('close', 'g-close', function () { state.focus = null; redraw(); }));
      return box;
    }
    function selectionNote(c) {
      var rows = Object.keys(state.selectedRows).map(Number);
      var vals = rows.map(function (i) { return E.columns[c][i]; }).filter(function (v) { return v !== null && v !== undefined; });
      var txt;
      if (vals.length && vals.every(isNum) && kindOf(G, c) === 'num') {
        txt = 'in the ' + rows.length + ' selected rows: ' + fmt(Math.min.apply(null, vals)) + ' to ' + fmt(Math.max.apply(null, vals));
      } else {
        var counts = {};
        vals.forEach(function (v) { counts[String(v)] = (counts[String(v)] || 0) + 1; });
        txt = 'in the ' + rows.length + ' selected rows: ' + Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; })
          .slice(0, 6).map(function (k) { return trunc(k, 20) + ' ×' + counts[k]; }).join(', ');
      }
      return el('p', { class: 'g-selnote', text: txt });
    }
    function orientPanel() {
      var box = el('div', { class: 'g-orient', 'data-panel': 'orientations' });
      if (!G.cycles.length) {
        box.appendChild(para('No cycle: every derivation here has one direction, so the basis has no alternative orientation.', 'g-muted'));
        return box;
      }
      box.appendChild(el('h4', { text: 'Orientations' }));
      var changed = false;
      G.cycles.forEach(function (cy) {
        var cur = state.activeOrientation[cy.id];
        if (cur !== cy._report) changed = true;
        var sec = el('div', { class: 'g-orient-cycle', 'data-cycle': cy.id });
        sec.appendChild(el('h5', { text: cy.id + ' (' + cy.members.join(', ') + ')' }));
        var btns = el('div', { class: 'g-btnrow' });
        cy.orientations.forEach(function (o, i) {
          btns.appendChild(button('derive ' + arr(o.derived).join(' and ') + (i === cy._report ? ' (report)' : ''),
            'g-orient-btn' + (i === cur ? ' g-on' : ''), function () {
              state.activeOrientation[cy.id] = i; redraw();
            }, { 'aria-pressed': i === cur ? 'true' : 'false', 'data-orientation': i }));
        });
        sec.appendChild(btns);
        var o = cy.orientations[cur] || {};
        var cons = consequenceBlock(o.consequences, 'what this orientation implies');
        cons.setAttribute('data-orientation-id', o.id || String(cur));
        cons.classList.add('g-orient-cons');
        sec.appendChild(cons);
        box.appendChild(sec);
      });
      if (changed) {
        box.appendChild(button('restore the report\'s orientation', 'g-restore', function () {
          G.cycles.forEach(function (cy) { state.activeOrientation[cy.id] = cy._report; });
          redraw();
        }));
      }
      return box;
    }

    /* ================================================================ View C — Orthogonality */
    function drawOrtho() {
      var V = views.orthogonality;
      clear(V);
      V.appendChild(el('h3', { text: 'Orthogonality' }));
      var O = isObj(M.orthogonality) ? M.orthogonality : {};
      if (O.reading) V.appendChild(para(O.reading));
      var pairs = arr(G.deps.pairs);
      if (!exploreOk || !pairs.length) {
        V.appendChild(empty('No explorer sample or no dependency measures ship with this layer: the heatmap ' +
          'needs the script\'s explore payload and its pair measures.'));
        return;
      }
      var idx = {};
      pairs.forEach(function (p) { if (isObj(p)) { idx[p.a + '\u0000' + p.b] = p; idx[p.b + '\u0000' + p.a] = p; } });
      var authored = {};
      arr(O.pairs).forEach(function (p) { if (isObj(p)) { authored[p.a + '\u0000' + p.b] = p; authored[p.b + '\u0000' + p.a] = p; } });
      var measured = arr(G.deps.columns);
      var ms = act.basis.filter(function (c) { return has(measured, c); });
      if (ms.length < 2) { V.appendChild(empty('Fewer than two basis members were measured.')); return; }
      var th = isObj(G.deps.thresholds) ? G.deps.thresholds : (isObj(O.thresholds) ? O.thresholds : {});
      var nmiT = isNum(th.nmi) ? th.nmi : 0.5;
      var C = 30, LW = 112, TH = 92, k = ms.length;
      var W = LW + k * C + 8, H = TH + k * C + 8;
      var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, class: 'g-svg g-heat', role: 'group',
                            'aria-label': 'normalized mutual information among ' + k + ' basis members' });
      ms.forEach(function (c, i) {
        svg.appendChild(sv('text', { x: LW - 6, y: TH + i * C + C / 2 + 4, 'text-anchor': 'end', class: 'g-tick', text: trunc(c, 16) }));
        var tx = LW + i * C + C / 2;
        svg.appendChild(sv('text', { x: tx, y: TH - 6, class: 'g-tick', transform: 'rotate(-45 ' + tx + ' ' + (TH - 6) + ')', text: trunc(c, 16) }));
      });
      ms.forEach(function (a, i) {
        ms.forEach(function (b, j) {
          var x = LW + j * C, y = TH + i * C;
          if (i === j) { svg.appendChild(sv('rect', { x: x + 1, y: y + 1, width: C - 2, height: C - 2, class: 'g-diag' })); return; }
          var p = idx[a + '\u0000' + b];
          var v = p && isNum(p.nmi) ? p.nmi : null;
          var sel = state.pair && ((state.pair[0] === a && state.pair[1] === b) || (state.pair[0] === b && state.pair[1] === a));
          var g = sv('g', { class: 'g-cell' + (v !== null && v >= nmiT ? ' g-strong' : '') + (sel ? ' g-cell-sel' : ''),
                            'data-pair': a + '|' + b, tabindex: 0, role: 'button',
                            'aria-label': a + ' and ' + b + ': ' + (v === null ? 'not measured' : 'NMI ' + fmt(v)) + ' — open as the space view\'s axes' });
          var r = sv('rect', { x: x + 1, y: y + 1, width: C - 2, height: C - 2, class: 'g-cellbg' });
          if (v !== null) r.setAttribute('style', 'fill-opacity:' + Math.max(0.06, Math.min(1, v)).toFixed(3));
          g.appendChild(r);
          g.appendChild(sv('title', { text: pairText(a, b, p, authored[a + '\u0000' + b]) }));
          var pick = function () {
            state.pair = [a, b];
            state.axes = { x: a, y: b, mode: 'named' };
            redraw();
          };
          g.addEventListener('click', pick);
          g.addEventListener('keydown', keyed(pick));
          svg.appendChild(g);
        });
      });
      var wrap = el('div', { class: 'g-row2' });
      wrap.appendChild(el('div', { class: 'g-heatwrap' }, [svg]));
      var side = el('div', { class: 'g-pairpanel', 'data-panel': 'pair' });
      if (state.pair) {
        var a = state.pair[0], b = state.pair[1], p = idx[a + '\u0000' + b], au = authored[a + '\u0000' + b];
        side.appendChild(el('h4', { text: a + ' × ' + b }));
        side.appendChild(para(pairText(a, b, p, null)));
        if (au && au.independence) side.appendChild(el('p', { class: 'g-independence', text: 'independence: ' + au.independence }));
        else side.appendChild(para('The report wrote no independence reading for this pair.', 'g-muted'));
      } else {
        side.appendChild(para('Darker is more dependent (normalized mutual information, 0 to 1); an outlined cell ' +
          'crosses the script\'s threshold (' + fmt(nmiT) + '). Click a cell to open the pair as the space view\'s axes.', 'g-muted'));
      }
      if (state.selectedRows) side.appendChild(para('These measures are the full-data ones; the selection does not change them.', 'g-muted'));
      wrap.appendChild(side);
      V.appendChild(wrap);
    }
    function pairText(a, b, p, au) {
      if (!p) return a + ' and ' + b + ': not measured by the script';
      var parts = [];
      ['nmi', 'pearson', 'spearman', 'eta2', 'cramers_v'].forEach(function (m) {
        if (isNum(p[m])) parts.push((m === 'eta2' ? 'η²' : m === 'cramers_v' ? 'Cramér\'s V' : m === 'nmi' ? 'NMI' : m) + ' ' + fmt(p[m]));
      });
      return a + ' and ' + b + ': ' + parts.join(', ') + (au && au.independence ? ' — ' + au.independence : '');
    }

    /* ================================================================ View D — Partition */
    function drawPartition() {
      var V = views.partition;
      clear(V);
      var cv = candidateView(G, act);
      V.setAttribute('data-others', cv.others.join(','));
      V.appendChild(el('h3', { text: 'Partition' }));
      var P = G.P;
      var head = el('p', { class: 'g-prose' });
      head.appendChild(document.createTextNode(P.chosen ? 'The report chose ' : 'No partition is chosen'));
      if (P.chosen) head.appendChild(el('strong', { text: P.chosen }));
      head.appendChild(document.createTextNode(P.provenance ? ' (' + P.provenance + ').' : '.'));
      if (P.provenance === 'abstained')
        head.appendChild(document.createTextNode(' The candidates are ranked; /dataset-shaper decides with them in view.'));
      V.appendChild(head);
      if (!cv.authored.length) V.appendChild(empty('The layer states no label candidate.'));
      var grid = el('div', { class: 'g-cards' });
      cv.authored.forEach(function (x) {
        var on = state.activeCandidate === x.label;
        var c = x.c;
        var cardEl = el('article', { class: 'g-cand' + (on ? ' g-on' : '') + (P.chosen === x.label ? ' g-chosen' : ''),
                                     'data-candidate': x.label });
        var pick = button((on ? 'selected: ' : 'select ') + x.label, 'g-cand-btn', function () {
          state.activeCandidate = on ? null : x.label;
          if (!on && exploreOk && E.columns[x.label] !== undefined) state.colourBy = x.label;
          redraw();
        }, { 'aria-pressed': on ? 'true' : 'false' });
        cardEl.appendChild(el('div', { class: 'g-cardhead' }, [pick, chip(c.task || 'task?', 'g-type'),
          c.rule_id ? chip(c.rule_id, 'g-rule') : null, P.chosen === x.label ? chip('chosen', 'g-chosen-chip') : null]));
        var dim = el('p', { class: 'g-dim' });
        dim.appendChild(el('strong', { text: 'input_dim ' + (isNum(c.input_dim) ? c.input_dim : '?') }));
        if (x.changed) dim.appendChild(document.createTextNode(' in the report; ' + x.input_dim + ' under the active orientation'));
        dim.setAttribute('data-input-dim', String(x.changed ? x.input_dim : (isNum(c.input_dim) ? c.input_dim : x.input_dim)));
        cardEl.appendChild(dim);
        cardEl.appendChild(el('p', { class: 'g-feats', text: 'features: ' + x.features.join(', ') }));
        var lk = el('p', { class: 'g-leakline' }, ['leakage set: ']);
        if (x.leak.length) x.leak.forEach(function (l) { lk.appendChild(el('s', { class: 'g-leakcol', text: l })); lk.appendChild(document.createTextNode(' ')); });
        else lk.appendChild(document.createTextNode('none — no column derives this label'));
        cardEl.appendChild(lk);
        x.notes.forEach(function (n) { cardEl.appendChild(el('p', { class: 'g-warnline', text: n })); });
        if (on && state.selectedRows && exploreOk && E.columns[x.label] !== undefined) cardEl.appendChild(selectionNote(x.label));
        cardEl.appendChild(consequenceBlock(c.consequences, 'what this partition implies'));
        grid.appendChild(cardEl);
      });
      V.appendChild(grid);
      if (cv.others.length) {
        var o = el('div', { class: 'g-others', 'data-panel': 'other-heads' });
        o.appendChild(para('Also rule heads whose body lies entirely in the current basis — mechanically label ' +
          'candidates, not weighed as such by the analyst, so no consequence block exists for them: ' +
          cv.others.join(', ') + '.', 'g-muted'));
        V.appendChild(o);
      }
      V.appendChild(el('p', { class: 'g-materialize', text: 'This layer cannot write files. The chosen partition ' +
        'becomes shaped data through /dataset-shaper <file> --target <label>, with the leakage drop and the split ' +
        'as traced recipe steps.' }));
    }

    /* ================================================================ footer */
    function footer() {
      var st = isObj(M.stats) ? M.stats : {};
      var keys = [];
      Object.keys(st).forEach(function (c) { Object.keys(st[c] || {}).forEach(function (k) { if (!has(keys, k)) keys.push(k); }); });
      if (keys.length) {
        var d = details('classic statistics (geometry.json, verbatim)', 'g-stats');
        var t = el('table', { class: 'g-table' });
        var tr = el('tr', null, [el('th', { text: 'column' })]);
        keys.forEach(function (k) { tr.appendChild(el('th', { text: k.replace(/_/g, ' ') })); });
        t.appendChild(el('thead', null, [tr]));
        var tb = el('tbody');
        Object.keys(st).forEach(function (c) {
          var r = el('tr', null, [el('td', { text: c })]);
          keys.forEach(function (k) { r.appendChild(el('td', { text: fmt((st[c] || {})[k]) })); });
          tb.appendChild(r);
        });
        t.appendChild(tb);
        d.appendChild(t);
        pane.appendChild(d);
      }
      if (arr(M.markers).length) {
        var mk = details('run markers (' + M.markers.length + ')', 'g-markers');
        M.markers.forEach(function (m) { mk.appendChild(el('p', { class: 'g-marker', text: String(m) })); });
        pane.appendChild(mk);
      }
      if (isObj(M.handoff)) {
        pane.appendChild(el('div', { class: 'g-handoff' }, [el('h3', { text: 'Hand-off' }),
          para((M.handoff.next ? M.handoff.next + ' — ' : '') + (M.handoff.note || ''))]));
      }
    }

    redraw();
    footer();
    window.__geometry = {
      version: VERSION,
      state: function () { return JSON.parse(JSON.stringify(state)); },
      basis: function () { return act.basis.slice(); },
      heads: function () { return Object.keys(act.heads); },
      candidates: function () { return candidateView(G, act); },
      orient: function (cycleId, i) { state.activeOrientation[cycleId] = i; redraw(); },
      restore: function () { G.cycles.forEach(function (cy) { state.activeOrientation[cy.id] = cy._report; }); redraw(); },
      views: function () { return Object.keys(views); }
    };
  });
})();
