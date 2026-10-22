/* /vega-forge — `architecture` layer render script (contract: references/netspec-contract.md §4).
   Vanilla JS, no libraries, no network, inline SVG only. Reads ONLY #layer-architecture-data.
   Mounts one tab (data-layer="architecture", labelled Network) with nine surfaces (the sample
   panel sits inside the blocks one):
     blueprint-map — module → port → file → boundary → status; a pending port is marked, because
       an unimplemented port is the one thing a reader must not mistake for a shipped one.
     conformance  — the mechanical verdict, each judgement tool's audits and what they confirmed
                    and left unconfirmed, the recorded
       findings with their anchors and the justification for each anchor.
     blocks       — the network as a diagram: one node per top-level block with the out-shape and
       the parameter count the smoke pass recorded, composites drawn as groups of their inner blocks;
       beside it the sample panel (data-view="sample"): what one sample is — a row, a window (length,
       stride, horizon, series, the windows of each part, context, autoregressive) or a graph (the
       level, protocol and batching, the training and valid views' nodes, edges and targets, the
       test edges withheld, and the exposure chip when the protocol is transductive).
     decisions    — every traced choice, grouped by decision family, with the source that justifies
       it, the rationale and the alternatives that were weighed.
     mechanics    — the loss with this spec's parameters, the optimizer's update rule, the schedule.
     smoke        — what the forward pass measured, and a regression-family head's fixed target
       affine y = z·scale + shift (the train target's mean and spread, `smoke.init_notes`, A37).
     forks        — what was asked and what answered it; an unanswered fork says the default applied.
     defaults     — each default the spec carries and the observation that would overturn it.
   It computes nothing: every number here was written by the emitter's smoke pass. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="architecture"]')) return;

  var VERSION = 'vega-forge/architecture-render@1';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var COMPOSITE = { residual: 1, sequential: 1, parallel: 1 };
  var FAM_ORDER = ['architecture', 'objective', 'optimization', 'regularization'];
  var FAM_LABEL = { architecture: 'Architecture', objective: 'Objective',
                    optimization: 'Optimization', regularization: 'Regularization' };
  /* what `smoke.madds_unit` counts per: a row, an anchor (one window), a target node, a graph */
  var UNIT = { row: 'row', anchor: 'anchor (one window)', target_node: 'target node',
               graph: 'graph' };

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (k.indexOf('on') === 0 && typeof attrs[k] === 'function') n[k] = attrs[k];
      else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) {
      if (c) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  function svg(tag, attrs, kids) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) n.appendChild(c); });
    return n;
  }
  function isArr(v) { return Object.prototype.toString.call(v) === '[object Array]'; }
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v === 'number') {
      if (v === Math.round(v)) return String(v);
      /* small learning rates and decays must not round to zero */
      if (Math.abs(v) < 1e-3) return v.toExponential(2);
      return String(Math.round(v * 1e6) / 1e6);
    }
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }
  function num(v) {
    if (typeof v !== 'number') return fmt(v);
    return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  }
  function shape(s) { return isArr(s) ? '[' + s.join('×') + ']' : fmt(s); }
  function clip(s, n) {
    s = String(s === null || s === undefined ? '' : s);
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  }
  function chip(text, cls, title) {
    var c = el('span', { class: 'a-chip ' + (cls || ''), text: text });
    if (title) c.setAttribute('title', title);
    return c;
  }
  function para(t, cls) { return el('p', { class: cls || 'a-prose', text: t }); }
  function details(summary, cls) {
    var d = el('details', { class: cls || 'a-det' });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }
  function sourceClass(src) {
    src = String(src || '');
    if (src.indexOf('geometry:') === 0) return 'a-src-geometry';
    if (src.indexOf('analysis:') === 0) return 'a-src-analysis';
    if (src.indexOf('shape:') === 0) return 'a-src-shape';
    if (src.indexOf('blueprint:') === 0) return 'a-src-blueprint';
    if (src.indexOf('user:') === 0) return 'a-src-user';
    return 'a-src-default';
  }
  function section(pane, view, title) {
    var d = el('div', { class: 'a-sec', 'data-view': view });
    d.appendChild(el('h3', { text: title }));
    pane.appendChild(d);
    return d;
  }
  function statGrid(pairs) {
    var g = el('div', { class: 'a-stats' });
    pairs.forEach(function (p) {
      if (!p) return;
      g.appendChild(el('div', { class: 'a-stat' }, [
        el('div', { class: 'a-statv', text: p[1] }),
        el('div', { class: 'a-statl', text: p[0] }),
        p[2] ? el('div', { class: 'a-statn', text: p[2] }) : null]));
    });
    return g;
  }
  function table(headers, cls) {
    var t = el('table', { class: 'a-table ' + (cls || '') });
    var hr = el('tr');
    headers.forEach(function (h) { hr.appendChild(el('th', { text: h })); });
    t.appendChild(el('thead', null, [hr]));
    t.appendChild(el('tbody'));
    return t;
  }

  function pickTabContainer() {
    return document.querySelector('nav.tabs') || document.querySelector('nav[role="tablist"]')
        || document.querySelector('[data-tabs]') || null;
  }
  function pickPaneSibling() {
    var panes = document.querySelectorAll('section.tab-pane');
    return panes.length ? panes[panes.length - 1] : null;
  }
  function switchTab(name) {
    document.querySelectorAll('nav.tabs button[data-tab], nav[role="tablist"] button[data-tab]')
      .forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-tab') === name); });
    document.querySelectorAll('section.tab-pane')
      .forEach(function (p) { p.classList.toggle('active', p.getAttribute('data-tab') === name); });
  }
  function mount() {
    var pane = el('section', { class: 'layer-architecture tab-pane', 'data-layer': 'architecture',
                               'data-tab': 'architecture', id: 'tab-architecture', role: 'tabpanel',
                               'aria-label': 'Network' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'architecture', 'data-layer': 'architecture', role: 'tab',
                             text: 'Network' });
      b.onclick = function () { switchTab('architecture'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  /* ---------------- the traced choices, enumerated exactly as the contract groups them.
     One entry per choice: the family's keys in order, a list-valued key (blocks,
     regularization) contributing one choice per member. The smoke test counts the same way. */
  function choiceLabel(key, o, i) {
    if (key === 'blocks') return (o.id || ('#' + (i + 1))) + ' · ' + (o.type || '?');
    if (key === 'regularization') return (o.kind || 'regularization') + ' = ' + fmt(o.value)
      + (isArr(o.where) && o.where.length ? ' on ' + o.where.join(', ') : '');
    if (key === 'family') return 'family: ' + (o.name || '?');
    if (key === 'metrics') return 'metrics: ' + (o.primary || '?') + ' (selection on '
      + (((o.selection || {}).monitor) || '?') + ')';
    if (key === 'training') return 'training: batch ' + fmt(o.batch_size) + ', up to '
      + fmt(o.max_epochs) + ' epochs';
    if (key === 'init') return 'init: ' + (o.scheme || '?') + ', bias ' + fmt(o.bias);
    if (key === 'early_stopping') return 'early stopping: ' + (o.monitor || '?')
      + ', patience ' + fmt(o.patience);
    return key + ': ' + (o.name || '?');
  }
  function altText(alt, key) {
    if (!alt || typeof alt !== 'object') return fmt(alt);
    var parts = [];
    Object.keys(alt).forEach(function (k) {
      if (k === 'when') return;
      parts.push(k === 'name' || k === 'value' ? fmt(alt[k]) : k + ' ' + fmt(alt[k]));
    });
    return (parts.join(', ') || key) + (alt.when ? ' — when ' + alt.when : '');
  }
  function enumerateChoices(spec) {
    var fams = spec.decision_families || {}, order = [], out = [];
    FAM_ORDER.forEach(function (f) { if (fams[f]) order.push(f); });
    Object.keys(fams).forEach(function (f) { if (order.indexOf(f) < 0) order.push(f); });
    order.forEach(function (fam) {
      (fams[fam] || []).forEach(function (key) {
        var v = spec[key];
        if (v === null || v === undefined) return;
        var items = isArr(v) ? v : [v];
        items.forEach(function (o, i) {
          if (!o || typeof o !== 'object') return;
          out.push({ family: fam, key: key, index: isArr(v) ? i : -1, obj: o,
                     label: choiceLabel(key, o, i), source: o.source,
                     catalog: o.catalog, rationale: o.rationale,
                     alternatives: o.alternatives || [] });
        });
      });
    });
    return out;
  }

  /* ---------------- the block diagram. The spec gives the blocks; the smoke record gives each
     one's out-shape and parameter count, keyed by id. Nothing here is computed. */
  function paramSummary(params) {
    if (!params) return '';
    var keep = {};
    Object.keys(params).forEach(function (k) { if (k !== 'inner') keep[k] = params[k]; });
    var ks = Object.keys(keep);
    if (!ks.length) return '';
    return ks.map(function (k) { return k + ' ' + fmt(keep[k]); }).join(', ');
  }
  function drawBlocks(host, blocks, perBlock) {
    if (!blocks.length) {
      host.appendChild(para('The spec carries no blocks; there is no network to draw.', 'a-empty'));
      return;
    }
    var W = 760, PAD = 14, GAP = 30, NW = 420, X0 = Math.round((W - NW) / 2), MID = Math.round(W / 2);
    var items = blocks.map(function (b) {
      var inner = (b.params && isArr(b.params.inner)) ? b.params.inner : [];
      var comp = COMPOSITE[b.type] && inner.length;
      return { b: b, inner: comp ? inner : [], h: comp ? 44 + inner.length * 22 + 8 : 58 };
    });
    var H = PAD * 2;
    items.forEach(function (it, i) { H += it.h + (i ? GAP : 0); });
    var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'a-svg a-blocks', role: 'img',
                         'aria-label': 'Block diagram: ' + blocks.length + ' top-level blocks, '
                           + 'each with its type, out-shape and parameter count' });
    var y = PAD;
    items.forEach(function (it, i) {
      var b = it.b, pb = perBlock[b.id] || null;
      var node = svg('g', { class: 'a-node', 'data-block': b.id || ('#' + (i + 1)) });
      node.appendChild(svg('rect', { x: X0, y: y, width: NW, height: it.h, rx: 6,
                                     class: 'a-nrect' + (it.inner.length ? ' a-ncomp' : '') }));
      node.appendChild(svg('text', { x: X0 + 12, y: y + 19, class: 'a-nid',
        text: (b.id || '') + ' · ' + (b.type || '?') }));
      node.appendChild(svg('text', { x: X0 + NW - 12, y: y + 19, 'text-anchor': 'end',
        class: 'a-nmeta', text: (pb && pb.out_shape !== undefined ? 'out ' + shape(pb.out_shape)
                                                                  : 'out —')
          + ' · ' + (pb && pb.params !== undefined ? num(pb.params) + ' params' : 'params —') }));
      var ps = paramSummary(b.params);
      if (ps) node.appendChild(svg('text', { x: X0 + 12, y: y + 37, class: 'a-nparam',
                                             text: clip(ps, 62) }));
      it.inner.forEach(function (inb, j) {
        var iy = y + 44 + j * 22;
        node.appendChild(svg('rect', { x: X0 + 18, y: iy, width: NW - 36, height: 18, rx: 3,
                                       class: 'a-inner' }));
        node.appendChild(svg('text', { x: X0 + 26, y: iy + 13, class: 'a-itext',
          text: clip((inb.type || '?') + (paramSummary(inb.params) ? ' — '
                     + paramSummary(inb.params) : ''), 70) }));
      });
      g.appendChild(node);
      if (i < items.length - 1) {
        var y1 = y + it.h, y2 = y1 + GAP;
        g.appendChild(svg('line', { x1: MID, y1: y1, x2: MID, y2: y2 - 7, class: 'a-arrow' }));
        g.appendChild(svg('polygon', { class: 'a-arrowhead',
          points: MID + ',' + y2 + ' ' + (MID - 5) + ',' + (y2 - 8) + ' ' + (MID + 5) + ',' + (y2 - 8) }));
      }
      y += it.h + GAP;
    });
    host.appendChild(g);
  }

  /* ---------------- the schedule curve: the learning rates the emitter recorded, over the
     epochs the spec allows. A missing record is said, not guessed at. */
  function drawSchedule(host, lrs, epochs) {
    if (!isArr(lrs) || !lrs.length) {
      host.appendChild(para('The smoke pass recorded no schedule; the curve cannot be drawn.',
                            'a-empty'));
      return;
    }
    var W = 720, H = 150, L = 62, R = 14, T = 14, B = 28;
    var max = lrs[0], i;
    for (i = 1; i < lrs.length; i++) if (lrs[i] > max) max = lrs[i];
    if (!(max > 0)) max = 1;
    var span = Math.max(1, (epochs && epochs > 1 ? epochs : lrs.length) - 1);
    var pts = [];
    for (i = 0; i < lrs.length; i++) {
      var x = L + (Math.min(i, span) / span) * (W - L - R);
      var yy = T + (1 - lrs[i] / max) * (H - T - B);
      pts.push((Math.round(x * 10) / 10) + ',' + (Math.round(yy * 10) / 10));
    }
    var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'a-svg a-chart', role: 'img',
                         'aria-label': 'Learning rate over ' + (epochs || lrs.length)
                           + ' epochs, peaking at ' + fmt(max) });
    g.appendChild(svg('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, class: 'a-axis' }));
    g.appendChild(svg('line', { x1: L, y1: T, x2: L, y2: H - B, class: 'a-axis' }));
    g.appendChild(svg('polyline', { points: pts.join(' '), class: 'a-line' }));
    g.appendChild(svg('text', { x: L - 6, y: T + 4, 'text-anchor': 'end', class: 'a-tick',
                                text: fmt(max) }));
    g.appendChild(svg('text', { x: L - 6, y: H - B, 'text-anchor': 'end', class: 'a-tick',
                                text: '0' }));
    g.appendChild(svg('text', { x: L, y: H - 8, class: 'a-tick', text: 'epoch 1' }));
    g.appendChild(svg('text', { x: W - R, y: H - 8, 'text-anchor': 'end', class: 'a-tick',
                                text: 'epoch ' + (epochs || lrs.length) }));
    host.appendChild(g);
  }

  /* ---------------- the sample panel (E20): what one sample is. The shape and its parameters
     are the spec's declaration; every count is the smoke pass's `sample` record (the windows
     each part yields, the training and valid views of the graph). Nothing here is computed. */
  function declaredSample(spec) {
    return spec.sample || (spec.data_contract || {}).sample || {};
  }
  function given(v) { return v !== undefined && v !== null && v !== ''; }
  function pick(measured, declared, key) {
    return measured && given(measured[key]) ? measured[key] : declared[key];
  }
  var GRID = 'display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.45rem';
  function kstat(k, label, value, note) {
    return el('div', { class: 'a-stat', 'data-k': k, style: 'min-width:0;padding:.3rem .6rem' }, [
      el('div', { class: 'a-statv', text: value }), el('div', { class: 'a-statl', text: label }),
      note ? el('div', { class: 'a-statn', text: note }) : null]);
  }
  function kchip(k, text, cls, title) {
    var c = chip(text, cls, title);
    c.setAttribute('data-k', k);
    return c;
  }
  /* one window as a strip of cells: the input rows up to the anchor, then the forecast steps */
  function windowStrip(L, H) {
    var CAP = 24, HCAP = 8, CW = 12, STEP = 14, more = L > CAP;
    var nL = Math.min(L, CAP), nH = Math.min(H, HCAP), x = 8;
    var W = 16 + (nL + (more ? 1 : 0) + nH + (H > HCAP ? 1 : 0)) * STEP + (nH ? 8 : 0);
    var g = svg('svg', { viewBox: '0 0 ' + W + ' 36', class: 'a-svg a-wstrip', role: 'img',
                         'aria-label': 'One window: ' + L + ' input rows ending at the anchor'
                           + (H ? ', then ' + H + ' forecast step' + (H > 1 ? 's' : '') : '') });
    if (more) { g.appendChild(svg('text', { x: x + 2, y: 16, class: 'a-tick', text: '…' })); x += STEP; }
    var x0 = x, i;
    for (i = 0; i < nL; i++, x += STEP)
      g.appendChild(svg('rect', { x: x, y: 5, width: CW, height: 14, rx: 2,
                                  class: i === nL - 1 ? 'a-nrect' : 'a-inner' }));
    g.appendChild(svg('text', { x: x0, y: 32, class: 'a-tick', text: 'L = ' + L }));
    if (nH) {
      x += 8;
      var h0 = x;
      for (i = 0; i < nH; i++, x += STEP)
        g.appendChild(svg('rect', { x: x, y: 5, width: CW, height: 14, rx: 2,
                                    class: 'a-nrect a-ncomp' }));
      if (H > HCAP) g.appendChild(svg('text', { x: x + 2, y: 16, class: 'a-tick', text: '…' }));
      g.appendChild(svg('text', { x: h0, y: 32, class: 'a-tick', text: 'H = ' + H }));
    }
    return g;
  }
  /* why a valid row gets no window (D1 §9.3): too little context before it, or (a forecast) a
     horizon past the part's end. The record counts those rows but not the reason, so the
     caption names every reason the declared context and horizon allow, and picks none. */
  function unscoredNote(ctx, H) {
    if (ctx === 'none') return 'too little context before them'
      + (H ? ', or a horizon past the part’s end' : ' for a window');
    if (ctx === 'previous-part') return H
      ? 'a horizon past the part’s end, or too little history in their series'
      : 'too little history in their series for a window';
    return 'no scorable window';
  }
  function windowPanel(box, M, D, missing) {
    var L = pick(M, D, 'length'), S = pick(M, D, 'stride'), H = pick(M, D, 'horizon') || 0;
    var ctx = pick(M, D, 'context'), ar = pick(M, D, 'autoregressive');
    box.appendChild(para('Each sample is a window of ' + fmt(L) + ' consecutive rows'
      + (D.series_by ? ' of one series (' + D.series_by + ')' : ', across the whole frame (no series key)')
      + (D.order_by ? ', in ' + D.order_by + ' order' : '') + '; '
      + (H ? 'it forecasts the next ' + fmt(H) + ' step' + (H > 1 ? 's' : '') + ' after its last row'
           : 'its target is its last row’s') + ', and training anchors are taken '
      + (S === 1 ? 'at every row' : 'every ' + fmt(S) + ' rows') + '.', 'a-muted'));
    if (typeof L === 'number' && L > 0) box.appendChild(windowStrip(L, H));
    var stats = [kstat('length', 'length', fmt(L)), kstat('stride', 'stride', fmt(S))];
    if (H) stats.push(kstat('horizon', 'horizon', fmt(H)));
    if (!missing) {
      var w = M.windows || {}, u = M.unscored || {};
      stats.push(kstat('series', 'series', num(M.series)),
                 kstat('windows_train', 'windows · train', num(w.train)),
                 kstat('windows_valid', 'windows · valid', num(w.valid)),
                 kstat('unscored_valid', 'unscored · valid', num(u.valid), unscoredNote(ctx, H)));
    }
    box.appendChild(el('div', { class: 'a-stats', style: GRID }, stats));
    box.appendChild(el('div', { class: 'a-chips' }, [
      kchip('context', 'context ' + fmt(ctx), 'a-tag', ctx === 'none'
        ? 'the first rows of each part are context only' : 'each series is prefixed with the last '
          + 'rows of the part before it'),
      kchip('autoregressive', 'autoregressive ' + (ar ? 'yes' : 'no'), 'a-tag', ar
        ? 'the past target is fed back as an input channel' : 'the target is never an input')]));
  }
  function graphPanel(box, M, D, missing) {
    var level = pick(M, D, 'level'), proto = pick(M, D, 'protocol'), batching = pick(M, D, 'batching');
    var byGraph = level === 'graph';
    var chips = el('div', { class: 'a-chips' }, [
      kchip('level', 'level ' + fmt(level), 'a-tag'),
      kchip('protocol', 'protocol ' + fmt(proto), proto === 'transductive' ? 'a-warn' : 'a-tag'),
      kchip('batching', 'batching ' + fmt(batching), 'a-tag')]);
    if (isArr(D.neighbours) && D.neighbours.length)
      chips.appendChild(kchip('neighbours', 'fan-out ' + D.neighbours.join(' × '), 'a-tag',
                              'neighbours sampled per conv block, seeds first'));
    box.appendChild(para(byGraph
      ? 'Each sample is a whole graph' + (D.graph_id ? ' (one ' + D.graph_id + ')' : '')
        + ', its nodes pooled to one answer; its edges never leave it.'
      : 'Each sample is a node, answered from the messages its neighbours send over the edges; '
        + 'the valid nodes are queries against the training graph.', 'a-muted'));
    box.appendChild(chips);
    var transductive = proto === 'transductive';
    if (transductive) {
      var dx = D.exposure || {};
      chips.appendChild(kchip('exposure', 'exposure: test features seen in training',
        'a-warn a-exposure', 'disclosed by ' + (dx.source || 'no recorded source')));
    }
    if (!missing) {
      /* the two views as a small table: what training read, and what the valid queries read */
      var t = M.train_view || {}, v = M.valid_view || {};
      var views = table(['', 'train view', 'valid view'], 'a-views');
      var vb = views.querySelector('tbody');
      [['nodes', 'nodes'], ['edges', 'edges'], ['targets', byGraph ? 'graphs' : 'targets']]
        .forEach(function (r) {
          vb.appendChild(el('tr', null, [el('th', { text: r[1] }),
            el('td', { 'data-k': 'train_' + r[0], text: num(t[r[0]]) }),
            el('td', { 'data-k': 'valid_' + r[0], text: num(v[r[0]]) })]));
        });
      box.appendChild(views);
      var extra = [kstat('test_edges_withheld', 'test edges withheld', num(M.test_edges_withheld),
                         byGraph ? 'inside the test graphs' : 'touching a test node')];
      if (transductive && given((M.exposure || {}).edges_into_test))
        extra.push(kstat('edges_into_test', 'edges into test', num(M.exposure.edges_into_test),
                         'in the training graph'));
      box.appendChild(el('div', { class: 'a-stats', style: GRID }, extra));
    }
    if (transductive) {
      box.appendChild(para('Transductive: the test nodes’ features and edges were in the graph '
        + 'during training, their labels sealed. The evaluation records this exposure, and the '
        + 'network is evaluated but not packed.', 'a-warntext'));
    } else if (!missing) {
      box.appendChild(para('Inductive: training read no test node and no test edge; '
        + num(M.test_edges_withheld) + ' test edge' + (M.test_edges_withheld === 1 ? ' was' : 's were')
        + ' withheld from its view.', 'a-muted'));
    }
  }
  function samplePanel(spec, SM) {
    var D = declaredSample(spec), M = (SM.sample && typeof SM.sample === 'object') ? SM.sample : null;
    var shapeName = D.shape || (M && M.shape) || 'row';
    var box = el('aside', { class: 'a-sample', 'data-view': 'sample', 'data-shape': shapeName,
                            'aria-label': 'The sample: ' + shapeName,
                            style: 'flex:0 1 19rem;min-width:14rem;border:1px solid var(--a-border);'
                              + 'border-radius:6px;background:var(--a-panel);padding:.2rem .8rem .6rem' });
    box.appendChild(el('h4', { text: 'The sample' }));
    box.appendChild(el('div', { class: 'a-chips' }, [kchip('shape', 'shape ' + shapeName, 'a-id')]));
    if (M && M.shape && M.shape !== shapeName)
      box.appendChild(para('The smoke record measured a ' + M.shape + ' sample, but the spec declares a '
        + shapeName + '; the counts below are not this spec’s.', 'a-warntext'));
    var missing = !M;
    if (shapeName === 'window') windowPanel(box, M || {}, D, missing);
    else if (shapeName === 'graph') graphPanel(box, M || {}, D, missing);
    else {
      box.appendChild(para('Each sample is one row of the shaped part: no window is cut and no '
                           + 'graph is read.', 'a-muted'));
      if (SM.rows_train !== undefined)
        box.appendChild(el('div', { class: 'a-stats', style: GRID }, [kstat('rows_train', 'rows · train',
                                                               num(SM.rows_train))]));
    }
    if (missing)
      box.appendChild(para('The smoke record carries no sample block, so the '
        + (shapeName === 'window' ? 'window counts are' : shapeName === 'graph' ? 'view counts are'
           : 'measured sample is') + ' missing; what is shown is the spec’s declaration. '
        + 'emit_backend.py --smoke writes it.', 'a-empty a-sample-missing'));
    return box;
  }

  ready(function () {
    /* the guard above catches a script injected into an already-rendered page; this one catches
       two copies of the script parsed before the document was ready, which both mount otherwise */
    if (document.querySelector('[data-layer="architecture"]')) return;
    var s = document.getElementById('layer-architecture-data');
    var pane = mount();
    pane.appendChild(el('header', { class: 'a-head' }, [el('h2', { text: 'Network' })]));
    var M = null;
    try { M = s ? JSON.parse(s.textContent) : null; } catch (e) { M = null; }
    if (!M) {
      pane.appendChild(para(s ? 'The architecture layer data did not parse; nothing to show.'
                              : 'This file carries an architecture layer marker but no data script.',
                            'a-empty'));
      window.__architecture = { version: VERSION, spec: function () { return null; },
        blocks: function () { return []; }, conformance: function () { return null; },
        smoke: function () { return null; }, sample: function () { return null; } };
      return;
    }
    var SPEC = M.netspec || {}, SM = M.smoke || {}, CONF = M.conformance || {},
        BM = M.blueprint_map || {}, CODE = M.code || {}, READ = M.readings || {},
        BLOCKS = isArr(SPEC.blocks) ? SPEC.blocks : [],
        FINDINGS = isArr(M.code_findings) ? M.code_findings : [],
        FORKS = isArr(M.forks) ? M.forks : [],
        DEF = isArr(M.defaults_to_test) ? M.defaults_to_test : [],
        PER = {};
    (isArr(SM.per_block) ? SM.per_block : []).forEach(function (p) { if (p && p.id) PER[p.id] = p; });

    /* ---------------- the abstract, and the one-line identity of the network */
    var top = el('div', { class: 'a-abstract' });
    var idbar = el('div', { class: 'a-chips' });
    if ((SPEC.family || {}).name) idbar.appendChild(chip((SPEC.family || {}).name, 'a-id'));
    if ((SPEC.task || {}).kind) idbar.appendChild(chip('task ' + SPEC.task.kind, 'a-tag'));
    if ((SPEC.data_contract || {}).input_dim !== undefined)
      idbar.appendChild(chip('input_dim ' + fmt(SPEC.data_contract.input_dim), 'a-tag'));
    if ((SPEC.backend || {}).name)
      idbar.appendChild(chip(SPEC.backend.name + ' ' + fmt(SM.backend_version || SPEC.backend.min_version), 'a-tag'));
    if (SM.params_total !== undefined)
      idbar.appendChild(chip(num(SM.params_total) + ' parameters', 'a-tag'));
    top.appendChild(idbar);
    top.appendChild(READ.abstract ? para(READ.abstract)
      : para('No abstract was recorded for this network.', 'a-empty'));
    pane.appendChild(top);

    /* ---------------- blueprint map */
    var bmv = section(pane, 'blueprint-map', 'The blueprint map');
    var entries = isArr(BM.entries) ? BM.entries : [];
    if (!entries.length) {
      bmv.appendChild(para('No blueprint map was recorded; the emitted files are not traced to '
                           + 'the blueprint’s ports.', 'a-empty'));
    } else {
      var bt = table(['module', 'port', 'file', 'boundary', 'status']);
      var tb = bt.querySelector('tbody');
      entries.forEach(function (e) {
        var st = e.status || (e.kind === 'support' ? 'support' : '');
        var pending = String(st).indexOf('pending') === 0;
        var row = el('tr', { class: 'a-brow' + (pending ? ' a-pending' : '') });
        row.appendChild(el('td', { text: e.module || '·' }));
        row.appendChild(el('td', { text: e.port || (isArr(e.ports) ? e.ports.length + ' ports'
                                   : (isArr(e.artifacts) ? e.artifacts.length + ' artifacts' : '·')) }));
        row.appendChild(el('td', { class: 'a-mono', text: e.file || '·' }));
        row.appendChild(el('td', { class: 'a-mono', text: isArr(e.boundary) ? e.boundary.join(' → ') : '·' }));
        row.appendChild(el('td', null, [chip(st || '·',
          pending ? 'a-warn' : st === 'implemented' ? 'a-ok' : 'a-na',
          pending ? 'this port is not implemented yet' : '')]));
        tb.appendChild(row);
      });
      bt.appendChild(tb);
      bmv.appendChild(bt);
      var pend = entries.filter(function (e) { return String(e.status || '').indexOf('pending') === 0; });
      bmv.appendChild(para(pend.length
        ? pend.length + ' port(s) are still pending — marked above; the rest are implemented in '
          + (CODE.package || BM.package || 'the package') + '.'
        : 'Every port in the map is implemented.', 'a-muted'));
      var un = isArr(BM.unmapped) ? BM.unmapped : [];
      bmv.appendChild(para(un.length ? un.length + ' emitted file(s) map to no port: ' + un.join(', ')
                                     : 'No emitted file is unmapped.', un.length ? 'a-warntext' : 'a-muted'));
      /* the blueprint's artifacts held outside the package, each with its owner (code-contract §3) */
      var ext = isArr(BM.external_artifacts) ? BM.external_artifacts : [];
      if (ext.length) bmv.appendChild(para('Held outside the package: ' + ext.map(function (x) {
        return (x.name || '?') + ' (' + (x.owner || 'no owner') + (x.where ? ', ' + x.where : '') + ')';
      }).join('; ') + '.', 'a-muted'));
    }

    /* ---------------- conformance */
    var cv = section(pane, 'conformance', 'Conformance');
    var crow = el('div', { class: 'a-chips' });
    var mech = CONF.mechanical;
    crow.appendChild(chip('mechanical: ' + (mech === undefined ? 'not run' : String(mech)),
      mech === 'pass' ? 'a-ok' : mech === undefined ? 'a-na' : 'a-bad',
      'the mechanical half of the conformance gate'));
    var arch = CONF.architect;
    if (arch && typeof arch === 'object') {
      ['structural', 'divergence', 'recorded'].forEach(function (k) {
        if (arch[k] === undefined) return;
        /* a confirmed finding is advisory (code-contract §5): a warning and an open item */
        crow.appendChild(chip(k + ': ' + fmt(arch[k]),
          k === 'recorded' ? 'a-na' : (arch[k] === 0 ? 'a-ok' : 'a-warn'),
          k === 'recorded' || arch[k] === 0 ? '' : 'confirmed by two of three audits: advisory, '
            + 'an open item for /architect and /improve'));
      });
    } else {
      /* "not installed" is a gate that did not run: a warning, never a pass */
      crow.appendChild(chip('architect: ' + (arch ? String(arch) : 'not run'), 'a-warn',
        'the judgement half of the gate did not run'));
    }
    cv.appendChild(crow);
    /* the judgement half, per tool: its audits (one clean first audit, or the first and two
       fresh ones), what two of three confirmed (blocking) and what one alone reported (recorded) */
    function judged(name, j) {
      var left = j.unconfirmed || {}, n = isArr(j.audits) ? j.audits.length : 0;
      var adv = j.advisory || {}, items = isArr(adv.open_items) ? adv.open_items : [];
      var p = para(name + ': ' + n + ' audit' + (n === 1 ? '' : 's') + ' from ' + (j.run || '·')
        + ' — confirmed ' + fmt(j.structural) + ' structural, ' + fmt(j.divergence)
        + ' divergence; unconfirmed (recorded, not blocking) ' + fmt(left.structural)
        + ' structural, ' + fmt(left.divergence) + ' divergence; ' + fmt(j.recorded)
        + ' recorded.' + (items.length ? ' Advisory (' + fmt(adv.mode) + ', ' + fmt(adv.outcome)
        + '): ' + items.length + ' open item' + (items.length === 1 ? '' : 's')
        + ' for /architect and /improve.' : ''), items.length ? 'a-warntext' : 'a-muted');
      p.setAttribute('data-judgement', name);
      if (items.length) p.setAttribute('data-open-items', String(items.length));
      return p;
    }
    if (arch && typeof arch === 'object') {
      cv.appendChild(judged('Architect', arch));
    } else {
      cv.appendChild(para('WARNING: the architect skill is ' + (arch ? String(arch) : 'not recorded')
        + ', so the judgement half of the conformance gate did not run. Only the mechanical check '
        + 'stands behind this package; read it as unreviewed, not as a pass.', 'a-warntext'));
    }
    /* the method-level half: /improve --report-only (structural findings block, like architect's) */
    var imp = CONF.improve;
    if (imp && typeof imp === 'object') {
      cv.appendChild(judged('Improve', imp));
    } else {
      cv.appendChild(para('WARNING: the improve skill is ' + (imp ? String(imp) : 'not recorded')
        + ', so the method-level half of the conformance gate did not run.', 'a-warntext'));
    }
    if (!FINDINGS.length) {
      cv.appendChild(para('No code findings were recorded.', 'a-muted'));
    } else {
      var ft = table(['id', 'tool', 'audit', 'depth', 'file', 'anchors', 'finding', 'next']);
      var ftb = ft.querySelector('tbody');
      FINDINGS.forEach(function (f) {
        var anch = el('td');
        (isArr(f.anchors) ? f.anchors : []).forEach(function (a) {
          anch.appendChild(chip(a, 'a-anchor', f.anchor_justification || ''));
        });
        var body = el('td');
        body.appendChild(el('div', { class: 'a-ftitle', text: f.title || '·' }));
        if (f.anchor_justification)
          body.appendChild(el('div', { class: 'a-just', text: f.anchor_justification }));
        if (f.open_item)
          body.appendChild(el('div', { class: 'a-warntext', 'data-open-item': '1',
                                       text: 'Open item: ' + f.open_item }));
        ftb.appendChild(el('tr', { class: 'a-finding', 'data-finding': f.id || '' }, [
          el('td', { class: 'a-mono', text: f.id || '·' }),
          el('td', { class: 'a-mono', text: f.tool || 'architect' }),
          el('td', { class: 'a-mono', text: (f.audit ? String(f.audit) : '·') + (f.quorum
            ? (f.quorum.confirmed ? ' · confirmed by ' : ' · unconfirmed, only ')
              + (isArr(f.quorum.reported_by) ? f.quorum.reported_by.join(', ') : '·') : '') }),
          el('td', null, [chip(f.depth || '·', f.depth === 'architectural' ? 'a-bad' : 'a-na')]),
          el('td', { class: 'a-mono', text: f.file || '·' }),
          anch, body, el('td', { class: 'a-mono', text: f.next || '·' })]));
      });
      cv.appendChild(ft);
    }
    var mk = isArr(CONF.markers) ? CONF.markers.slice() : [];
    (isArr(M.markers) ? M.markers : []).forEach(function (m) { if (mk.indexOf(m) < 0) mk.push(m); });
    if (mk.length) {
      var md = details('run markers (' + mk.length + ')');
      mk.forEach(function (m) { md.appendChild(el('p', { class: 'a-marker', text: m })); });
      cv.appendChild(md);
    }

    /* ---------------- blocks */
    var bv = section(pane, 'blocks', 'The network');
    /* the diagram and, beside it, the sample panel; they wrap one under the other when narrow */
    var split = el('div', { class: 'a-split', style: 'display:flex;flex-wrap:wrap;gap:1rem;'
                                                   + 'align-items:flex-start' });
    var diagram = el('div', { class: 'a-diagram', style: 'flex:1 1 26rem;min-width:0' });
    drawBlocks(diagram, BLOCKS, PER);
    split.appendChild(diagram);
    split.appendChild(samplePanel(SPEC, SM));
    bv.appendChild(split);
    var measured = isArr(SM.per_block) && SM.per_block.length;
    if (BLOCKS.length && !measured) {
      /* the diagram still draws: the blocks are the spec's, only the measurements are missing */
      bv.appendChild(para('The smoke record carries no per-block numbers, so the nodes above show '
        + 'the spec’s blocks without their out-shapes or parameter counts.', 'a-empty'));
    } else if (BLOCKS.length) {
      bv.appendChild(para('One node per top-level block, in the order the forward pass runs them; '
        + 'a composite block shows the inner blocks it repeats. Out-shapes, parameter counts and '
        + 'the sample’s counts beside them are the smoke pass’s, not this page’s.', 'a-muted'));
    }

    /* ---------------- decisions */
    var dv = section(pane, 'decisions', 'Decisions');
    var choices = enumerateChoices(SPEC);
    if (!choices.length) {
      dv.appendChild(para('The spec traces no decisions; nothing to table.', 'a-empty'));
    } else {
      dv.appendChild(para('Every choice carries the source that justifies it. A choice without a '
        + 'source is refused by the compiler, so that column is never empty.', 'a-muted'));
      var byFam = {}, famOrder = [];
      choices.forEach(function (c) {
        if (!byFam[c.family]) { byFam[c.family] = []; famOrder.push(c.family); }
        byFam[c.family].push(c);
      });
      famOrder.forEach(function (fam) {
        var box = el('div', { class: 'a-fam', 'data-family': fam });
        box.appendChild(el('h4', { text: FAM_LABEL[fam] || fam }));
        if ((READ.per_family || {})[fam]) box.appendChild(para(READ.per_family[fam], 'a-famread'));
        var t = table(['choice', 'source', 'rationale', 'alternatives']);
        var tbody = t.querySelector('tbody');
        byFam[fam].forEach(function (c) {
          var srcTd = el('td');
          srcTd.appendChild(chip(c.source || 'unsourced', sourceClass(c.source),
                                 'the decision this choice cites'));
          if (c.catalog) {
            var cat = isArr(c.catalog) ? c.catalog.join(', ') : c.catalog;
            srcTd.appendChild(chip(cat, 'a-cat', 'the catalog row the default cites'));
          }
          var altTd = el('td');
          if (c.alternatives.length) {
            var d = details(c.alternatives.length + ' alternative'
                            + (c.alternatives.length > 1 ? 's' : '') + ' weighed');
            c.alternatives.forEach(function (alt) {
              d.appendChild(para(altText(alt, c.key), 'a-alt'));
            });
            altTd.appendChild(d);
          } else {
            altTd.appendChild(chip('structural', 'a-na',
              'a forced choice: the rationale says what forces it'));
          }
          tbody.appendChild(el('tr', { class: 'a-drow', 'data-choice': c.key }, [
            el('td', { class: 'a-dlabel', text: c.label }), srcTd,
            el('td', { text: c.rationale || '·' }), altTd]));
        });
        box.appendChild(t);
        dv.appendChild(box);
      });
    }

    /* ---------------- mechanics */
    var mv = section(pane, 'mechanics', 'Mechanics');
    var loss = SPEC.loss || {}, opt = SPEC.optimizer || {}, sch = SPEC.schedule || {},
        train = SPEC.training || {};
    var formula = SM.loss_formula || loss.formula;
    mv.appendChild(el('h4', { text: 'Loss' }));
    if (formula) {
      mv.appendChild(el('pre', { class: 'a-formula', text: formula }));
      mv.appendChild(para((loss.name || 'the loss') + ' with '
        + (Object.keys(loss.params || {}).length
           ? Object.keys(loss.params).map(function (k) { return k + ' = ' + fmt(loss.params[k]); }).join(', ')
           : 'no parameters') + '.', 'a-muted'));
    } else {
      mv.appendChild(para('Neither the smoke record nor the spec carries the loss formula.', 'a-empty'));
    }
    mv.appendChild(el('h4', { text: 'Update rule' }));
    if (opt.update_rule) {
      mv.appendChild(el('pre', { class: 'a-formula', text: opt.update_rule }));
      var pchips = el('div', { class: 'a-chips' });
      pchips.appendChild(chip(opt.name || 'optimizer', 'a-id'));
      Object.keys(opt.params || {}).forEach(function (k) {
        pchips.appendChild(chip(k + ' = ' + fmt(opt.params[k]), 'a-tag'));
      });
      mv.appendChild(pchips);
    } else {
      mv.appendChild(para('The spec’s optimizer carries no update rule.', 'a-empty'));
    }
    mv.appendChild(el('h4', { text: 'Schedule' }));
    mv.appendChild(para((sch.name || 'no schedule') + (Object.keys(sch.params || {}).length
      ? ' (' + Object.keys(sch.params).map(function (k) { return k + ' ' + fmt(sch.params[k]); }).join(', ') + ')'
      : '') + ' over up to ' + fmt(train.max_epochs) + ' epochs.', 'a-muted'));
    drawSchedule(mv, SM.schedule_lrs, train.max_epochs);
    if (READ.mechanics) mv.appendChild(para(READ.mechanics));

    /* ---------------- smoke */
    var sv = section(pane, 'smoke', 'The forward pass');
    if (SM.params_total === undefined && SM.forward_ok === undefined) {
      sv.appendChild(para('No smoke record was written; the network was not built.', 'a-empty'));
    } else {
      var madds = SM.madds_per_sample;
      sv.appendChild(statGrid([
        ['parameters', num(SM.params_total), SM.params_trainable !== undefined
          ? num(SM.params_trainable) + ' trainable' : ''],
        ['mult-adds / ' + (UNIT[SM.madds_unit] || 'sample'),
          (SM.madds_lower_bound_only ? '≥ ' : '') + num(madds),
          SM.madds_lower_bound_only ? 'a lower bound: some blocks were not counted' : ''],
        ['forward pass', SM.forward_ok === true ? 'ok' : SM.forward_ok === false ? 'FAILED' : '·'],
        ['out shape', shape(SM.out_shape)],
        ['device', fmt(SM.device)],
        ['backend', fmt(SM.backend_version)],
        ['train rows', num(SM.rows_train)],
        ['batch size', num(SM.batch_size)]]));
      if (SM.forward_ok === false)
        sv.appendChild(para('The forward pass did not succeed; every number above is suspect.',
                            'a-warntext'));
      /* A37: a regression-family head ends on the train target's affine, y = z·scale + shift */
      var ta = (SM.init_notes || {}).target_affine;
      if (ta && isArr(ta.shift) && isArr(ta.scale)) {
        var tbox = el('div', { class: 'a-init', 'data-init': 'target-affine' });
        tbox.appendChild(el('h4', { text: 'The head’s last operation' }));
        tbox.appendChild(para('The ' + (ta.block || 'head') + ' ends on a fixed affine fitted on '
          + 'the train target, one per ' + fmt(ta.per) + ' (' + num(ta.samples) + ' training '
          + 'samples of the train part alone): y = z · scale + shift. The network learns in '
          + 'standardised units; what it answers — and what the loss, the metrics and serving '
          + 'read — is in the target’s own.', 'a-muted'));
        var tchips = el('div', { class: 'a-chips' });
        ta.shift.forEach(function (m, k) {
          tchips.appendChild(chip('× ' + fmt(ta.scale[k]) + ' + ' + fmt(m), 'a-tag a-affine'));
        });
        tbox.appendChild(tchips);
        sv.appendChild(tbox);
      }
    }

    /* ---------------- forks */
    var fv = section(pane, 'forks', 'Forks');
    if (!FORKS.length) {
      fv.appendChild(para('No fork was raised: nothing in the spec was open to a choice the user '
                          + 'had to settle.', 'a-empty'));
    } else {
      var ftab = table(['family', 'asked', 'answer']);
      var ftbody = ftab.querySelector('tbody');
      FORKS.forEach(function (f) {
        var unattended = f.answer === undefined || f.answer === null;
        ftbody.appendChild(el('tr', { class: 'a-fork' + (unattended ? ' a-unattended' : '') }, [
          el('td', { text: f.family || '·' }), el('td', { text: f.asked || '·' }),
          el('td', null, [unattended
            ? chip(f.or || 'default applied (unattended)', 'a-warn',
                   'no one answered this fork; the default stands')
            : chip(String(f.answer), 'a-ok', 'answered')])]));
      });
      fv.appendChild(ftab);
    }

    /* ---------------- defaults to test */
    var dfv = section(pane, 'defaults', 'Defaults the first run must test');
    if (!DEF.length) {
      dfv.appendChild(para('The spec carries no testable default.', 'a-empty'));
    } else {
      var dt = table(['path', 'the observation that would overturn it']);
      var dtb = dt.querySelector('tbody');
      DEF.forEach(function (d) {
        dtb.appendChild(el('tr', { class: 'a-default' }, [
          el('td', { class: 'a-mono', text: d.path || '·' }),
          el('td', { text: d.observation || '·' })]));
      });
      dfv.appendChild(dt);
      dfv.appendChild(para('Each row is a default that stands only until the first run says '
        + 'otherwise; the ablation ladder starts here.', 'a-muted'));
    }

    window.__architecture = {
      version: VERSION,
      spec: function () { return SPEC; },
      blocks: function () { return BLOCKS.slice(); },
      conformance: function () { return CONF; },
      smoke: function () { return SM; },
      /* the sample as the spec declares it and as the smoke pass measured it (null: not recorded) */
      sample: function () {
        return { declared: declaredSample(SPEC), measured: SM.sample || null };
      }
    };
  });
})();
