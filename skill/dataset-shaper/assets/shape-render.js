/* dataset-shaper — `shape` layer render script (contract: references/shape-contract.md §4).
   Vanilla JS, no libraries, no network, inline SVG only. Reads ONLY #layer-shape-data.
   Mounts one tab (data-layer="shape") with four surfaces:
     Before / after — what the dataset became, in one strip.
     Recipe — one row per step with the decision it cites, its rationale, its consequences and
       the alternatives that were weighed; the source chip is the provenance, always visible.
     Lineage — input columns, the steps that touched them, output columns; a dropped column
       ends at the step that dropped it, with the reason.
     Verification — the structural, semantic, distributional, split and determinism results,
       with the run's markers, and the forks with what was asked and what answered them.
   It computes nothing: every number here was written by the executor. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="shape"]')) return;

  var VERSION = 'dataset-shaper/shape-render@1';
  var SVGNS = 'http://www.w3.org/2000/svg';

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
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v === 'number') {
      if (v === Math.round(v)) return String(v);
      return String(Math.round(v * 1e4) / 1e4);
    }
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }
  function chip(text, cls, title) {
    var c = el('span', { class: 's-chip ' + (cls || ''), text: text });
    if (title) c.setAttribute('title', title);
    return c;
  }
  function para(t, cls) { return el('p', { class: cls || 's-prose', text: t }); }
  function details(summary, cls) {
    var d = el('details', { class: cls || 's-det' });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }
  function sourceClass(src) {
    src = String(src || '');
    if (src.indexOf('geometry:') === 0) return 's-src-geometry';
    if (src.indexOf('analysis:') === 0) return 's-src-analysis';
    if (src.indexOf('user:') === 0) return 's-src-user';
    return 's-src-default';
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
    var pane = el('section', { class: 'layer-shape tab-pane', 'data-layer': 'shape',
                               'data-tab': 'shape', id: 'tab-shape', role: 'tabpanel',
                               'aria-label': 'Shape' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'shape', 'data-layer': 'shape', role: 'tab',
                             text: 'Shape' });
      b.onclick = function () { switchTab('shape'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  ready(function () {
    var s = document.getElementById('layer-shape-data');
    var pane = mount();
    pane.appendChild(el('header', { class: 's-head' }, [el('h2', { text: 'Shape' })]));
    var M = null;
    try { M = s ? JSON.parse(s.textContent) : null; } catch (e) { M = null; }
    if (!M) {
      pane.appendChild(para(s ? 'The shape layer data did not parse; nothing to show.'
                              : 'This file carries a shape layer marker but no data script.',
                            's-empty'));
      return;
    }
    var R = M.recipe || {}, man = M.manifest || {}, V = M.verification || {},
        BA = M.before_after || {}, LIN = M.lineage || {}, steps = R.steps || [];

    /* ---------------- before / after */
    var strip = el('div', { class: 's-strip', 'data-view': 'before-after' });
    strip.appendChild(el('h3', { text: 'What the dataset became' }));
    var grid = el('div', { class: 's-ba' });
    function stat(label, value, note) {
      return el('div', { class: 's-stat' }, [
        el('div', { class: 's-statv', text: fmt(value) }),
        el('div', { class: 's-statl', text: label }),
        note ? el('div', { class: 's-statn', text: note }) : null]);
    }
    var rows = BA.rows || [], colsBA = BA.columns || [];
    grid.appendChild(stat('rows', (rows[0] !== undefined ? rows[0] + ' → ' + rows[1] : '·')));
    grid.appendChild(stat('columns', (colsBA[0] !== undefined ? colsBA[0] + ' → ' + colsBA[1] : '·')));
    if (BA.basis_kept !== undefined) grid.appendChild(stat('basis kept', BA.basis_kept));
    if (BA.derived_dropped !== undefined) grid.appendChild(stat('derived dropped', BA.derived_dropped));
    if (BA.leakage_dropped !== undefined) grid.appendChild(stat('leakage dropped', BA.leakage_dropped));
    if ((BA.added || []).length) grid.appendChild(stat('added', BA.added.length,
                                                        BA.added.slice(0, 6).join(', ')));
    strip.appendChild(grid);
    var files = ((man.outputs || {}).files) || {};
    var fkeys = Object.keys(files);
    if (fkeys.length) {
      var ft = el('table', { class: 's-table' });
      var th = el('tr');
      ['part', 'file', 'rows', 'columns', 'digest'].forEach(function (h) {
        th.appendChild(el('th', { text: h })); });
      ft.appendChild(el('thead', null, [th]));
      var tb = el('tbody');
      fkeys.forEach(function (k) {
        var f = files[k];
        tb.appendChild(el('tr', null, [
          el('td', { text: k }), el('td', { class: 's-mono', text: (f.path || '').split('/').pop() }),
          el('td', { text: fmt(f.rows) }), el('td', { text: fmt(f.columns) }),
          el('td', { class: 's-mono s-dig', text: (f.digest || '·').slice(0, 23) + '…' })]));
      });
      ft.appendChild(tb);
      strip.appendChild(ft);
    }
    if ((M.readings || {}).abstract) strip.appendChild(para(M.readings.abstract));
    pane.appendChild(strip);

    /* ---------------- recipe */
    var rec = el('div', { class: 's-recipe', 'data-view': 'recipe' });
    rec.appendChild(el('h3', { text: 'The recipe' }));
    rec.appendChild(para('Every step cites the decision that justifies it. A step with no source '
                         + 'is refused by the executor, so this column is never empty.', 's-muted'));
    var byPhase = {};
    var PHASE = { retype: 1, drop_identity: 1, drop_constant: 1, dedupe: 1, parse_datetime: 1,
      parse_geometry: 1, orient_cycle: 2, drop_derived: 2, select_partition: 2, drop_leakage: 2,
      keep_columns: 2, split: 3, impute: 4, clip: 4, winsorize: 4, transform: 4, bin: 4,
      datetime_expand: 4, lag: 4, spatial_reproject: 4, spatial_distance: 4, spatial_join: 4,
      spatial_grid: 4, spatial_features: 4, encode: 5, scale: 5, project: 5, select_features: 6 };
    var NAMES = { 1: 'typing and structure', 2: 'geometry', 3: 'split', 4: 'values',
                  5: 'representation', 6: 'selection' };
    var manSteps = {};
    (man.steps || []).forEach(function (m) { manSteps[m.id] = m; });
    steps.forEach(function (st) {
      var ph = PHASE[st.op] || ((st.params || {}).phase) || 4;
      (byPhase[ph] = byPhase[ph] || []).push(st);
    });
    Object.keys(byPhase).sort().forEach(function (ph) {
      var box = el('div', { class: 's-phase', 'data-phase': ph });
      box.appendChild(el('h4', { text: 'Phase ' + ph + ' — ' + (NAMES[ph] || '') }));
      byPhase[ph].forEach(function (st) {
        var m = manSteps[st.id] || {};
        var row = el('article', { class: 's-step', 'data-step': st.id });
        var head = el('div', { class: 's-stephead' }, [
          chip(st.id, 's-id'), el('span', { class: 's-op', text: st.op }),
          el('span', { class: 's-cols s-mono', text: (st.columns || []).join(', ') }),
          chip(st.source || '—', sourceClass(st.source), 'the decision this step cites')]);
        if (m.fit_on) head.appendChild(chip('fitted on ' + m.fit_on, 's-fit',
          'a fitted step is fitted on the training part alone'));
        if (st.reversible === false) head.appendChild(chip('not reversible', 's-irrev'));
        row.appendChild(head);
        row.appendChild(para(st.rationale || '', 's-rat'));
        if (st.params && Object.keys(st.params).length)
          row.appendChild(el('pre', { class: 's-params', text: JSON.stringify(st.params) }));
        var c = st.consequences || {};
        if (c.rows || c.columns || c.downstream) {
          var dl = el('dl', { class: 's-cons' });
          if (c.rows) { dl.appendChild(el('dt', { text: 'rows' })); dl.appendChild(el('dd', { text: String(c.rows) })); }
          if (c.columns) { dl.appendChild(el('dt', { text: 'columns' })); dl.appendChild(el('dd', { text: String(c.columns) })); }
          Object.keys(c.downstream || {}).forEach(function (k) {
            dl.appendChild(el('dt', { text: k.replace(/_/g, ' ') }));
            dl.appendChild(el('dd', { text: String(c.downstream[k]) }));
          });
          row.appendChild(dl);
        }
        if ((st.alternatives || []).length) {
          var d = details((st.alternatives.length) + ' alternative weighed');
          st.alternatives.forEach(function (alt) {
            d.appendChild(para((alt.op || st.op) + ' ' + JSON.stringify(alt.params || {}) +
                               (alt.when ? ' — when ' + alt.when : ''), 's-alt'));
          });
          row.appendChild(d);
        }
        if (m.columns_added && (m.columns_added.length || (m.columns_removed || []).length)) {
          row.appendChild(el('p', { class: 's-delta s-mono', text:
            (m.columns_added.length ? '+' + m.columns_added.join(', ') : '') +
            (m.columns_removed && m.columns_removed.length ? '  −' + m.columns_removed.join(', ') : '') }));
        }
        box.appendChild(row);
      });
      rec.appendChild(box);
    });
    pane.appendChild(rec);

    /* ---------------- lineage */
    var lin = el('div', { class: 's-lineage', 'data-view': 'lineage' });
    lin.appendChild(el('h3', { text: 'Lineage' }));
    var colsL = LIN.columns || {}, removed = LIN.removed || {};
    var names = Object.keys(colsL);
    if (!names.length) {
      lin.appendChild(para('No lineage was recorded.', 's-empty'));
    } else {
      var W = 720, rowH = 18, pad = 8;
      var H = pad * 2 + rowH * names.length;
      var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 's-svg', role: 'img',
                           'aria-label': 'Column lineage: which steps touched each column and '
                                         + 'where a dropped column ends' });
      names.sort(function (a, b) {
        var pa = colsL[a].present_in_output ? 0 : 1, pb = colsL[b].present_in_output ? 0 : 1;
        return pa - pb || a.localeCompare(b);
      });
      names.forEach(function (n, i) {
        var y = pad + i * rowH + rowH / 2;
        var info = colsL[n] || {};
        var touched = info.touched_by || [];
        var alive = info.present_in_output;
        g.appendChild(svg('text', { x: 4, y: y + 4, class: 's-lname' + (alive ? '' : ' s-dead'),
                                    text: n }));
        var x0 = 170, x1 = alive ? W - 130 : x0 + Math.max(40, touched.length * 46);
        g.appendChild(svg('line', { x1: x0, y1: y, x2: x1, y2: y,
                                    class: 's-lline' + (alive ? '' : ' s-dead') }));
        touched.forEach(function (sid, j) {
          var x = x0 + 24 + j * 46;
          if (x > x1 - 8) return;
          g.appendChild(svg('circle', { cx: x, cy: y, r: 4.5, class: 's-lnode' }));
          g.appendChild(svg('text', { x: x, y: y - 7, class: 's-lstep', 'text-anchor': 'middle',
                                      text: sid }));
        });
        if (alive) {
          g.appendChild(svg('text', { x: W - 124, y: y + 4, class: 's-lout', text: 'output' }));
        } else {
          var r = removed[n] || {};
          g.appendChild(svg('text', { x: x1 + 6, y: y + 4, class: 's-ldrop',
                                      text: '✕ ' + (r.step || '') + ' — ' + (r.why || 'dropped') }));
        }
      });
      lin.appendChild(g);
      lin.appendChild(para('A line that stops is a column that left, with the step that removed '
                           + 'it and why. A line that reaches the right edge is in the output.',
                           's-muted'));
    }
    pane.appendChild(lin);

    /* ---------------- verification */
    var ver = el('div', { class: 's-verify', 'data-view': 'verification' });
    ver.appendChild(el('h3', { text: 'Verification' }));
    var vrow = el('div', { class: 's-chips' });
    ['structural', 'split', 'determinism', 'spatial'].forEach(function (k) {
      if (V[k] === undefined) return;
      var v = String(V[k]);
      vrow.appendChild(chip(k + ': ' + v, v === 'pass' ? 's-ok' : v === 'fail' ? 's-bad' : 's-na'));
    });
    ver.appendChild(vrow);
    if ((V.semantic || []).length) {
      var t = el('table', { class: 's-table' });
      var hr = el('tr');
      ['rule', 'column', 'empirical', 'symbolic', 'detail'].forEach(function (h) {
        hr.appendChild(el('th', { text: h })); });
      t.appendChild(el('thead', null, [hr]));
      var tb2 = el('tbody');
      V.semantic.forEach(function (r) {
        tb2.appendChild(el('tr', null, [
          el('td', { class: 's-mono', text: r.rule_id || '' }),
          el('td', { class: 's-mono', text: r.column || '' }),
          el('td', null, [chip(r.empirical || '·', r.empirical === 'confirmed' ? 's-ok'
            : r.empirical === 'refuted' ? 's-bad' : 's-na')]),
          el('td', null, [chip(r.symbolic || 'untested',
            r.symbolic === 'confirmed' ? 's-ok' : r.symbolic === 'refuted' ? 's-bad' : 's-na')]),
          el('td', { text: r.rows || r.why || (r.determination_ratio !== undefined
            ? 'determination ' + fmt(r.determination_ratio) : '') })]));
      });
      t.appendChild(tb2);
      ver.appendChild(t);
    }
    var drift = (V.distributional || []).filter(function (r) { return r.expected === false; });
    if (drift.length) {
      var dd = details(drift.length + ' column(s) moved although no step names them');
      drift.forEach(function (r) {
        dd.appendChild(para(r.column + ': PSI ' + fmt(r.psi) + ' — suspect ' +
                            (r.suspected_steps || []).join(', '), 's-alt'));
      });
      ver.appendChild(dd);
    } else if ((V.distributional || []).length) {
      ver.appendChild(para((V.distributional.length) + ' untouched column(s) did not move.',
                           's-muted'));
    }
    if ((M.forks || []).length) {
      ver.appendChild(el('h4', { text: 'Forks' }));
      M.forks.forEach(function (f) {
        ver.appendChild(para((f.step ? f.step + ' — ' : '') + (f.asked || '') + ' → ' +
                             (f.answer || f.or || 'unanswered'), 's-fork'));
      });
    }
    if ((V.markers || []).length) {
      var mk = details('run markers (' + V.markers.length + ')');
      V.markers.forEach(function (m2) { mk.appendChild(el('p', { class: 's-marker', text: m2 })); });
      ver.appendChild(mk);
    }
    if (M.readings && M.readings.per_phase) {
      var pp = details('reading, phase by phase');
      Object.keys(M.readings.per_phase).forEach(function (k) {
        if (!M.readings.per_phase[k]) return;
        pp.appendChild(el('h5', { text: k }));
        pp.appendChild(para(M.readings.per_phase[k]));
      });
      ver.appendChild(pp);
    }
    pane.appendChild(ver);

    window.__shape = {
      version: VERSION,
      steps: function () { return steps.slice(); },
      lineage: function () { return LIN; },
      verification: function () { return V; }
    };
  });
})();
