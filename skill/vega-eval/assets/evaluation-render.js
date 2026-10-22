/* /vega-eval — `evaluation` layer render script (contract: references/eval-contract.md §6).
   Vanilla JS, no libraries, no network, inline SVG only. Reads ONLY #layer-evaluation-data.
   Mounts one tab (data-layer="evaluation", labelled Evaluation) with six surfaces:
     seal        — the run, when the test part was opened, and how many times. It is first
       because it qualifies everything below it: a number from exposure three is not the same
       number as one from exposure one, and a reader who scrolls past the seal has already
       misread the page.
     verdict     — earned or not earned, against the named strongest fitted baseline, with the
       paired difference and its interval, then the five readings of §5.
     metrics     — the network and every baseline, point and CI side by side; the paired
       difference marked when its interval excludes zero, which is the only place on this page
       where a comparison becomes a claim.
     curves      — ROC, PR, reliability, the score histogram and the confusion matrix, drawn
       from the points the evaluation stored. Nothing is recomputed here.
     slices      — sortable by the primary metric, with n on every row. A slice that holds one
       class is shown greyed with its note rather than as a score of zero.
     robustness  — each check with its interval, and each skip with the reason it was skipped,
       because a check that silently vanished reads as a check that passed.
   It computes nothing: every number here was written by evaluate.py. Rounding for display and
   choosing an axis are the only arithmetic in this file. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="evaluation"]')) return;

  var VERSION = 'vega-eval/evaluation-render@1';
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
  function isArr(v) { return Object.prototype.toString.call(v) === '[object Array]'; }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v === 'number') {
      if (!isFinite(v)) return String(v);
      if (v === Math.round(v)) return String(v);
      if (Math.abs(v) < 1e-3) return v.toExponential(2);
      return String(Math.round(v * 1e6) / 1e6);
    }
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }
  function para(t, cls) { return el('p', { class: cls || 'ev-prose', text: t }); }
  function details(summary, cls) {
    var d = el('details', { class: cls || '' });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }
  function table(headers, cls) {
    var t = el('table', { class: cls || '' });
    var hr = el('tr');
    headers.forEach(function (h) {
      hr.appendChild(el('th', typeof h === 'string' ? { text: h }
                                                    : { text: h.text, class: h.cls || '' }));
    });
    t.appendChild(el('thead', null, [hr]));
    t.appendChild(el('tbody'));
    return t;
  }

  /* ---------------- an estimate, shown as the record wrote it. A missing interval is said,
     never filled in: `Estimate` only omits one when the number could not be resampled, and a
     bare point on a held-out part is exactly the reading this stage refuses to publish. */
  function ciText(e) {
    if (!e || typeof e !== 'object') return null;
    var ci = isArr(e.ci) ? e.ci : null;
    if (ci && ci.length === 2 && isNum(ci[0]) && isNum(ci[1]))
      return '[' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']';
    return null;
  }
  /* `ev-est` marks a cell that holds a *measured* quantity, as opposed to a count (a slice's
     n, a confusion cell). The smoke test uses it to assert the property that matters: every
     estimate the page renders shows its interval beside it. */
  function estCell(e, cls) {
    var td = el('td', { class: 'ev-num ev-est ' + (cls || '') });
    if (!e || typeof e !== 'object' || e.value === null || e.value === undefined) {
      /* a per-element reading has a vector where a scalar metric has a point; say how many
         elements it holds rather than showing the same · as a metric that could not be
         computed at all — they are different facts */
      if (e && isArr(e.values) && e.values.length) {
        td.appendChild(el('span', { class: 'ev-none',
                                    text: e.values.length + ' values' }));
        td.setAttribute('title', (e.note ? e.note + ' — ' : '')
                                 + e.values.map(fmt).join(', '));
        return td;
      }
      td.appendChild(el('span', { class: 'ev-none', text: '·' }));
      if (e && e.note) td.setAttribute('title', e.note);
      return td;
    }
    td.appendChild(document.createTextNode(fmt(e.value)));
    var ci = ciText(e);
    if (ci) td.appendChild(el('span', { class: 'ev-ci', text: ' ' + ci }));
    else if (e._upstream)
      td.appendChild(el('span', { class: 'ev-ci ev-upstream', text: ' (upstream)' }));
    if (e.note) td.setAttribute('title', e.note);
    if (isNum(e.n)) td.setAttribute('data-n', e.n);
    td.setAttribute('data-value', e.value);
    return td;
  }
  /* a difference is a claim only when its interval excludes zero; the class says which way */
  function diffCell(e, mode) {
    var td = estCell(e, '');
    var ci = isArr(e && e.ci) ? e.ci : null;
    if (ci && ci.length === 2 && isNum(ci[0]) && isNum(ci[1])) {
      if (ci[0] > 0) td.className += ' ev-excludes-zero ' + (mode === 'min' ? 'ev-down' : 'ev-up');
      else if (ci[1] < 0)
        td.className += ' ev-excludes-zero ' + (mode === 'min' ? 'ev-up' : 'ev-down');
    }
    return td;
  }

  /* ---------------- SVG. The curves carry their points; this draws them and nothing else. */
  function axes(g, w, h, pad, xlabel, ylabel) {
    g.appendChild(svg('line', { x1: pad, y1: h - pad, x2: w - 6, y2: h - pad,
                                stroke: 'currentColor', 'stroke-width': 1, opacity: .4 }));
    g.appendChild(svg('line', { x1: pad, y1: 6, x2: pad, y2: h - pad,
                                stroke: 'currentColor', 'stroke-width': 1, opacity: .4 }));
    if (xlabel) g.appendChild(svg('text', { x: (w + pad) / 2, y: h - 4, 'text-anchor': 'middle',
                                            'font-size': 9, fill: 'currentColor', opacity: .7,
                                            text: xlabel }));
    if (ylabel) g.appendChild(svg('text', { x: 10, y: 12, 'font-size': 9, fill: 'currentColor',
                                            opacity: .7, text: ylabel }));
  }
  function lineChart(points, opts) {
    opts = opts || {};
    var w = 360, h = 250, pad = 34;
    var node = svg('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img',
                            'aria-label': opts.label || 'curve',
                            preserveAspectRatio: 'xMidYMid meet' });
    var pts = (points || []).filter(function (p) {
      return isArr(p) ? (isNum(p[0]) && isNum(p[1])) : (p && isNum(p.x) && isNum(p.y));
    }).map(function (p) { return isArr(p) ? { x: p[0], y: p[1] } : p; });
    if (!pts.length) {
      node.appendChild(svg('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle',
                                     'font-size': 11, fill: 'currentColor', opacity: .6,
                                     text: 'no points recorded' }));
      return node;
    }
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var x0 = isNum(opts.x0) ? opts.x0 : Math.min.apply(null, xs);
    var x1 = isNum(opts.x1) ? opts.x1 : Math.max.apply(null, xs);
    var y0 = isNum(opts.y0) ? opts.y0 : Math.min.apply(null, ys);
    var y1 = isNum(opts.y1) ? opts.y1 : Math.max.apply(null, ys);
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) y1 = y0 + 1;
    function sx(v) { return pad + (v - x0) / (x1 - x0) * (w - pad - 8); }
    function sy(v) { return (h - pad) - (v - y0) / (y1 - y0) * (h - pad - 10); }
    axes(node, w, h, pad, opts.xlabel, opts.ylabel);
    if (opts.diagonal)
      node.appendChild(svg('line', { x1: sx(x0), y1: sy(y0), x2: sx(x1), y2: sy(y1),
                                     stroke: 'currentColor', opacity: .3,
                                     'stroke-dasharray': '3 3' }));
    node.appendChild(svg('path', {
      d: pts.map(function (p, i) { return (i ? 'L' : 'M') + sx(p.x).toFixed(1) + ' '
                                              + sy(p.y).toFixed(1); }).join(' '),
      fill: 'none', stroke: 'currentColor', 'stroke-width': 1.7 }));
    if (opts.dots) pts.forEach(function (p) {
      node.appendChild(svg('circle', { cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r: 2.2,
                                       fill: 'currentColor' }));
    });
    [[x0, sx(x0)], [x1, sx(x1)]].forEach(function (t) {
      node.appendChild(svg('text', { x: t[1], y: h - pad + 11, 'text-anchor': 'middle',
                                     'font-size': 9, fill: 'currentColor', opacity: .7,
                                     text: fmt(t[0]) }));
    });
    [[y0, sy(y0)], [y1, sy(y1)]].forEach(function (t) {
      node.appendChild(svg('text', { x: pad - 4, y: t[1] + 3, 'text-anchor': 'end',
                                     'font-size': 9, fill: 'currentColor', opacity: .7,
                                     text: fmt(t[0]) }));
    });
    return node;
  }
  function barChart(bars, opts) {
    opts = opts || {};
    var w = 360, h = 250, pad = 34;
    var node = svg('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img',
                            'aria-label': opts.label || 'histogram',
                            preserveAspectRatio: 'xMidYMid meet' });
    var vals = bars.filter(function (b) { return isNum(b.v); });
    if (!vals.length) return node;
    var top = Math.max.apply(null, vals.map(function (b) { return b.v; })) || 1;
    var bw = (w - pad - 8) / vals.length;
    axes(node, w, h, pad, opts.xlabel, opts.ylabel);
    vals.forEach(function (b, i) {
      var height = (b.v / top) * (h - pad - 10);
      node.appendChild(svg('rect', { x: (pad + i * bw + .5).toFixed(1),
                                     y: (h - pad - height).toFixed(1),
                                     width: Math.max(1, bw - 1).toFixed(1),
                                     height: Math.max(0, height).toFixed(1),
                                     fill: 'currentColor',
                                     opacity: b.alt ? .38 : .72 },
                           [svg('title', { text: (b.k === undefined ? '' : fmt(b.k) + ': ')
                                                 + fmt(b.v) })]));
    });
    return node;
  }
  function figure(title, node, note) {
    var f = el('figure', { class: 'ev-curve' });
    f.appendChild(node);
    f.appendChild(el('figcaption', { text: title + (note ? ' — ' + note : '') }));
    return f;
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
    var pane = el('section', { class: 'layer-evaluation tab-pane', 'data-layer': 'evaluation',
                               'data-tab': 'evaluation', id: 'tab-evaluation',
                               role: 'tabpanel', 'aria-label': 'Evaluation' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'evaluation', 'data-layer': 'evaluation', role: 'tab',
                             text: 'Evaluation' });
      b.onclick = function () { switchTab('evaluation'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  /* ---------------- a sortable table head. Sorting is a reading aid; it never reorders the
     record, and the arrow says which column and which direction the reader chose. */
  function makeSortable(t, numericFrom) {
    var ths = t.querySelectorAll('thead th');
    Array.prototype.forEach.call(ths, function (th, i) {
      th.setAttribute('aria-sort', 'none');
      th.setAttribute('tabindex', '0');
      th.setAttribute('role', 'columnheader');
      function sort() {
        var dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
        Array.prototype.forEach.call(ths, function (o) { o.setAttribute('aria-sort', 'none'); });
        th.setAttribute('aria-sort', dir);
        var body = t.querySelector('tbody');
        var rows = Array.prototype.slice.call(body.querySelectorAll('tr'));
        rows.sort(function (a, b) {
          var ca = a.children[i], cb = b.children[i];
          var va = ca && ca.getAttribute('data-value'), vb = cb && cb.getAttribute('data-value');
          var na = va === null ? NaN : parseFloat(va), nb = vb === null ? NaN : parseFloat(vb);
          if (i >= numericFrom && (isFinite(na) || isFinite(nb))) {
            if (!isFinite(na)) return 1;          /* a row without a number sinks either way */
            if (!isFinite(nb)) return -1;
            return dir === 'ascending' ? na - nb : nb - na;
          }
          var sa = (ca ? ca.textContent : ''), sb = (cb ? cb.textContent : '');
          return dir === 'ascending' ? sa.localeCompare(sb) : sb.localeCompare(sa);
        });
        rows.forEach(function (r) { body.appendChild(r); });
      }
      th.onclick = sort;
      th.onkeydown = function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); }
      };
    });
  }

  ready(function () {
    if (document.querySelector('[data-layer="evaluation"]')) return;
    var s = document.getElementById('layer-evaluation-data');
    var pane = mount();
    pane.appendChild(el('header', null, [el('h2', { text: 'Evaluation' })]));
    var M = null;
    try { M = s ? JSON.parse(s.textContent) : null; } catch (e) { M = null; }
    if (!M || typeof M !== 'object') {
      pane.appendChild(para(s ? 'The evaluation layer data did not parse; there is no reading '
                              + 'to show. The evaluation directory on disk is still the record.'
                              : 'This file carries an evaluation layer marker but no data '
                              + 'script.', 'ev-empty'));
      window.__evaluation = { version: VERSION, metrics: function () { return null; },
                              slices: function () { return []; } };
      return;
    }

    var SEAL = M.seal || {}, RUN = M.run || {}, VERD = M.verdict || {},
        MET = M.metrics || {}, THR = M.threshold || {}, BASE = M.baselines || {},
        SLI = M.slices || {}, ROB = M.robustness || {}, CUR = (M.curves || {}).curves
              || M.curves || {},
        NB = M.notebook || {}, READ = M.readings || {},
        MARK = isArr(M.markers) ? M.markers : [];
    var PRIMARY = MET.primary || 'primary';
    var MODE = MET.mode || 'max';
    var MODELS = MET.models || {};
    var DIFFS = MET.differences || {};
    var BROWS = isArr(BASE.baselines) ? BASE.baselines : [];
    var SROWS = isArr(SLI.slices) ? SLI.slices : [];
    var CHECKS = isArr(ROB.checks) ? ROB.checks : [];
    var EXPOSURES = isNum(SEAL.exposures) ? SEAL.exposures : 1;

    /* ---- the seal strip */
    var strip = el('dl', { class: 'ev-seal' + (EXPOSURES > 1 ? ' ev-reopened' : ''),
                           'data-view': 'seal' });
    function pair(label, value, title) {
      var d = el('div', { class: 'ev-pair' }, [el('dt', { text: label }),
                                               el('dd', { text: fmt(value) })]);
      if (title) d.setAttribute('title', title);
      strip.appendChild(d);
    }
    pair('run', RUN.id);
    pair('test rows', MET.n_test);
    pair('opened', SEAL.opened_at);
    pair('exposures', EXPOSURES, EXPOSURES > 1
         ? 'every number on this page was read after the test part had already been seen'
         : 'the test part was opened once, for this evaluation');
    pair('bootstrap', fmt(MET.n_boot) + ' @ seed ' + fmt(MET.boot_seed),
         'every model is scored on the same resamples, so the differences have intervals');
    /* three states: matched, mismatched, and never compared because the shaper's manifest
       recorded no digest. The third is not silence — a reader must not take a blank strip for
       a passed check. */
    var matched = (SEAL.test || {}).digest_matches_manifest;
    if (matched === false) pair('digest', 'MISMATCH',
      'the test file does not hash to what the shaper recorded');
    else if (matched === null || matched === undefined) pair('digest', 'not checked',
      'the shaper\u2019s manifest recorded no digest for the test part, so this file was ' +
      'never compared against it');
    pane.appendChild(strip);
    if (EXPOSURES > 1) {
      var hist = details('the seal’s history — ' + EXPOSURES + ' exposures');
      var ht = table(['at', 'run', 'gate', 'reason']);
      (isArr(SEAL.history) ? SEAL.history : []).forEach(function (h) {
        ht.querySelector('tbody').appendChild(el('tr', null, [
          el('td', { text: fmt(h.at) }), el('td', { text: fmt(h.run) }),
          el('td', { text: fmt(h.gate) }), el('td', { text: fmt(h.reason) })]));
      });
      hist.appendChild(ht);
      hist.appendChild(para('A held-out part measures generalisation once. Each further look '
        + 'spends some of that: the count is here so the verdict can be read against it, not '
        + 'to be corrected away.', 'ev-skip'));
      pane.appendChild(hist);
    }

    /* ---- the verdict: earned, not earned, or — `earned` null — not assessable, which is
       said in words and drawn as neither a win nor a loss (eval-contract §5) */
    var earned = !!VERD.earned;
    var assessable = VERD.earned !== null;
    var vbox = el('section', { class: 'ev-verdict' + (earned ? ' ev-earned' : '')
                                      + (assessable ? '' : ' ev-not-assessable'),
                               'data-view': 'verdict' });
    vbox.appendChild(assessable ? el('p', null, [
      el('span', { class: 'ev-flag', text: earned ? 'Earned' : 'Not earned' }),
      document.createTextNode(' — against ' + fmt(VERD.strongest_baseline)
        + ', the paired difference on ' + PRIMARY + ' is ' + fmt(VERD.diff)
        + (isArr(VERD.ci) && VERD.ci.length === 2
           ? ' [' + fmt(VERD.ci[0]) + ', ' + fmt(VERD.ci[1]) + ']' : '')
        + (earned ? ', and the interval excludes zero.'
                  : ', and the interval does not exclude zero.'))])
      : el('p', null, [
      el('span', { class: 'ev-flag', text: 'Not assessable' }),
      document.createTextNode(' — no paired difference on ' + PRIMARY + ' was computed: '
        + fmt(VERD.reason))]));
    if (READ.verdict) vbox.appendChild(para(READ.verdict));
    pane.appendChild(vbox);

    /* ---- the metrics table */
    var names = [];
    (function () {
      var seen = {};
      Object.keys(MODELS).forEach(function (m) {
        Object.keys(MODELS[m] || {}).forEach(function (k) {
          if (seen[k]) return;
          seen[k] = 1; names.push(k);
        });
      });
      names.sort(function (a, b) { return a === PRIMARY ? -1 : b === PRIMARY ? 1 : 0; });
    })();
    var order = ['network'].concat(Object.keys(MODELS).filter(function (m) {
      return m !== 'network';
    }));
    var mt = table(['metric'].concat(order.map(function (m) {
      return { text: m, cls: 'ev-num' };
    })).concat([{ text: 'net − ' + fmt(VERD.strongest_baseline), cls: 'ev-num' }]), 'ev-metrics');
    names.forEach(function (metric) {
      var tr = el('tr', metric === PRIMARY ? { class: 'ev-network' } : null,
                  [el('th', { scope: 'row', text: metric
                              + (metric === PRIMARY ? ' (primary)' : '') })]);
      order.forEach(function (m) { tr.appendChild(estCell((MODELS[m] || {})[metric])); });
      tr.appendChild(metric === PRIMARY && VERD.strongest_baseline
                     ? diffCell(DIFFS[VERD.strongest_baseline], MODE)
                     : el('td', { class: 'ev-num ev-none', text: '·' }));
      mt.querySelector('tbody').appendChild(tr);
    });
    pane.appendChild(el('section', { 'data-view': 'metrics' },
                        [el('h4', { text: 'Metrics on the test part' }), mt]));
    if (Object.keys(DIFFS).length) {
      var dt = table(['baseline', { text: 'network − baseline (' + PRIMARY + ')',
                                    cls: 'ev-num' }, 'reads as']);
      Object.keys(DIFFS).forEach(function (name) {
        var e = DIFFS[name], ci = isArr(e.ci) ? e.ci : null;
        var says = (ci && ci.length === 2 && isNum(ci[0]) && isNum(ci[1]))
          ? ((MODE === 'max' ? ci[0] > 0 : ci[1] < 0) ? 'the network is better'
             : (MODE === 'max' ? ci[1] < 0 : ci[0] > 0) ? 'the baseline is better'
             : 'the interval covers zero — no difference is shown')
          : 'no interval, so no comparison';
        dt.querySelector('tbody').appendChild(el('tr', null, [
          el('th', { scope: 'row', text: name }), diffCell(e, MODE),
          el('td', { text: says })]));
      });
      pane.appendChild(el('section', { 'data-view': 'differences' }, [
        el('h4', { text: 'Paired differences' }), dt,
        para('Each difference is measured resample by resample on the same draws, which is '
             + 'what gives it an interval of its own. Two separately drawn intervals overlap '
             + 'or do not, and neither says whether one model beat the other.', 'ev-skip')]));
    }

    /* ---- the baselines */
    if (BROWS.length) {
      var bt = table(['baseline', { text: 'valid', cls: 'ev-num' },
                      { text: 'test', cls: 'ev-num' }, 'what it is']);
      BROWS.forEach(function (b) {
        var tr = el('tr', null, [el('th', { scope: 'row', text: fmt(b.name) })]);
        tr.appendChild(estCell(b.valid));
        tr.appendChild(estCell(b.test));
        var why = b.fitted ? (b.note || b.definition) : (b.why || b.definition);
        tr.appendChild(el('td', { text: fmt(why) }));
        if (!b.fitted) tr.className = 'ev-degenerate';
        bt.querySelector('tbody').appendChild(tr);
      });
      pane.appendChild(el('section', { 'data-view': 'baselines' }, [
        el('h4', { text: 'Baselines' }), bt,
        para('Every baseline is fitted on the train part and selected on valid, on exactly the '
             + 'shaped features the network saw. A row that could not be fitted says so and is '
             + 'kept, because a baseline that quietly disappeared reads as one the network '
             + 'beat.', 'ev-skip')]));
    }

    /* ---- the threshold */
    if (THR && THR.value !== undefined && THR.value !== null) {
      var tp = el('section', { class: 'ev-panel', 'data-view': 'threshold' }, [
        el('h4', { text: 'Decision threshold' }),
        para(fmt(THR.value) + ' — ' + fmt(THR.source)
             + (THR.why ? ': ' + THR.why : '')
             + (THR.applies === false ? ' (this task does not decide with a threshold)' : ''))]);
      if (String(THR.source || '').indexOf('valid') === 0)
        tp.appendChild(para('Chosen on the valid part. A threshold optimised on the part it is '
                            + 'then measured on is the answer chosen after the look.',
                            'ev-skip'));
      pane.appendChild(tp);
    }

    /* ---- the curves */
    var curveHost = el('div', { class: 'ev-curves' });
    var drawn = 0;
    function pointsOf(c) {
      if (!c) return null;
      if (isArr(c.points)) return c.points;
      if (isArr(c.x) && isArr(c.y)) return c.x.map(function (x, i) {
        return { x: x, y: c.y[i] };
      });
      return null;
    }
    [['roc', 'ROC', 'false positive rate', 'true positive rate', true],
     ['pr', 'Precision–recall', 'recall', 'precision', false],
     ['reliability', 'Reliability', 'predicted', 'observed', true]].forEach(function (spec) {
      var pts = pointsOf(CUR[spec[0]]);
      if (!pts || !pts.length) return;
      drawn++;
      curveHost.appendChild(figure(spec[1], lineChart(pts, {
        label: spec[1] + ' curve', xlabel: spec[2], ylabel: spec[3], diagonal: spec[4],
        x0: 0, x1: 1, y0: 0, y1: 1, dots: spec[0] === 'reliability' }),
        spec[0] === 'reliability' ? 'above the diagonal is under-confident, below it '
                                  + 'over-confident' : null));
    });
    if (CUR.scores && (isArr(CUR.scores.positive) || isArr(CUR.scores.negative))) {
      var bars = [];
      (isArr(CUR.scores.negative) ? CUR.scores.negative : []).forEach(function (v, i) {
        bars.push({ k: (CUR.scores.x || [])[i], v: v, alt: true });
      });
      (isArr(CUR.scores.positive) ? CUR.scores.positive : []).forEach(function (v, i) {
        bars.push({ k: (CUR.scores.x || [])[i], v: v });
      });
      if (bars.length) {
        drawn++;
        curveHost.appendChild(figure('Score distribution',
          barChart(bars, { label: 'score histogram', xlabel: 'score', ylabel: 'rows' }),
          'the negatives faint, the positives solid'));
      }
    }
    if (CUR.confusion && isArr(CUR.confusion.matrix)) {
      var cm = CUR.confusion, labels = isArr(cm.labels) ? cm.labels : [];
      var ct = table([''].concat(labels.map(function (l) {
        return { text: 'predicted ' + l, cls: 'ev-num' };
      })), 'ev-confusion');
      cm.matrix.forEach(function (row, i) {
        var tr = el('tr', null, [el('th', { scope: 'row',
                                            text: 'actual ' + fmt(labels[i] === undefined
                                                                  ? i : labels[i]) })]);
        (isArr(row) ? row : []).forEach(function (v) {
          tr.appendChild(el('td', { class: 'ev-num', text: fmt(v), 'data-count': v }));
        });
        ct.querySelector('tbody').appendChild(tr);
      });
      var cf = el('figure', { class: 'ev-curve' }, [ct]);
      cf.appendChild(el('figcaption', { text: 'Confusion at threshold '
                                              + fmt(cm.threshold) }));
      curveHost.appendChild(cf);
      drawn++;
    }
    if (drawn) pane.appendChild(el('section', { 'data-view': 'curves' },
                                    [el('h4', { text: 'Curves' }), curveHost]));

    /* ---- the slices */
    // a slice the chain named that this look could not cut is said, with its reason
    function notCut(host) {
      (isArr(SLI.not_cut) ? SLI.not_cut : []).forEach(function (g) {
        host.appendChild(para('Not cut (' + fmt(g && g.source) + '): ' + fmt(g && g.why),
                              'ev-skip ev-not-cut'));
      });
    }
    if (SROWS.length) {
      var worst = SLI.worst || null;
      var st = table(['slice', { text: 'n', cls: 'ev-num' },
                      { text: PRIMARY, cls: 'ev-num' }, 'where it came from', 'note'],
                     'ev-slices');
      SROWS.forEach(function (row) {
        var e = (row.metrics || {})[PRIMARY];
        var tr = el('tr', row.degenerate ? { class: 'ev-degenerate' } : null, [
          el('th', { scope: 'row', text: fmt(row.name)
                     + (worst && worst.name === row.name ? '  ← worst' : '') })]);
        tr.appendChild(el('td', { class: 'ev-num', text: fmt(row.n), 'data-value': row.n }));
        tr.appendChild(estCell(e));
        tr.appendChild(el('td', { text: fmt(row.source) }));
        tr.appendChild(el('td', { text: row.note ? row.note
                                        : (isNum(row.n) && row.n < 30
                                           ? 'fewer than 30 rows — read the interval, not the '
                                             + 'point' : '') }));
        st.querySelector('tbody').appendChild(tr);
      });
      makeSortable(st, 1);
      var sec = el('section', { 'data-view': 'slices' },
                    [el('h4', { text: 'Slices' }), st]);
      if (READ.slices) sec.appendChild(para(READ.slices, 'ev-reading'));
      sec.appendChild(para('Sort by any column. A slice holding a single class is greyed: a '
        + 'ranking metric on it is not a low score, it is not a score at all.', 'ev-skip'));
      notCut(sec);
      pane.appendChild(sec);
    } else {
      // no slice was named (no nominal input, segment, drifting column or time order): the
      // surface still says so, rather than vanishing as if it had been forgotten
      var none = el('section', { 'data-view': 'slices' }, [el('h4', { text: 'Slices' })]);
      if (READ.slices) none.appendChild(para(READ.slices, 'ev-reading'));
      none.appendChild(para('The chain named no slice this evaluation could read, so no part of '
        + 'the test rows is singled out.', 'ev-skip'));
      notCut(none);
      pane.appendChild(none);
    }

    /* ---- robustness */
    if (CHECKS.length) {
      var rt = table(['check', { text: PRIMARY, cls: 'ev-num' },
                      { text: 'Δ vs unperturbed', cls: 'ev-num' }, 'note']);
      CHECKS.forEach(function (c) {
        var tr = el('tr', c.status === 'skipped' ? { class: 'ev-degenerate' } : null,
                    [el('th', { scope: 'row', text: fmt(c.check) })]);
        tr.appendChild(estCell(c.primary));
        tr.appendChild(c.delta ? diffCell(c.delta, MODE)
                               : el('td', { class: 'ev-num ev-none', text: '·' }));
        tr.appendChild(el('td', { class: c.status === 'skipped' ? 'ev-skip' : '',
                                  text: c.status === 'skipped' ? 'skipped — ' + fmt(c.why)
                                        : fmt(c.shift || c.source || '') }));
        rt.querySelector('tbody').appendChild(tr);
      });
      var rs = el('section', { 'data-view': 'robustness' },
                   [el('h4', { text: 'Robustness' }), rt]);
      if (READ.robustness) rs.appendChild(para(READ.robustness, 'ev-reading'));
      rs.appendChild(para('A skipped check is listed with its reason. A check that vanished '
                          + 'silently would read as a check that passed.', 'ev-skip'));
      pane.appendChild(rs);
    }

    /* ---- the remaining readings */
    var rdHost = el('section', { 'data-view': 'readings' },
                      [el('h4', { text: 'Readings' })]);
    [['calibration', 'Calibration'], ['limits', 'What this evaluation cannot say']]
      .forEach(function (r) {
        if (!READ[r[0]]) return;
        rdHost.appendChild(el('div', { class: 'ev-reading' }, [
          el('h4', { text: r[1] }), para(READ[r[0]])]));
      });
    if (rdHost.children.length > 1) pane.appendChild(rdHost);

    /* ---- provenance */
    var prov = details('how this page was produced');
    prov.setAttribute('data-view', 'provenance');
    prov.appendChild(para('Run ' + fmt(RUN.id) + ' of campaign ' + fmt(RUN.campaign)
      + '; checkpoint ' + fmt(RUN.checkpoint_digest) + '; spec ' + fmt(RUN.spec_digest)
      + '. Notebook ' + fmt(NB.path) + (NB.executed ? ' (executed)' : ' (NOT executed)')
      + (NB.reopened_test === false ? ', recomputed from the stored predictions'
         : NB.reopened_test === true ? ', which opened the test part again' : '') + '.'));
    if ((RUN.valid || {})[PRIMARY])
      prov.appendChild(para('The selected run’s valid ' + PRIMARY + ' was '
        + fmt((RUN.valid[PRIMARY] || {}).value)
        + ' — carried over from the ledger, and not re-measured here.'));
    if (MARK.length) {
      var ul = el('ul', { class: 'ev-markers' });
      MARK.forEach(function (m) {
        ul.appendChild(el('li', { class: String(m).indexOf('WARN') === 0 ? 'ev-m-warn' : '',
                                  text: String(m) }));
      });
      prov.appendChild(ul);
    }
    pane.appendChild(prov);

    window.__evaluation = {
      version: VERSION,
      metrics: function () { return MET; },
      slices: function () { return SROWS; },
      baselines: function () { return BROWS; },
      verdict: function () { return VERD; },
      seal: function () { return SEAL; }
    };
  });
})();
