/* /vega-train — `training` layer render script (contract: references/run-contract.md §5).
   Vanilla JS, no libraries, no network, inline SVG only. Reads ONLY #layer-training-data.
   Mounts one tab (data-layer="training", labelled Training) with seven surfaces:
     ledger      — one row per run: kind, hypothesis, delta, valid primary, gap, epochs, what
       stopped it, whether its expect held, the diagnostics that fired. The selected run is
       marked, and so is every run the selection rule may not pick — a pruned trial or a run a
       diagnostic disqualified — because a reader must not mistake a cut run for a result.
     curves      — a run selector and, for the chosen run, the curves the trainer recorded:
       train/valid loss, train/valid primary, learning rate, global and per-block gradient
       norms, update ratios. Switching runs redraws from `histories`; nothing is refetched.
     diagnostics — each fired diagnostic with its code, name, run, evidence, reading, and the
       rung it led to, linked to that run's row.
     hpo         — the study: its trials against the primary metric with the pruned ones marked
       apart, and the parameter importance with the method that produced it, because fANOVA and
       a marginal spread over quantiles are not the same claim.
     mechanics   — the loss, the update rule, the schedule as a curve, the clip, and what epoch
       1 step 1 did to the head's weights.
     selection   — the selected run, the rule that picked it, its margin over the baseline, the
       alternatives, and any override, kept visibly apart from the rule's own pick.
     forks       — what was asked and what answered it; an unanswered fork says the default
       applied unattended.
   It computes nothing: every number here was written by the trainer. Rounding for display is
   the only arithmetic in this file. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="training"]')) return;

  var VERSION = 'vega-train/training-render@1';
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
      /* learning rates, weight decays and update ratios must not round to zero */
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
  function clip(s, n) {
    s = String(s === null || s === undefined ? '' : s);
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  }
  function chip(text, cls, title) {
    var c = el('span', { class: 't-chip ' + (cls || ''), text: text });
    if (title) c.setAttribute('title', title);
    return c;
  }
  function para(t, cls) { return el('p', { class: cls || 't-prose', text: t }); }
  function details(summary, cls) {
    var d = el('details', { class: cls || 't-det' });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }
  function section(pane, view, title) {
    var d = el('div', { class: 't-sec', 'data-view': view });
    d.appendChild(el('h3', { text: title }));
    pane.appendChild(d);
    return d;
  }
  function table(headers, cls) {
    var t = el('table', { class: 't-table ' + (cls || '') });
    var hr = el('tr');
    headers.forEach(function (h) { hr.appendChild(el('th', { text: h })); });
    t.appendChild(el('thead', null, [hr]));
    t.appendChild(el('tbody'));
    return t;
  }
  function statGrid(pairs) {
    var g = el('div', { class: 't-stats' });
    pairs.forEach(function (p) {
      if (!p) return;
      g.appendChild(el('div', { class: 't-stat' }, [
        el('div', { class: 't-statv', text: p[1] }),
        el('div', { class: 't-statl', text: p[0] }),
        p[2] ? el('div', { class: 't-statn', text: p[2] }) : null]));
    });
    return g;
  }
  function sourceClass(src) {
    src = String(src || '');
    if (src.indexOf('netspec') === 0) return 't-src-spec';
    if (src.indexOf('train:diagnostic') === 0) return 't-src-diag';
    if (src.indexOf('train:hpo') === 0) return 't-src-hpo';
    if (src.indexOf('vega:') === 0) return 't-src-vega';
    if (src.indexOf('analysis:') === 0) return 't-src-analysis';
    if (src.indexOf('user:') === 0) return 't-src-user';
    return 't-src-default';
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
    var pane = el('section', { class: 'layer-training tab-pane', 'data-layer': 'training',
                               'data-tab': 'training', id: 'tab-training', role: 'tabpanel',
                               'aria-label': 'Training' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'training', 'data-layer': 'training', role: 'tab',
                             text: 'Training' });
      b.onclick = function () { switchTab('training'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  /* ---------------- the mechanical exclusions of the selection rule (run-contract §4). This is
     a reading of what the trainer recorded, not a judgement: a pruned trial, a run that diverged
     (D1) or never learned (D2), a run the budget cut before its first improvement (D9). */
  function ineligible(row) {
    var d = isArr(row.diagnostics) ? row.diagnostics : [];
    if (row.status === 'pruned') return 'a pruned trial';
    if (row.status === 'failed' || row.status === 'error') return 'the run did not finish';
    if (d.indexOf('D1') >= 0) return 'D1: it diverged';
    if (d.indexOf('D2') >= 0) return 'D2: it never learned';
    if (d.indexOf('D9') >= 0 && !row.best_epoch)
      return 'D9: the budget cut it before its first improvement';
    return null;
  }
  function deltaText(row) {
    if (row.delta_text) return row.delta_text;
    var d = isArr(row.delta) ? row.delta : [];
    if (!d.length) return '(none — the spec as engineered)';
    return d.map(function (e) {
      var to = (e.to && typeof e.to === 'object') ? (e.to.name || e.to.id || 'a new value')
                                                  : fmt(e.to);
      return (e.path || '?') + ' → ' + to;
    }).join('; ');
  }
  function studyOf(row) {
    if (row.study) return String(row.study);
    var m = /^train:hpo\/(.+)$/.exec(String(row.source || ''));
    if (m) return m[1];
    m = /^(H[^-]*)-/.exec(String(row.id || ''));
    return m ? m[1] : null;
  }

  /* ---------------- the curves. Every point is a record the trainer wrote; the only choice this
     code makes is the axis. A series whose values span decades (update ratios, gradient norms)
     is drawn on a log axis and says so — on a linear axis it would read as a flat zero. */
  function keysWithPrefix(hist, prefix) {
    var out = [];
    for (var k in hist) {
      if (!Object.prototype.hasOwnProperty.call(hist, k)) continue;
      if (k.indexOf(prefix) === 0 && isArr(hist[k]) && hist[k].length) out.push(k);
    }
    out.sort();
    return out;
  }
  function seriesFor(hist, keys, labels) {
    var out = [];
    keys.forEach(function (k, i) {
      if (isArr(hist[k]) && hist[k].length)
        out.push({ label: (labels && labels[i]) || k, values: hist[k] });
    });
    return out;
  }
  function drawChart(host, id, title, epochs, series, note) {
    var fig = el('figure', { class: 't-fig', 'data-chart': id });
    fig.appendChild(el('figcaption', { class: 't-figt', text: title }));
    var vals = [], i;
    series.forEach(function (s) {
      s.values.forEach(function (v) { if (isNum(v)) vals.push(v); });
    });
    if (!series.length || !vals.length) {
      fig.appendChild(para('This run’s record carries no ' + title.toLowerCase() + '.', 't-empty'));
      host.appendChild(fig);
      return fig;
    }
    var dlo = vals[0], dhi = vals[0];
    for (i = 1; i < vals.length; i++) {
      if (vals[i] < dlo) dlo = vals[i];
      if (vals[i] > dhi) dhi = vals[i];
    }
    var log = dlo > 0 && dhi / dlo >= 100;
    var lo = log ? Math.log(dlo) : dlo, hi = log ? Math.log(dhi) : dhi;
    if (hi === lo) { var pad = Math.abs(lo) || 1; lo -= pad; hi += pad; }
    var W = 720, H = 168, L = 78, R = 16, T = 16, B = 26;
    var e0 = epochs.length ? epochs[0] : 1;
    var eN = epochs.length ? epochs[epochs.length - 1] : 1;
    var span = (eN - e0) || 1;
    function X(i) {
      var e = epochs[i] === undefined ? e0 + i : epochs[i];
      return Math.round((L + ((e - e0) / span) * (W - L - R)) * 10) / 10;
    }
    function Y(v) {
      var t = log ? Math.log(v) : v;
      return Math.round((T + (1 - (t - lo) / (hi - lo)) * (H - T - B)) * 10) / 10;
    }
    var legend = el('div', { class: 't-chips t-legend' });
    series.forEach(function (s, si) {
      legend.appendChild(chip(s.label, 't-leg t-s' + (si % 6)));
    });
    fig.appendChild(legend);
    var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 't-svg t-chart', role: 'img',
      'aria-label': title + ': ' + series.map(function (s) { return s.label; }).join(', ')
        + ' over epochs ' + e0 + ' to ' + eN + ', from ' + fmt(dlo) + ' to ' + fmt(dhi)
        + (log ? ', on a log axis' : '') });
    g.appendChild(svg('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, class: 't-axis' }));
    g.appendChild(svg('line', { x1: L, y1: T, x2: L, y2: H - B, class: 't-axis' }));
    series.forEach(function (s, si) {
      var d = '', pen = false;
      for (var i = 0; i < s.values.length; i++) {
        var v = s.values[i];
        /* a null the backend wrote is a gap in the record, not a zero: the line breaks */
        if (!isNum(v) || (log && v <= 0)) { pen = false; continue; }
        d += (pen ? ' L ' : ' M ') + X(i) + ' ' + Y(v);
        pen = true;
      }
      if (d) g.appendChild(svg('path', { d: d.slice(1), fill: 'none',
        class: 't-line t-s' + (si % 6), 'data-series': s.label }));
    });
    g.appendChild(svg('text', { x: L - 6, y: T + 4, 'text-anchor': 'end', class: 't-tick',
                                text: fmt(dhi) }));
    g.appendChild(svg('text', { x: L - 6, y: H - B, 'text-anchor': 'end', class: 't-tick',
                                text: fmt(dlo) }));
    g.appendChild(svg('text', { x: L, y: H - 8, class: 't-tick', text: 'epoch ' + e0 }));
    g.appendChild(svg('text', { x: W - R, y: H - 8, 'text-anchor': 'end', class: 't-tick',
                                text: 'epoch ' + eN }));
    fig.appendChild(g);
    var cap = [];
    if (log) cap.push('log axis: this series spans decades');
    if (note) cap.push(note);
    if (cap.length) fig.appendChild(para(cap.join(' · '), 't-muted'));
    host.appendChild(fig);
    return fig;
  }

  /* ---------------- the study's trials against the primary metric. A pruned trial is drawn as a
     cross, not a dot: it was stopped on a partial history and its value is not comparable. */
  function drawTrials(host, trials, primaryName, baseline) {
    if (!trials.length) {
      host.appendChild(para('The study recorded no trial rows in the ledger.', 't-empty'));
      return;
    }
    var vals = [], i;
    trials.forEach(function (t) { if (isNum(t.value)) vals.push(t.value); });
    if (baseline && isNum(baseline.value)) vals.push(baseline.value);
    if (!vals.length) {
      host.appendChild(para('No trial carries a primary value; there is nothing to plot.',
                            't-empty'));
      return;
    }
    var lo = vals[0], hi = vals[0];
    for (i = 1; i < vals.length; i++) {
      if (vals[i] < lo) lo = vals[i];
      if (vals[i] > hi) hi = vals[i];
    }
    if (hi === lo) { hi = lo + (Math.abs(lo) || 1); lo = lo - (Math.abs(lo) || 1); }
    var W = 720, H = 208, L = 78, R = 16, T = 16, B = 40;
    var span = (trials.length - 1) || 1;
    function X(i) { return Math.round((L + (i / span) * (W - L - R)) * 10) / 10; }
    function Y(v) { return Math.round((T + (1 - (v - lo) / (hi - lo)) * (H - T - B)) * 10) / 10; }
    var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 't-svg t-scatter', role: 'img',
      'aria-label': trials.length + ' trials against valid ' + primaryName + ', from ' + fmt(lo)
        + ' to ' + fmt(hi) + '; pruned trials drawn as crosses' });
    g.appendChild(svg('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, class: 't-axis' }));
    g.appendChild(svg('line', { x1: L, y1: T, x2: L, y2: H - B, class: 't-axis' }));
    if (baseline && isNum(baseline.value)) {
      var by = Y(baseline.value);
      g.appendChild(svg('line', { x1: L, y1: by, x2: W - R, y2: by, class: 't-baseline' }));
      g.appendChild(svg('text', { x: W - R, y: by - 4, 'text-anchor': 'end', class: 't-tick',
        text: 'baseline ' + baseline.id + ' · ' + fmt(baseline.value) }));
    }
    trials.forEach(function (t, i) {
      if (!isNum(t.value)) return;
      var x = X(i), y = Y(t.value), pruned = t.status === 'pruned';
      var node = svg('g', { class: 't-trial' + (pruned ? ' t-trial-pruned' : '')
                              + (t.best ? ' t-trial-best' : ''), 'data-trial': t.id,
                            'data-status': t.status || 'done' });
      if (pruned) {
        node.appendChild(svg('path', { class: 't-cross',
          d: 'M ' + (x - 4.5) + ' ' + (y - 4.5) + ' L ' + (x + 4.5) + ' ' + (y + 4.5)
           + ' M ' + (x - 4.5) + ' ' + (y + 4.5) + ' L ' + (x + 4.5) + ' ' + (y - 4.5) }));
      } else {
        node.appendChild(svg('circle', { cx: x, cy: y, r: t.best ? 5.5 : 4,
                                         class: 't-pt' + (t.best ? ' t-pt-best' : '') }));
      }
      node.appendChild(svg('title', { text: t.id + ': ' + fmt(t.value) + ' (' + (t.status || 'done')
        + ')' }));
      node.appendChild(svg('text', { x: x, y: H - B + 13, 'text-anchor': 'middle',
                                     class: 't-tick', text: t.id }));
      g.appendChild(node);
    });
    g.appendChild(svg('text', { x: L - 6, y: T + 4, 'text-anchor': 'end', class: 't-tick',
                                text: fmt(hi) }));
    g.appendChild(svg('text', { x: L - 6, y: H - B, 'text-anchor': 'end', class: 't-tick',
                                text: fmt(lo) }));
    g.appendChild(svg('text', { x: L, y: H - 6, class: 't-tick', text: 'valid ' + primaryName
                                + ', trials in the order the sampler drew them' }));
    host.appendChild(g);
  }

  /* ---------------- parameter importance. The bars are the study's numbers; the method that
     produced them is printed beside them, because it is what the bars actually claim. */
  function drawImportance(host, imp, how) {
    var keys = [];
    for (var k in imp) if (Object.prototype.hasOwnProperty.call(imp, k)) keys.push(k);
    if (!keys.length) {
      // the study's own words for why there is none (too few trials, one parameter) are what
      // the reader needs; a bare "none" would hide them
      host.appendChild(para('The study recorded no parameter importance'
        + (how ? ': ' + how : '') + '.', 't-empty'));
      return;
    }
    keys.sort(function (a, b) { return (imp[b] || 0) - (imp[a] || 0); });
    var max = 0;
    keys.forEach(function (k) { if (isNum(imp[k]) && imp[k] > max) max = imp[k]; });
    if (!(max > 0)) max = 1;
    var W = 720, ROW = 22, PAD = 8, LW = 300, VW = 58;
    var H = PAD * 2 + keys.length * ROW;
    var g = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 't-svg t-bars', role: 'img',
      'aria-label': 'Parameter importance by ' + (how || 'an unrecorded method') + ': '
        + keys.map(function (k) { return k + ' ' + fmt(imp[k]); }).join(', ') });
    keys.forEach(function (k, i) {
      var y = PAD + i * ROW, w = Math.max(1, ((imp[k] || 0) / max) * (W - LW - VW - 16));
      var row = svg('g', { class: 't-barrow', 'data-param': k });
      row.appendChild(svg('text', { x: LW - 8, y: y + 14, 'text-anchor': 'end', class: 't-barlabel',
                                    text: clip(k, 44) }));
      row.appendChild(svg('rect', { x: LW, y: y + 4, width: w, height: 13, rx: 2, class: 't-bar' }));
      row.appendChild(svg('text', { x: LW + w + 6, y: y + 14, class: 't-tick', text: fmt(imp[k]) }));
      g.appendChild(row);
    });
    host.appendChild(g);
    host.appendChild(para('Importance computed by: ' + (how || 'method not recorded')
      + '. A fANOVA decomposition and a marginal spread across a dimension’s quantiles are '
      + 'different claims; this line says which one these bars make.',
      how ? 't-muted' : 't-warntext'));
  }

  ready(function () {
    /* the guard above catches a script injected into an already-rendered page; this one catches
       two copies of the script parsed before the document was ready, which both mount otherwise */
    if (document.querySelector('[data-layer="training"]')) return;
    var s = document.getElementById('layer-training-data');
    var pane = mount();
    pane.appendChild(el('header', { class: 't-head' }, [el('h2', { text: 'Training' })]));
    var M = null;
    try { M = s ? JSON.parse(s.textContent) : null; } catch (e) { M = null; }
    if (!M || typeof M !== 'object') {
      pane.appendChild(para(s ? 'The training layer data did not parse; there is no campaign to '
                              + 'show. The campaign directory on disk is still the record.'
                              : 'This file carries a training layer marker but no data script.',
                            't-empty'));
      window.__training = { version: VERSION, runs: function () { return []; },
        history: function () { return null; }, selection: function () { return null; },
        diagnostics: function () { return []; } };
      return;
    }
    var CAMP = M.campaign || {}, LAD = M.ladder || {}, EXP = M.experiments || {},
        HIST = M.histories || {}, DIAG = isArr(M.diagnostics) ? M.diagnostics : [],
        MECH = M.mechanics || {}, NB = M.notebook || {}, READ = M.readings || {},
        MARK = isArr(M.markers) ? M.markers : [],
        ROWS = isArr(EXP.runs) ? EXP.runs : [],
        SEL = EXP.selection || null,
        HPO = isArr(EXP.hpo) ? EXP.hpo : [],
        FORKS = isArr(M.forks) ? M.forks : (isArr(LAD.forks) ? LAD.forks : []);
    var LADBY = {};
    (isArr(LAD.runs) ? LAD.runs : []).forEach(function (r) { if (r && r.id) LADBY[r.id] = r; });
    var BY = {};
    ROWS.forEach(function (r) { if (r && r.id) BY[r.id] = r; });
    var PRIMARY = 'primary';
    for (var i = 0; i < ROWS.length; i++) if (ROWS[i].primary_name) {
      PRIMARY = ROWS[i].primary_name; break;
    }
    var selectRun = null;   /* bound once the curves surface is built */
    function goRun(id) {
      if (selectRun) selectRun(id);
      var row = document.getElementById('t-run-' + id);
      if (row && row.scrollIntoView) row.scrollIntoView({ block: 'center' });
    }

    /* ---------------- the campaign's identity, the abstract, the markers */
    var top = el('div', { class: 't-abstract' });
    var idbar = el('div', { class: 't-chips' });
    if (CAMP.run_dir) idbar.appendChild(chip(CAMP.run_dir, 't-id', 'the campaign directory'));
    if (CAMP.status) idbar.appendChild(chip('status ' + CAMP.status, 't-tag'));
    if (CAMP.mode) idbar.appendChild(chip('mode ' + CAMP.mode, 't-tag'));
    if (CAMP.device) idbar.appendChild(chip('device ' + CAMP.device, 't-tag'));
    if ((CAMP.backend || {}).name)
      idbar.appendChild(chip(CAMP.backend.name + ' ' + fmt(CAMP.backend.version), 't-tag'));
    if (CAMP.seed !== undefined) idbar.appendChild(chip('seed ' + fmt(CAMP.seed), 't-tag'));
    var bud = EXP.budget || CAMP.budget || {};
    if (bud.wall_clock_min !== undefined)
      idbar.appendChild(chip('budget ' + fmt(bud.spent_min) + ' / ' + fmt(bud.wall_clock_min)
        + ' min', 't-tag', 'wall clock spent against the campaign budget'));
    idbar.appendChild(chip(ROWS.length + ' runs', 't-tag'));
    if (NB.executed === false)
      idbar.appendChild(chip('notebook not executed', 't-warn',
        'the notebook that replays this campaign was emitted but never run'));
    else if (NB.executed === true)
      idbar.appendChild(chip('notebook replayed ' + fmt(NB.replay_epochs) + ' epochs: '
        + (NB.replay_ok === false ? 'MISMATCH' : 'ok'),
        NB.replay_ok === false ? 't-bad' : 't-ok', NB.path || ''));
    top.appendChild(idbar);
    top.appendChild(READ.abstract ? para(READ.abstract)
      : para('No abstract was recorded for this campaign.', 't-empty'));
    if (CAMP.test_seal)
      top.appendChild(para('The test part has been opened: ' + fmt(CAMP.test_seal)
        + '. Nothing below was trained after that.', 't-warntext'));
    if ((CAMP.protocol || {}).applied)
      top.appendChild(para('Split protocol: ' + CAMP.protocol.applied + ' ('
        + (CAMP.protocol.source || 'unsourced') + ')'
        + (CAMP.protocol.reason ? ' — ' + CAMP.protocol.reason : ''), 't-muted'));
    if (MARK.length) {
      var md = details('run markers (' + MARK.length + ')');
      MARK.forEach(function (m) { md.appendChild(el('p', { class: 't-marker', text: m })); });
      top.appendChild(md);
    }
    pane.appendChild(top);

    /* ---------------- ledger */
    var lv = section(pane, 'ledger', 'The ledger');
    if (!ROWS.length) {
      lv.appendChild(para('The ledger carries no run: this campaign trained nothing.', 't-empty'));
    } else {
      var lt = table(['run', 'kind', 'hypothesis', 'delta', 'valid ' + PRIMARY, 'gap', 'epochs',
                      'stopped by', 'expect held', 'diagnostics']);
      var ltb = lt.querySelector('tbody');
      ROWS.forEach(function (r) {
        var bad = ineligible(r), sel = SEL && SEL.run === r.id;
        var tr = el('tr', { class: 't-row' + (sel ? ' t-selected' : '') + (bad ? ' t-ineligible' : ''),
                            id: 't-run-' + r.id, 'data-run': r.id,
                            'data-status': r.status || '', 'data-eligible': bad ? 'no' : 'yes' });
        var idTd = el('td');
        var btn = el('button', { class: 't-gotorun', type: 'button', 'data-run': r.id,
                                 text: r.id, title: 'show this run’s curves' });
        btn.onclick = function () { goRun(r.id); };
        idTd.appendChild(btn);
        if (sel) idTd.appendChild(chip('selected', 't-ok', 'the run the selection rule picked'));
        if (bad) idTd.appendChild(chip(bad, 't-warn', 'the selection rule may not pick this run'));
        if (r.supersedes) idTd.appendChild(chip('supersedes ' + r.supersedes, 't-na'));
        tr.appendChild(idTd);
        var kindTd = el('td');
        kindTd.appendChild(chip(r.kind || '·', 't-na'));
        if (r.family) kindTd.appendChild(chip(r.family, 't-tag'));
        kindTd.appendChild(chip(r.source || 'unsourced', sourceClass(r.source),
                                'the decision this rung cites'));
        tr.appendChild(kindTd);
        var hyp = r.hypothesis || (LADBY[r.id] || {}).hypothesis || '';
        tr.appendChild(el('td', { class: 't-hyp', text: clip(hyp, 130), title: hyp }));
        var dt = deltaText(r);
        tr.appendChild(el('td', { class: 't-mono', text: clip(dt, 90), title: dt }));
        tr.appendChild(el('td', { class: 't-numcell',
          text: isNum(r.primary) ? fmt(r.primary) : fmt((r.valid || {})[PRIMARY]) }));
        tr.appendChild(el('td', { class: 't-numcell',
          text: fmt((r.gap || {}).loss !== undefined ? r.gap.loss : r.gap) }));
        tr.appendChild(el('td', { class: 't-numcell', text: fmt(r.epochs)
          + (r.best_epoch ? ' (best ' + r.best_epoch + ')' : '') }));
        tr.appendChild(el('td', null, [chip(r.stopped_by || r.status || '·',
          r.stopped_by === 'divergence' ? 't-bad'
            : (r.stopped_by === 'budget' || r.status === 'pruned') ? 't-warn' : 't-na')]));
        tr.appendChild(el('td', null, [r.expect_held === true ? chip('held', 't-ok')
          : r.expect_held === false ? chip('did not hold', 't-bad')
          : chip('not judged', 't-na', 'the campaign recorded no verdict on this expect')]));
        var dg = el('td');
        (isArr(r.diagnostics) ? r.diagnostics : []).forEach(function (c) {
          dg.appendChild(chip(c, 't-diagchip', 'a diagnostic that fired on this run'));
        });
        (isArr(r.unavailable) ? r.unavailable : []).forEach(function (c) {
          dg.appendChild(chip(c + ' n/a', 't-na',
            'this diagnostic needed a record the run does not carry; it was never guessed'));
        });
        if (!dg.childNodes.length) dg.appendChild(chip('none', 't-na'));
        tr.appendChild(dg);
        ltb.appendChild(tr);
      });
      lv.appendChild(lt);
      var nIn = ROWS.filter(function (r) { return !!ineligible(r); }).length;
      lv.appendChild(para('One row per run in the ledger, in the order the campaign wrote them. '
        + (SEL && SEL.run ? 'The selected run (' + SEL.run + ') is marked. ' : '')
        + (nIn ? nIn + ' run(s) are marked ineligible: the selection rule may not pick them, '
                 + 'and their numbers are not results.'
               : 'Every run is eligible for the selection rule.'), 't-muted'));
      if (EXP.ranking && isArr(EXP.ranking.order) && EXP.ranking.order.length)
        lv.appendChild(para('Ranked by ' + (EXP.ranking.by || '?') + ' ('
          + (EXP.ranking.mode || '?') + '): ' + EXP.ranking.order.join(' › '), 't-muted'));
    }

    /* ---------------- curves */
    var cv = section(pane, 'curves', 'Curves');
    var selWrap = el('div', { class: 't-runsel', role: 'group', 'aria-label': 'Choose a run' });
    var curveHost = el('div', { class: 't-curves' });
    var readHost = el('div', { class: 't-perread' });
    cv.appendChild(selWrap);
    cv.appendChild(readHost);
    cv.appendChild(curveHost);
    var curveIds = ROWS.map(function (r) { return r.id; }).filter(function (id) { return !!id; });
    if (!curveIds.length) for (var hk in HIST)
      if (Object.prototype.hasOwnProperty.call(HIST, hk)) curveIds.push(hk);
    if (!curveIds.length) {
      cv.appendChild(para('No run to draw: the layer carries neither a ledger nor a history.',
                          't-empty'));
    } else {
      var buttons = {};
      curveIds.forEach(function (id) {
        var b = el('button', { class: 't-runbtn', type: 'button', 'data-run': id, text: id,
                               'aria-pressed': 'false' });
        b.onclick = function () { selectRun(id); };
        buttons[id] = b;
        selWrap.appendChild(b);
      });
      selectRun = function (id) {
        if (!buttons[id]) return;
        curveIds.forEach(function (o) {
          buttons[o].classList.toggle('active', o === id);
          buttons[o].setAttribute('aria-pressed', o === id ? 'true' : 'false');
        });
        while (curveHost.firstChild) curveHost.removeChild(curveHost.firstChild);
        while (readHost.firstChild) readHost.removeChild(readHost.firstChild);
        curveHost.setAttribute('data-run', id);
        var row = BY[id] || {};
        var head = el('div', { class: 't-chips' });
        head.appendChild(chip(id, 't-id'));
        if (row.status) head.appendChild(chip(row.status, row.status === 'pruned' ? 't-warn' : 't-na'));
        if (isNum(row.primary)) head.appendChild(chip('valid ' + PRIMARY + ' ' + fmt(row.primary),
                                                      't-tag'));
        if (row.time_s !== undefined) head.appendChild(chip(fmt(row.time_s) + ' s', 't-tag'));
        readHost.appendChild(head);
        var pr = (READ.per_run || {})[id];
        readHost.appendChild(pr ? para(pr)
          : para('No reading was recorded for ' + id + '.', 't-empty'));
        var h = HIST[id];
        if (!h || typeof h !== 'object') {
          curveHost.appendChild(para('The layer carries no history for ' + id
            + ', so no curve can be drawn for it. The epoch records are on disk in the campaign '
            + 'directory; this layer was written without them.', 't-empty'));
          return;
        }
        var epochs = isArr(h.epochs) ? h.epochs : [];
        var note = h.downsampled_from ? 'downsampled from ' + fmt(h.downsampled_from)
          + ' epochs; the full record is ' + (h.path || 'in the campaign directory') : (h.path || '');
        var any = false;
        [['loss', 'Loss', ['train.loss', 'valid.loss'], ['train loss', 'valid loss']],
         ['primary', 'Primary metric — ' + PRIMARY,
          ['train.' + PRIMARY, 'valid.' + PRIMARY], ['train ' + PRIMARY, 'valid ' + PRIMARY]],
         ['lr', 'Learning rate', ['lr'], ['lr']]].forEach(function (c) {
          var ser = seriesFor(h, c[2], c[3]);
          if (ser.length) any = true;
          drawChart(curveHost, c[0], c[1], epochs, ser, note);
        });
        var gk = keysWithPrefix(h, 'grad_norm.');
        var gser = seriesFor(h, gk, gk.map(function (k) { return k.slice(10); }));
        if (gser.length) any = true;
        drawChart(curveHost, 'grad-norm', 'Gradient norms — global and per block', epochs, gser, note);
        var uk = keysWithPrefix(h, 'update_ratio.');
        var user = seriesFor(h, uk, uk.map(function (k) { return k.slice(13); }));
        if (user.length) any = true;
        drawChart(curveHost, 'update-ratio', 'Update ratios ‖Δθ‖/‖θ‖ per block', epochs, user, note);
        if (!any)
          curveHost.appendChild(para('The history for ' + id + ' carries no series this layer '
            + 'knows how to draw.', 't-empty'));
      };
      var first = (SEL && SEL.run && BY[SEL.run]) ? SEL.run : curveIds[0];
      selectRun(first);
      cv.appendChild(para('The selector redraws from the layer’s own records; nothing is fetched. '
        + 'Every point is an epoch the trainer wrote.', 't-muted'));
    }

    /* ---------------- diagnostics */
    var gv = section(pane, 'diagnostics', 'Diagnostics');
    if (!DIAG.length) {
      gv.appendChild(para('No diagnostic fired on any run of this campaign.', 't-empty'));
    } else {
      var gt = table(['code', 'name', 'run', 'evidence', 'reading', 'led to']);
      var gtb = gt.querySelector('tbody');
      DIAG.forEach(function (d) {
        var tr = el('tr', { class: 't-diag', 'data-code': d.code || '', 'data-run': d.run || '' });
        tr.appendChild(el('td', null, [chip(d.code || '·', 't-diagchip')]));
        tr.appendChild(el('td', { text: d.name || '·' }));
        var runTd = el('td');
        if (d.run && BY[d.run]) {
          var rb = el('a', { class: 't-runlink', href: '#t-run-' + d.run, text: d.run });
          rb.onclick = function () { goRun(d.run); };
          runTd.appendChild(rb);
        } else {
          runTd.appendChild(el('span', { class: 't-mono', text: d.run || '·' }));
        }
        tr.appendChild(runTd);
        tr.appendChild(el('td', { class: 't-evidence', text: d.evidence || '·' }));
        tr.appendChild(el('td', { text: d.reading || '—' }));
        var ledTd = el('td', { class: 't-led' });
        if (d.led_to && BY[d.led_to]) {
          var lb = el('a', { class: 't-runlink', href: '#t-run-' + d.led_to, text: d.led_to });
          lb.onclick = function () { goRun(d.led_to); };
          ledTd.appendChild(lb);
        } else if (d.led_to) {
          ledTd.appendChild(chip(d.led_to, 't-warn', 'this rung is not in the ledger'));
        } else {
          ledTd.appendChild(chip('no rung ran', 't-na',
            'the diagnostic proposed a remedy; no run in this campaign carried it out'));
        }
        tr.appendChild(ledTd);
        gtb.appendChild(tr);
      });
      gv.appendChild(gt);
      gv.appendChild(para('A diagnostic is a mechanical condition on the epoch records; the '
        + 'evidence is the trainer’s, the reading is the skill’s, and the rung — where one ran — '
        + 'is what the campaign did about it.', 't-muted'));
    }

    /* ---------------- hpo */
    var hv = section(pane, 'hpo', 'The search');
    if (!HPO.length) {
      hv.appendChild(para('This campaign ran no hyper-parameter study.', 't-empty'));
    } else {
      HPO.forEach(function (st) {
        var box = el('div', { class: 't-study', 'data-study': st.study || '' });
        box.appendChild(el('h4', { text: 'Study ' + (st.study || '·') }));
        box.appendChild(statGrid([
          ['trials', fmt(st.trials)],
          ['completed', fmt(st.completed)],
          ['pruned', fmt(st.pruned)],
          ['best trial', fmt(st.best)],
          st.best_value !== undefined ? ['best valid ' + PRIMARY, fmt(st.best_value)] : null,
          st.sampler ? ['sampler', String(st.sampler)] : null,
          st.pruner ? ['pruner', String(st.pruner)] : null]));
        var trials = ROWS.filter(function (r) {
          return (r.kind === 'hpo' || r.kind === 'trial') && studyOf(r) === st.study;
        }).map(function (r) {
          return { id: r.id, value: isNum(r.primary) ? r.primary : (r.valid || {})[PRIMARY],
                   status: r.status, best: st.best === r.id };
        });
        var base = null;
        ROWS.forEach(function (r) {
          if (!base && r.kind === 'baseline' && isNum(r.primary))
            base = { id: r.id, value: r.primary };
        });
        drawTrials(box, trials, PRIMARY, base);
        if (isArr(st.space) && st.space.length)
          box.appendChild(para('Searched: ' + st.space.join(', ')
            + (st.generic_space ? ' — a generic space: the family has no row in the catalog.' : ''),
            st.generic_space ? 't-warntext' : 't-muted'));
        drawImportance(box, st.importance || {}, st.importance_how || st.how);
        hv.appendChild(box);
      });
      if (READ.hpo) hv.appendChild(para(READ.hpo));
      else hv.appendChild(para('No reading was recorded for the search.', 't-empty'));
    }

    /* ---------------- mechanics */
    var mv = section(pane, 'mechanics', 'Mechanics');
    mv.appendChild(el('h4', { text: 'Loss' }));
    if (MECH.loss) mv.appendChild(el('pre', { class: 't-formula', text: String(MECH.loss) }));
    else mv.appendChild(para('The layer carries no loss string.', 't-empty'));
    mv.appendChild(el('h4', { text: 'Update rule' }));
    if (MECH.update_rule)
      mv.appendChild(el('pre', { class: 't-formula', text: String(MECH.update_rule) }));
    else mv.appendChild(para('The layer carries no update rule.', 't-empty'));
    mv.appendChild(el('h4', { text: 'Schedule' }));
    var sch = MECH.schedule || {};
    var schChips = el('div', { class: 't-chips' });
    schChips.appendChild(chip(sch.name || 'no schedule', 't-id'));
    if (sch.warmup_epochs !== undefined)
      schChips.appendChild(chip('warmup ' + fmt(sch.warmup_epochs) + ' epochs', 't-tag'));
    schChips.appendChild(chip('grad clip ' + (MECH.clip === null || MECH.clip === undefined
      ? 'off' : fmt(MECH.clip)), MECH.clip ? 't-tag' : 't-warn',
      'the pre-clip norm the trainer clipped every step to'));
    mv.appendChild(schChips);
    var lrs = isArr(sch.lr_by_epoch) ? sch.lr_by_epoch : [];
    var lrEpochs = [];
    for (var li = 0; li < lrs.length; li++) lrEpochs.push(li + 1);
    drawChart(mv, 'schedule', 'Learning rate by epoch', lrEpochs,
              lrs.length ? [{ label: 'lr', values: lrs }] : [], null);
    mv.appendChild(el('h4', { text: 'The first step' }));
    // run-contract §5 writes `one_step` as prose and the apply script leaves the record it was
    // read from beside it as `one_step_record`. Both are shown: the numbers as tiles, because
    // a ratio of 5e-06 is a number a reader compares, and the sentence beneath them, because
    // it is the sentence the layer and the notebook both carry.
    var osRec = (MECH.one_step_record && typeof MECH.one_step_record === 'object')
      ? MECH.one_step_record
      : (typeof MECH.one_step === 'object' ? MECH.one_step : null);
    if (MECH.one_step === null || MECH.one_step === undefined) {
      mv.appendChild(para('No first-step record: what epoch 1 step 1 did to the head’s weights '
        + 'was not written.', 't-empty'));
    } else if (osRec) {
      var os = osRec;
      mv.appendChild(statGrid([
        os.block ? ['head block', String(os.block)] : null,
        ['‖g‖', fmt(os.grad_norm_head !== undefined ? os.grad_norm_head : os.grad_norm)],
        ['‖Δθ‖', fmt(os.delta_norm_head !== undefined ? os.delta_norm_head : os.delta_norm)],
        ['ratio', fmt(os.ratio_head !== undefined ? os.ratio_head : os.ratio)],
        os.loss !== undefined ? ['loss', fmt(os.loss)] : null,
        os.lr !== undefined ? ['lr', fmt(os.lr)] : null]));
      if (typeof MECH.one_step === 'string') mv.appendChild(para(MECH.one_step));
    } else {
      mv.appendChild(para(String(MECH.one_step)));
    }
    if (READ.mechanics) mv.appendChild(para(READ.mechanics));

    /* ---------------- selection */
    var sv = section(pane, 'selection', 'The selection');
    if (!SEL || !SEL.run) {
      sv.appendChild(para('No run was selected: the ledger carries no selection record, so this '
        + 'campaign hands nothing to /vega-eval.', 't-empty'));
    } else {
      var selRow = BY[SEL.run] || {}, selBad = selRow.id ? ineligible(selRow) : null;
      var selChips = el('div', { class: 't-chips' });
      var sb = el('a', { class: 't-runlink t-id', href: '#t-run-' + SEL.run, text: SEL.run });
      sb.onclick = function () { goRun(SEL.run); };
      selChips.appendChild(sb);
      if (isNum(selRow.primary))
        selChips.appendChild(chip('valid ' + PRIMARY + ' ' + fmt(selRow.primary), 't-ok'));
      if (SEL.margin_over_baseline !== undefined && SEL.margin_over_baseline !== null)
        selChips.appendChild(chip('margin over baseline ' + fmt(SEL.margin_over_baseline),
          isNum(SEL.margin_over_baseline) && SEL.margin_over_baseline > 0 ? 't-ok' : 't-warn',
          'what the selected run buys over the run of the spec as engineered'));
      if (SEL.source) selChips.appendChild(chip(SEL.source, sourceClass(SEL.source)));
      if (SEL.catalog) selChips.appendChild(chip(SEL.catalog, 't-cat'));
      sv.appendChild(selChips);
      sv.appendChild(el('div', { class: 't-rule' }, [
        el('div', { class: 't-rulel', text: 'the rule' }),
        para(SEL.rule || 'No rule was recorded — a selection without a rule is a preference.',
             SEL.rule ? 't-prose' : 't-warntext')]));
      var ov = SEL.override;
      var ovBox = el('div', { class: 't-override' + (ov ? ' t-override-on' : ''),
                              'data-override': ov ? 'yes' : 'no' });
      if (ov) {
        ovBox.appendChild(el('div', { class: 't-ovhead' }, [
          chip('OVERRIDE', 't-bad', 'a human overruled the rule'),
          chip(ov.source || 'user', sourceClass(ov.source || 'user:'))]));
        ovBox.appendChild(para('This run was not the rule’s pick: a person chose it. '
          + (ov.reason || 'No reason was recorded.')));
        if (ov.rule_pick) ovBox.appendChild(para('The rule would have picked ' + ov.rule_pick + '.',
                                                 't-muted'));
      } else {
        ovBox.appendChild(para('No override: ' + SEL.run + ' is the rule’s own pick.', 't-muted'));
      }
      sv.appendChild(ovBox);
      if (selBad && !ov)
        sv.appendChild(para('The selected run is marked ineligible (' + selBad + ') and no '
          + 'override is recorded. The apply script refuses this; read the selection as unsettled.',
          't-warntext'));
      var alts = isArr(SEL.alternatives) ? SEL.alternatives : [];
      sv.appendChild(el('h4', { text: 'Alternatives' }));
      if (!alts.length) {
        sv.appendChild(para('No alternative was recorded beside the rule’s pick.', 't-empty'));
      } else {
        var at = table(['run', 'when it would be the better pick']);
        var atb = at.querySelector('tbody');
        alts.forEach(function (a) {
          var tr = el('tr', { class: 't-alt', 'data-run': a.run || '' });
          var td = el('td');
          if (a.run && BY[a.run]) {
            var ab = el('a', { class: 't-runlink', href: '#t-run-' + a.run, text: a.run });
            ab.onclick = function () { goRun(a.run); };
            td.appendChild(ab);
          } else td.appendChild(el('span', { class: 't-mono', text: a.run || '·' }));
          tr.appendChild(td);
          tr.appendChild(el('td', { text: a.when || a.reason || '·' }));
          atb.appendChild(tr);
        });
        sv.appendChild(at);
      }
      sv.appendChild(READ.selection ? para(READ.selection)
        : para('No reading was recorded for the selection.', 't-empty'));
    }

    /* ---------------- forks */
    var fv = section(pane, 'forks', 'Forks');
    if (!FORKS.length) {
      fv.appendChild(para('No fork was raised: nothing in the campaign was open to a choice the '
        + 'user had to settle.', 't-empty'));
    } else {
      var ft = table(['asked', 'answer']);
      var ftb = ft.querySelector('tbody');
      FORKS.forEach(function (f) {
        if (typeof f === 'string') f = { asked: f, answer: null };
        var unattended = f.answer === undefined || f.answer === null;
        var ans = el('td');
        if (unattended) {
          ans.appendChild(chip('default applied (unattended)', 't-warn',
            'no one answered this fork; the default stands'));
          if (f.or && String(f.or) !== 'default applied (unattended)')
            ans.appendChild(el('span', { class: 't-ortext', text: String(f.or) }));
        } else {
          ans.appendChild(chip(String(f.answer), 't-ok', 'answered'));
        }
        ftb.appendChild(el('tr', { class: 't-fork' + (unattended ? ' t-unattended' : '') }, [
          el('td', { text: f.asked || '·' }), ans]));
      });
      fv.appendChild(ft);
    }

    window.__training = {
      version: VERSION,
      runs: function () { return ROWS.slice(); },
      history: function (id) { return HIST[id] || null; },
      selection: function () { return SEL; },
      diagnostics: function () { return DIAG.slice(); }
    };
  });
})();
