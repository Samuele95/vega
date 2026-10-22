/* /vega-pack — `bundle` layer render script (contract: references/bundle-contract.md §8).
   Vanilla JS, no libraries, no network, inline SVG only. Reads ONLY #layer-bundle-data.
   Mounts one tab (data-layer="bundle", labelled Bundle) with nine surfaces:
     verdict     — first, and large. This is the page a reader opens to decide whether to *use*
       the model, so the sentence that governs that decision cannot be a badge in a corner:
       earned, not earned, or beaten by a baseline — three states, not two, because "a baseline
       is better" is a stronger and different claim from "no difference was shown". The paired
       difference is drawn against zero, and the stances the bundle ships under are in the same
       box: a stance folded away below the fold would let a reader take an unearned bundle for
       an endorsed one.
     summary     — the manifest in one strip, with the digests folded into it: reachable, and
       out of the way of the reading.
     roundtrip   — the claim the whole stage rests on: raw rows through the bundle's own
       preprocess and network reproduce the scores the evaluation stored. Status, rows, and the
       tolerance *formula* that "pass" was measured against, because a pass without its bound
       is not a result; the shaper's twin check on its own line, being a test of another file.
     examples    — what the shipped test proves. `kind: test` means the bundle reproduces the
       evaluation; `kind: synthetic` means it reproduces only itself. Different guarantees, and
       not rendered as though they were the same one.
     schema      — every input column with its dtype, whether it is required, and the domain
       seen in training. A column shown without its domain invites silent extrapolation, so the
       domain is rendered for every column that has one. Sortable, by mouse or by Enter.
     exports     — each exported format against predict.py at its parity tolerance; a failing
       parity ships marked rather than quietly. The serving record sits with it.
     findings    — the audits' findings on the serving slice (architect and improve, with the
                   audits that reported each), each tool's judgement, or the statement of none.
     usage       — the CLI and module snippets verbatim from the manifest (and, for a
                    --include full bundle, its train.py and evaluate.py lines with what they
                    are not), the link to the
       model card (a sibling file, linked and never inlined), and the notebook.
     readings    — the abstract and the intended-use prose, as written.
   It computes nothing: every number here was written by pack.py. Rounding for display and
   placing an interval on an axis are the only arithmetic in this file. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="bundle"]')) return;

  var VERSION = 'vega-pack/bundle-render@1';
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
  /* A recorded statistic, written as the record writes it: Python's repr of the float (the
     shortest digits that round-trip, '.0' on a whole number, an exponent below 1e-4 and from
     1e16). fmt() rounds for reading; a verdict's difference, its primary metric and the round
     trip's worst difference are quoted, so a reader can find them in MANIFEST.json as shown. */
  function rec(v) {
    if (!isNum(v)) return fmt(v);
    if (v === 0) return 1 / v < 0 ? '-0.0' : '0.0';
    var m = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(v.toExponential());
    var sign = m[1], dig = m[2] + (m[3] || ''), e = parseInt(m[4], 10), z = '';
    if (e < -4 || e >= 16) {
      var a = Math.abs(e);
      return sign + dig.charAt(0) + (dig.length > 1 ? '.' + dig.slice(1) : '') + 'e'
        + (e < 0 ? '-' : '+') + (a < 10 ? '0' : '') + a;
    }
    if (e < 0) {
      while (z.length < -e - 1) z += '0';
      return sign + '0.' + z + dig;
    }
    while (dig.length < e + 1) dig += '0';
    return sign + dig.slice(0, e + 1) + '.' + (dig.slice(e + 1) || '0');
  }
  function ciText(ci, f) {
    f = f || fmt;
    if (!isArr(ci) || ci.length !== 2 || !isNum(ci[0]) || !isNum(ci[1])) return '';
    return '[' + f(ci[0]) + ', ' + f(ci[1]) + ']';
  }
  function para(t, cls) { return el('p', { class: cls || 'bn-prose', text: t }); }
  function code(t) { return el('code', { class: 'bn-code', text: String(t) }); }
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
  function rows(t) { return t.querySelector('tbody'); }
  function section(view, title, cls) {
    var s = el('section', { 'data-view': view, class: cls || '' });
    if (title) s.appendChild(el('h4', { text: title }));
    return s;
  }
  /* a status word rendered as what it is: `pass` and everything else, never a bare tick. The
     class is on the word so a reader who prints the page in grey still reads the word. */
  function statusWord(value, cls) {
    var s = String(value === null || value === undefined ? 'not recorded' : value);
    var ok = s === 'pass' || s === 'passed' || s === 'ok';
    return el('span', { class: 'bn-status ' + (ok ? 'bn-ok' : 'bn-notok') + ' ' + (cls || ''),
                        'data-status': s, text: s });
  }

  /* ---------------- the paired difference against zero. Nothing is recomputed: the interval is
     the one the evaluation recorded, placed on an axis that always contains zero, because where
     the interval sits relative to zero is the whole of the verdict. */
  function intervalBar(diff, ci, label) {
    var w = 340, h = 54, pad = 26;
    var node = svg('svg', { viewBox: '0 0 ' + w + ' ' + h, class: 'bn-ci-bar', role: 'img',
                            preserveAspectRatio: 'xMidYMid meet',
                            'aria-label': label || 'paired difference against zero' });
    var lo = isArr(ci) && isNum(ci[0]) ? ci[0] : diff;
    var hi = isArr(ci) && isNum(ci[1]) ? ci[1] : diff;
    if (!isNum(lo) || !isNum(hi)) return null;
    node.setAttribute('data-lo', lo);
    node.setAttribute('data-hi', hi);
    var a = Math.min(lo, 0), b = Math.max(hi, 0), span = b - a;
    if (!(span > 0)) span = Math.abs(a) || 1;
    a -= span * 0.18; b += span * 0.18;
    function sx(v) { return pad + (v - a) / (b - a) * (w - 2 * pad); }
    var y = 26;
    node.appendChild(svg('line', { x1: pad, y1: y, x2: w - pad, y2: y, stroke: 'currentColor',
                                   'stroke-width': 1, opacity: .3 }));
    node.appendChild(svg('line', { x1: sx(0).toFixed(1), y1: 8, x2: sx(0).toFixed(1), y2: 40,
                                   stroke: 'currentColor', opacity: .55,
                                   'stroke-dasharray': '3 3' }));
    node.appendChild(svg('text', { x: sx(0).toFixed(1), y: 51, 'text-anchor': 'middle',
                                   'font-size': 9, fill: 'currentColor', opacity: .7,
                                   text: '0' }));
    node.appendChild(svg('rect', { x: Math.min(sx(lo), sx(hi)).toFixed(1), y: y - 5,
                                   width: Math.max(2, Math.abs(sx(hi) - sx(lo))).toFixed(1),
                                   height: 10, rx: 2, fill: 'currentColor', opacity: .28 }));
    if (isNum(diff))
      node.appendChild(svg('circle', { cx: sx(diff).toFixed(1), cy: y, r: 3.4,
                                       fill: 'currentColor' }));
    node.appendChild(svg('text', { x: pad, y: 16, 'font-size': 9, fill: 'currentColor',
                                   opacity: .7, text: fmt(lo) }));
    node.appendChild(svg('text', { x: w - pad, y: 16, 'text-anchor': 'end', 'font-size': 9,
                                   fill: 'currentColor', opacity: .7, text: fmt(hi) }));
    return node;
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
    var pane = el('section', { class: 'layer-bundle tab-pane', 'data-layer': 'bundle',
                               'data-tab': 'bundle', id: 'tab-bundle', role: 'tabpanel',
                               'aria-label': 'Bundle' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'bundle', 'data-layer': 'bundle', role: 'tab',
                             text: 'Bundle' });
      b.onclick = function () { switchTab('bundle'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  /* ---------------- a sortable table head. Sorting is a reading aid; it never reorders the
     record, and the arrow says which column and which direction the reader chose. Keyboard and
     pointer reach the same function: a table only a mouse can sort is not operable. */
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
        var rs = Array.prototype.slice.call(body.querySelectorAll('tr'));
        rs.sort(function (x, y) {
          var cx = x.children[i], cy = y.children[i];
          var vx = cx && cx.getAttribute('data-value'), vy = cy && cy.getAttribute('data-value');
          var nx = vx === null ? NaN : parseFloat(vx), ny = vy === null ? NaN : parseFloat(vy);
          if (i >= numericFrom && (isFinite(nx) || isFinite(ny))) {
            if (!isFinite(nx)) return 1;          /* a row without a number sinks either way */
            if (!isFinite(ny)) return -1;
            return dir === 'ascending' ? nx - ny : ny - nx;
          }
          var sx = (cx ? cx.textContent : ''), sy = (cy ? cy.textContent : '');
          return dir === 'ascending' ? sx.localeCompare(sy) : sy.localeCompare(sx);
        });
        rs.forEach(function (r) { body.appendChild(r); });
      }
      th.onclick = sort;
      th.onkeydown = function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); }
      };
    });
  }

  ready(function () {
    /* the guard at the top catches a script injected into an already-rendered page; this one
       catches two copies parsed before the document was ready, which both mount otherwise */
    if (document.querySelector('[data-layer="bundle"]')) return;
    var s = document.getElementById('layer-bundle-data');
    var pane = mount();
    pane.appendChild(el('header', { class: 'bn-head' }, [el('h2', { text: 'Bundle' })]));
    var M = null;
    try { M = s ? JSON.parse(s.textContent) : null; } catch (e) { M = null; }
    if (!M || typeof M !== 'object') {
      pane.appendChild(para(s ? 'The bundle layer data did not parse; there is nothing here to '
                              + 'say what ships. The bundle directory on disk, and its '
                              + 'MANIFEST.json, are still the record.'
                              : 'This file carries a bundle layer marker but no data script.',
                            'bn-empty'));
      window.__bundle = { version: VERSION, manifest: function () { return null; },
                          schema: function () { return null; },
                          roundtrip: function () { return null; },
                          findings: function () { return []; },
                          markers: function () { return []; } };
      return;
    }

    var MAN = M.manifest || {}, CHAIN = MAN.chain || {}, EV = CHAIN.evaluation || {},
        VERD = EV.verdict || {}, PRIM = EV.primary || {},
        SCH = M.input_schema || {}, RT = M.roundtrip || MAN.roundtrip || {},
        TOL = RT.tolerance || {}, EXA = MAN.examples || {},
        EXP = isArr(M.exports) ? M.exports : (isArr(MAN.exports) ? MAN.exports : []),
        SRV = M.serve || MAN.serve || {}, FIND = isArr(M.code_findings) ? M.code_findings : [],
        CARD = M.model_card || {}, NB = M.notebook || {},
        USE = M.usage || MAN.usage || {}, READ = M.readings || {},
        MARK = isArr(M.markers) ? M.markers : [],
        STANCES = isArr(MAN.stances) ? MAN.stances : [],
        FILES = MAN.files || {};
    var INPUTS = isArr(SCH.input) ? SCH.input : [];
    var EXPOSURES = isNum(EV.seal_exposures) ? EV.seal_exposures : null;

    /* ---- the verdict, first and whole. Three states, because "a baseline is better" is a
       different claim from "no difference was shown", and a reader who is given the second
       when the record says the first has been told the wrong thing. */
    var earned = VERD.earned === true;
    var beaten = VERD.beaten_by_baseline === true;
    var state = earned ? 'earned' : beaten ? 'beaten' : 'not-earned';
    var vbox = el('section', { class: 'bn-verdict bn-v-' + state, 'data-view': 'verdict',
                               'data-verdict': state });
    var flagText = earned ? 'Earned'
                 : beaten ? 'Not earned: beaten by a baseline'
                 : 'Not earned';
    vbox.appendChild(el('p', { class: 'bn-flagline' }, [
      el('span', { class: 'bn-flag', 'data-verdict': state, text: flagText }),
      document.createTextNode(' — ' + fmt(MAN.slug) + ' ' + fmt(MAN.run)
        + (EV.task ? ', a ' + fmt(EV.task) + ' model' : '')
        + (CHAIN.label ? ' for ' + fmt(CHAIN.label) : '') + '.')]));
    var base = fmt(VERD.strongest_baseline);
    var pname = fmt(PRIM.name || 'the primary metric');
    var sentence;
    if (earned) {
      sentence = 'On ' + pname + ' the network scores ' + rec(PRIM.value) + ' '
        + ciText(PRIM.ci, rec) + ', and the paired difference against the strongest fitted '
        + 'baseline, ' + base + ', is ' + rec(VERD.diff) + ' ' + ciText(VERD.ci, rec)
        + ' — an interval that excludes zero in the network’s favour.';
    } else if (beaten) {
      sentence = 'On ' + pname + ' the network scores ' + rec(PRIM.value) + ' '
        + ciText(PRIM.ci, rec) + ', but the paired difference against ' + base + ' is '
        + rec(VERD.diff) + ' ' + ciText(VERD.ci, rec) + ' — an interval that excludes zero on the '
        + 'wrong side. This is not an undecided comparison: on the held-out part ' + base
        + ' is better than the network, and this bundle ships the network anyway.';
    } else {
      sentence = 'On ' + pname + ' the network scores ' + rec(PRIM.value) + ' '
        + ciText(PRIM.ci, rec) + ', and the paired difference against the strongest fitted '
        + 'baseline, ' + base + ', is ' + rec(VERD.diff) + ' ' + ciText(VERD.ci, rec)
        + ' — an interval that covers zero, so the held-out part does not separate them. '
        + base + ' is the cheaper model and the comparison did not rule it out.';
    }
    vbox.appendChild(para(sentence, 'bn-verdict-text'));
    var bar = intervalBar(VERD.diff, VERD.ci, 'network minus ' + base + ' on ' + pname
                                              + ', against zero');
    if (bar) vbox.appendChild(el('figure', { class: 'bn-figure' }, [bar,
      el('figcaption', { text: 'network − ' + base + ' on ' + pname
                               + ', with zero marked' })]));

    /* the stance is the condition under which an unearned bundle ships. It sits inside the
       verdict box: folded away, it would read as an endorsement with a footnote. */
    var stanceBox = el('div', { class: 'bn-stances', 'data-view': 'stances' });
    if (STANCES.length) {
      stanceBox.appendChild(para('It ships under ' + STANCES.length + ' recorded stance'
        + (STANCES.length === 1 ? '' : 's') + ':', 'bn-stance-lead'));
      var sul = el('ul', { class: 'bn-stance-list' });
      STANCES.forEach(function (st) { sul.appendChild(el('li', { text: String(st) })); });
      stanceBox.appendChild(sul);
    } else if (!earned) {
      stanceBox.appendChild(para('No stance is recorded. An unearned network with no recorded '
        + 'reason for shipping is a bundle nobody has taken responsibility for.',
        'bn-stance-lead bn-warn'));
    } else {
      stanceBox.appendChild(para('No stance was needed: the verdict was earned.',
                                 'bn-stance-lead'));
    }
    vbox.appendChild(stanceBox);
    if (EXPOSURES !== null)
      vbox.appendChild(para('The held-out part was opened ' + EXPOSURES + ' time'
        + (EXPOSURES === 1 ? '' : 's') + (EXPOSURES > 1
           ? ' — every number above was read after the test part had already been seen.'
           : ', for the evaluation these numbers come from.'),
        'bn-exposures' + (EXPOSURES > 1 ? ' bn-warn' : '')));
    if (READ.abstract) vbox.appendChild(para(String(READ.abstract), 'bn-abstract'));
    pane.appendChild(vbox);

    /* ---- the manifest summary, and the digests folded into it */
    var strip = el('dl', { class: 'bn-summary', 'data-view': 'summary' });
    function pair(label, value, title) {
      if (value === null || value === undefined || value === '') return;
      var d = el('div', { class: 'bn-pair' }, [el('dt', { text: label }),
                                               el('dd', { text: fmt(value) })]);
      if (title) d.setAttribute('title', title);
      strip.appendChild(d);
    }
    pair('slug', MAN.slug);
    pair('run', MAN.run);
    pair('campaign', MAN.campaign);
    pair('task', EV.task || (MAN.output || {}).kind);
    pair('output', isArr((MAN.output || {}).fields) ? MAN.output.fields.join(', ') : null);
    if (isNum((MAN.output || {}).threshold))
      pair('threshold', MAN.output.threshold, (MAN.output.threshold_source || '') +
           ' — every decision this bundle makes moves with it');
    pair('backend', fmt((MAN.backend || {}).name) + ' ' + fmt((MAN.backend || {}).version));
    pair('python', MAN.python);
    pair('built', MAN.created_at);
    pane.appendChild(strip);

    var dig = details('digests — ' + Object.keys(FILES).length + ' file'
                      + (Object.keys(FILES).length === 1 ? '' : 's') + ' as shipped',
                      'bn-digests');
    dig.setAttribute('data-view', 'digests');
    var dt = table(['file', 'sha256'], 'bn-files');
    Object.keys(FILES).forEach(function (name) {
      rows(dt).appendChild(el('tr', null, [
        el('th', { scope: 'row', text: name }),
        el('td', { class: 'bn-digest', text: String(FILES[name]) })]));
    });
    dig.appendChild(dt);
    var chainDigests = [['the evaluated model page', CHAIN.model_digest],
                        ['netspec', (CHAIN.architecture || {}).netspec_digest],
                        ['shape manifest', (CHAIN.shape || {}).manifest_digest],
                        ['recipe', (CHAIN.shape || {}).recipe_digest]]
                       .filter(function (r) { return r[1]; });
    if (chainDigests.length) {
      var ct = table(['what the chain recorded', 'sha256'], 'bn-files');
      chainDigests.forEach(function (r) {
        rows(ct).appendChild(el('tr', null, [el('th', { scope: 'row', text: r[0] }),
                                             el('td', { class: 'bn-digest', text: r[1] })]));
      });
      dig.appendChild(ct);
    }
    dig.appendChild(para('MANIFEST.files is checkable from inside the bundle: '
      + 'tests/test_roundtrip.py re-hashes each of these without needing the chain that built '
      + 'them.', 'bn-note'));
    pane.appendChild(dig);

    /* ---- the round trip: the claim the stage rests on */
    var rtSec = section('roundtrip', 'The round trip', 'bn-panel');
    var head = el('p', { class: 'bn-rt-head' }, [
      document.createTextNode('Raw rows through the bundle’s own preprocess and network '
        + 'reproduce the scores the evaluation stored: '),
      statusWord(RT.status, 'bn-rt-status'),
      document.createTextNode(' on ' + fmt(RT.rows) + ' row'
        + (RT.rows === 1 ? '' : 's') + ' from ' + fmt(RT.source) + '.')]);
    head.setAttribute('data-check', 'roundtrip');
    rtSec.appendChild(head);
    var tolLine = el('p', { class: 'bn-tolerance' }, [
      document.createTextNode('Held to '),
      code('|Δ| ≤ ' + fmt(TOL.value)),
      document.createTextNode(' — the tolerance is a formula, '),
      el('code', { class: 'bn-code bn-rule', text: String(TOL.rule || 'not recorded') }),
      document.createTextNode(', evaluated at ' + fmt(TOL.precision)
        + (isNum(TOL.scale) ? ' on a scale of ' + fmt(TOL.scale) : '')
        + '. The worst difference actually seen was ' + rec(RT.max_abs_diff) + '.')]);
    tolLine.setAttribute('data-check', 'tolerance');
    rtSec.appendChild(tolLine);
    var shp = el('p', { class: 'bn-shaper' }, [
      document.createTextNode('The shaper’s twin check on features/preprocess.py: '),
      statusWord(RT.shaper_transform_check),
      document.createTextNode(' — a separate result from the round trip above, and the '
        + 'only test of the transform that does not go through the network.')]);
    shp.setAttribute('data-check', 'shaper');
    rtSec.appendChild(shp);
    if (RT.first_disagreement) {
      var fd = details('the first row that disagreed');
      var ft = table(['field', 'value']);
      Object.keys(RT.first_disagreement).forEach(function (k) {
        rows(ft).appendChild(el('tr', null, [el('th', { scope: 'row', text: k }),
                                             el('td', { text: fmt(RT.first_disagreement[k]) })]));
      });
      fd.appendChild(ft);
      rtSec.appendChild(fd);
    }
    pane.appendChild(rtSec);

    /* ---- what the shipped test proves. `test` and `synthetic` are different guarantees. */
    var kind = String(EXA.kind || 'unknown');
    var exSec = section('examples', 'What the shipped test proves',
                        'bn-panel' + (kind === 'synthetic' ? ' bn-synthetic' : ''));
    exSec.setAttribute('data-kind', kind);
    /* what the shipped test proves, said once in a form a checker can read and once in prose */
    exSec.setAttribute('data-proves', kind === 'test' ? 'the evaluation'
                                    : kind === 'synthetic' ? 'itself' : 'unrecorded');
    exSec.appendChild(el('p', null, [
      document.createTextNode('examples/rows.csv holds '),
      el('strong', { text: fmt(EXA.rows) + ' ' + kind + ' row'
                           + (EXA.rows === 1 ? '' : 's') }),
      document.createTextNode(' from ' + fmt(EXA.source) + '.')]));
    if (kind === 'test')
      exSec.appendChild(para('They are rows of the held-out test part, and '
        + 'examples/expected.parquet holds the scores /vega-eval stored for them. '
        + 'tests/test_roundtrip.py therefore proves the bundle reproduces the evaluation.'));
    else if (kind === 'synthetic')
      exSec.appendChild(para('The test rows do not ship, so examples/expected.parquet holds '
        + 'the predictor’s own outputs at pack time. tests/test_roundtrip.py therefore '
        + 'proves only that the bundle reproduces itself — it does not prove that the bundle '
        + 'reproduces the evaluation. The pack-time round trip above is where that was '
        + 'checked, and it does not travel with the bundle.', 'bn-warn'));
    else
      exSec.appendChild(para('The manifest does not say which kind of rows these are, so what '
        + 'the shipped test proves is not recorded.', 'bn-warn'));
    pane.appendChild(exSec);

    /* ---- the input schema. A column without its domain invites silent extrapolation. */
    var scSec = section('schema', 'Input schema', '');
    var st = table(['column', 'dtype', { text: 'required', cls: 'bn-num' },
                    'seen in training', 'note'], 'bn-schema');
    INPUTS.forEach(function (c) {
      var tr = el('tr', { 'data-column': String(c.name) }, [
        el('th', { scope: 'row', text: fmt(c.name), 'data-value': String(c.name) }),
        el('td', { text: fmt(c.dtype), 'data-value': String(c.dtype) }),
        el('td', { class: 'bn-num', text: c.required ? 'required' : 'optional',
                   'data-value': c.required ? 1 : 0 })]);
      var dom = el('td', { class: 'bn-dom' });
      if (isArr(c.allowed)) {
        var n = isNum(c.distinct_seen) ? c.distinct_seen : c.allowed.length;
        dom.setAttribute('data-domain', 'allowed:' + n);
        dom.setAttribute('data-value', n);
        dom.appendChild(el('span', { text: n + ' value' + (n === 1 ? '' : 's') + ': ' }));
        dom.appendChild(el('span', { class: 'bn-vals',
                                     text: c.allowed.slice(0, 5).map(fmt).join(', ')
                                           + (c.allowed.length > 5 ? ' …' : '') }));
        if (c.allowed.length > 5) {
          var dd = details('all ' + c.allowed.length);
          dd.appendChild(el('p', { class: 'bn-vals', text: c.allowed.map(fmt).join(', ') }));
          dom.appendChild(dd);
        }
      } else if (isArr(c.range_seen) && c.range_seen.length === 2) {
        dom.setAttribute('data-domain', 'range:' + c.range_seen[0] + '..' + c.range_seen[1]);
        dom.setAttribute('data-value', c.range_seen[1]);
        dom.appendChild(el('span', { text: fmt(c.range_seen[0]) + ' … '
                                           + fmt(c.range_seen[1]) }));
      } else if (typeof c.allowed === 'string') {
        dom.setAttribute('data-domain', 'allowed:' + c.allowed);
        dom.appendChild(el('span', { text: String(c.allowed) }));
      } else {
        dom.className += ' bn-none';
        dom.appendChild(el('span', { text: '· not recorded' }));
      }
      tr.appendChild(dom);
      tr.appendChild(el('td', { class: 'bn-note', text: c.note ? String(c.note) : '' }));
      rows(st).appendChild(tr);
    });
    makeSortable(st, 2);
    scSec.appendChild(st);
    scSec.appendChild(para('predict.py validates rows against this schema. A value outside the '
      + 'range seen in training is answered and flagged, not refused — the model card says so, '
      + 'because extrapolation past these bounds was never measured.', 'bn-note'));
    var OUT = SCH.output || {};
    if (Object.keys(OUT).length) {
      var ot = table(['field', 'what it is'], 'bn-output');
      Object.keys(OUT).forEach(function (k) {
        rows(ot).appendChild(el('tr', { 'data-field': k }, [
          el('th', { scope: 'row', text: k }), el('td', { text: fmt(OUT[k]) })]));
      });
      scSec.appendChild(el('h4', { text: 'Output' }));
      scSec.appendChild(ot);
    }
    if (isArr(SCH.ignored) && SCH.ignored.length)
      scSec.appendChild(el('p', { class: 'bn-ignored' }, [
        document.createTextNode('Ignored, and not required on input: '),
        el('span', { class: 'bn-code', text: SCH.ignored.join(', ') }),
        document.createTextNode('.')]));
    pane.appendChild(scSec);

    /* ---- exports and serving */
    var exSecOut = section('exports', 'Exports and serving', '');
    if (EXP.length) {
      var et = table(['format', 'file', { text: 'parity atol', cls: 'bn-num' }, 'status'],
                     'bn-exports');
      EXP.forEach(function (e) {
        var ok = String(e.status) === 'pass';
        var tr = el('tr', ok ? { 'data-format': String(e.format) }
                             : { class: 'bn-warn-row', 'data-format': String(e.format) }, [
          el('th', { scope: 'row', text: fmt(e.format) }),
          el('td', { class: 'bn-code', text: fmt(e.file) }),
          el('td', { class: 'bn-num', text: fmt(e.parity_atol), 'data-value': e.parity_atol })]);
        var td = el('td');
        td.appendChild(statusWord(e.status));
        if (!ok) td.appendChild(el('span', { text: ' — this export ships marked; call predict.py '
                                                   + 'if the numbers must match the evaluation' }));
        tr.appendChild(td);
        rows(et).appendChild(tr);
      });
      exSecOut.appendChild(et);
    } else {
      exSecOut.appendChild(para('No exports were requested, so predict.py is the only entry '
        + 'point and there is no second copy of the network to keep in step.', 'bn-note'));
    }
    var serveLine = SRV.present
      ? 'A serving scaffold is present: mode ' + fmt(SRV.mode) + ', ' + fmt(SRV.framework)
        + ', ' + fmt(SRV.endpoint) + ' (from ' + fmt(SRV.source) + ').'
      : 'No serving scaffold: mode ' + fmt(SRV.mode) + ' — ' + fmt(SRV.endpoint) + '.';
    var sp = para(serveLine + (SRV.note ? ' Note: ' + SRV.note : ''), 'bn-serve');
    sp.setAttribute('data-view', 'serve');
    sp.setAttribute('data-mode', String(SRV.mode || 'unknown'));
    exSecOut.appendChild(sp);
    pane.appendChild(exSecOut);

    /* ---- the audits' findings on the serving slice, and how the audits judged them */
    var fSec = section('findings', 'Findings on the code that ships', '');
    var JUDGE = M.conformance || {};
    ['architect', 'improve'].forEach(function (tool) {
      var j = JUDGE[tool];
      if (!j || typeof j !== 'object') return;
      var left = j.unconfirmed || {};
      var adv = j.advisory || {}, items = isArr(adv.open_items) ? adv.open_items : [];
      var jp = para(tool + ': ' + (isArr(j.audits) ? j.audits.length : 0) + ' audit(s); confirmed '
        + fmt(j.structural) + ' structural, ' + fmt(j.divergence) + ' divergence; unconfirmed '
        + '(recorded, not blocking) ' + fmt(left.structural) + ' structural, '
        + fmt(left.divergence) + ' divergence.' + (items.length ? ' Advisory (' + fmt(adv.mode)
        + ', ' + fmt(adv.outcome) + '): ' + items.length + ' open item(s) for /architect and '
        + '/improve.' : ''), 'bn-note');
      jp.setAttribute('data-judgement', tool);
      if (items.length) jp.setAttribute('data-open-items', String(items.length));
      fSec.appendChild(jp);
      items.forEach(function (it) {
        var ip = para('Open item: ' + fmt(it.text), 'bn-note');
        ip.setAttribute('data-open-item', fmt(it.finding));
        fSec.appendChild(ip);
      });
    });
    if (FIND.length) {
      var ft2 = table(['id', 'tool', 'severity', 'where', 'what', 'audits', 'next'], 'bn-findings');
      FIND.forEach(function (f) {
        var sev = String(f.severity || f.depth || 'unrecorded');
        var qr = f.quorum ? ((f.quorum.confirmed ? 'confirmed by ' : 'unconfirmed: only ')
          + (isArr(f.quorum.reported_by) ? f.quorum.reported_by.join(', ') : '·'))
          : (f.audit ? 'audit ' + f.audit : '·');
        rows(ft2).appendChild(el('tr', { 'data-finding': String(f.id), 'data-severity': sev }, [
          el('th', { scope: 'row', text: fmt(f.id) }),
          el('td', { class: 'bn-code', text: fmt(f.tool || 'architect') }),
          el('td', { class: 'bn-sev bn-sev-' + sev.replace(/[^a-z]/gi, ''), text: sev }),
          el('td', { class: 'bn-code', text: fmt(f.where || f.file) }),
          el('td', { text: fmt(f.title) }),
          el('td', { class: 'bn-code', text: qr }),
          el('td', { class: 'bn-code', text: f.next ? String(f.next) : '·' })]));
      });
      fSec.appendChild(ft2);
      fSec.appendChild(para('These are findings on the serving slice as it ships, not on the '
        + 'training code that was left behind. A structural or divergence finding that two of '
        + 'three independent audits report is confirmed and advisory: an open item for /architect '
        + 'and /improve, published unattended and, attended, only on the user\'s answer; one only '
        + 'a single audit reported is listed as unconfirmed, and the rest are tactical or '
        + 'architectural.', 'bn-note'));
    } else {
      fSec.appendChild(para('The audits recorded no findings on the serving slice.',
                            'bn-note'));
    }
    pane.appendChild(fSec);

    /* ---- how to use it, and where the card is */
    var uSec = section('usage', 'How to use it', '');
    if (USE.cli) uSec.appendChild(el('pre', { class: 'bn-snippet', 'data-usage': 'cli',
                                              text: String(USE.cli) }));
    if (USE.module) uSec.appendChild(el('pre', { class: 'bn-snippet', 'data-usage': 'module',
                                                 text: String(USE.module) }));
    /* a full bundle carries the loop and the evaluator, so the two extra entry points and
       what they are not belong here rather than only on the card */
    if (USE.train || USE.evaluate) {
      [USE.train, USE.evaluate].forEach(function (line) {
        if (line) uSec.appendChild(el('pre', { class: 'bn-snippet', 'data-usage': 'full',
                                               text: String(line) }));
      });
      uSec.appendChild(para('This bundle was packed with --include full: the whole emitted '
        + 'package ships. Neither extra entry point is the stage it echoes — train.py runs one '
        + 'spec with no ladder, no diagnostics and no ledger, so its checkpoint is a network '
        + 'no model card describes, and evaluate.py prints point metrics with no intervals, no '
        + 'baselines and no verdict. Both want shaped parts, which do not travel. The test '
        + 'part stays sealed in both.', 'bn-note'));
    }
    if (CARD.path) {
      var link = el('a', { class: 'bn-card-link', href: String(CARD.path),
                           'data-card': String(CARD.path),
                           text: 'model card — ' + String(CARD.path) });
      uSec.appendChild(el('p', null, [link, document.createTextNode(
        ' (' + (isNum(CARD.bytes) ? Math.round(CARD.bytes / 1024) + ' KB' : 'size not recorded')
        + '; it is a sibling file and opens on its own, without this page or any skill '
        + 'installed).')]));
      if (CARD.digest)
        uSec.appendChild(el('p', { class: 'bn-note' }, [
          document.createTextNode('card digest '),
          el('span', { class: 'bn-digest', text: String(CARD.digest) })]));
    }
    if (NB.path)
      uSec.appendChild(el('p', { class: 'bn-note', 'data-notebook': String(NB.path) }, [
        document.createTextNode('Notebook ' + String(NB.path)
          + (NB.executed ? ' (executed' : ' (NOT executed')
          + (isNum(NB.code_cells) ? ', ' + NB.code_cells + ' code cells).' : ').'))]));
    pane.appendChild(uSec);

    /* ---- the readings, as written */
    var rSec = section('readings', 'Intended use and limits', '');
    if (READ.intended_use)
      rSec.appendChild(para(String(READ.intended_use), 'bn-reading bn-intended'));
    else
      rSec.appendChild(para('No intended-use prose was recorded. The model card is the '
        + 'document that carries it.', 'bn-warn'));
    pane.appendChild(rSec);

    /* ---- provenance and the markers pack.py wrote */
    var prov = details('how this bundle was produced — ' + MARK.length + ' marker'
                       + (MARK.length === 1 ? '' : 's'));
    prov.setAttribute('data-view', 'provenance');
    prov.appendChild(para('Bundle ' + fmt(MAN.slug) + '-' + fmt(MAN.run) + ' of campaign '
      + fmt(MAN.campaign) + ', built ' + fmt(MAN.created_at) + ' against '
      + fmt((MAN.backend || {}).name) + ' ' + fmt((MAN.backend || {}).version)
      + ' on python ' + fmt(MAN.python) + '. Selection rule: '
      + fmt((CHAIN.training || {}).selection_rule) + '.'));
    var conf = (MAN.blueprint || {}).conformance || {};
    if (Object.keys(conf).length)
      prov.appendChild(para('Blueprint conformance — mechanical: ' + fmt(conf.mechanical)
        + ['architect', 'improve'].map(function (tool) {
          var j = (M.conformance || {})[tool];
          return '; ' + tool + ': ' + (j && typeof j === 'object'
            ? fmt(j.structural) + ' structural, ' + fmt(j.divergence) + ' divergence confirmed, '
              + fmt(j.recorded) + ' recorded' : 'not run');
        }).join('') + '.'));
    if (MARK.length) {
      var ul = el('ul', { class: 'bn-markers' });
      MARK.forEach(function (m) {
        var t = String(m);
        ul.appendChild(el('li', { class: t.indexOf('WARN') === 0 ? 'bn-m-warn'
                                         : t.indexOf('ERROR') === 0 ? 'bn-m-error' : '',
                                  text: t }));
      });
      prov.appendChild(ul);
    }
    pane.appendChild(prov);

    window.__bundle = {
      version: VERSION,
      manifest: function () { return MAN; },
      schema: function () { return SCH; },
      roundtrip: function () { return RT; },
      findings: function () { return FIND; },
      markers: function () { return MARK; }
    };
  });
})();
