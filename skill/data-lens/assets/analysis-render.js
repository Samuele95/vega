/* data-lens — `analysis` layer render script (contract: references/analysis-contract.md §7).
   Vanilla JS, no libraries, no network. Reads ONLY #layer-analysis-data. Mounts one tab
   (data-layer="analysis") with four surfaces:
     Context strip — what the geometry layer settled, as read-only chips.
     Findings board — severity-ordered cards: evidence, method with its assumption verdict,
       the so-what, the linked figures, the transformation candidates.
     Module panels — one collapsible panel per module; a skipped module states its reason.
     Transcript — the dialogue turns: question, method, folded code, result, answer, caveats.
   It computes nothing. Numbers are shipped; figures are shipped SVG; the "re-run" button
   re-displays the stored code and result, because a page cannot execute Python. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="analysis"]')) return;

  var VERSION = 'data-lens/analysis-render@1';
  var SEV = { high: 3, medium: 2, low: 1 };

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
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (!isNum(v)) return String(v);
    if (v === Math.round(v) && Math.abs(v) < 1e15) return String(v);
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
    return String(Math.round(v * 1e4) / 1e4);
  }
  function chip(text, cls, title) {
    var c = el('span', { class: 'a-chip ' + (cls || ''), text: text });
    if (title) c.setAttribute('title', title);
    return c;
  }
  function para(text, cls) { return el('p', { class: cls || 'a-prose', text: text }); }
  function keyed(fn) {
    return function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(e); } };
  }
  /* SVG arrives as a string in the layer data. Parse it, drop anything executable, and
     import the root element — never innerHTML. */
  function svgFrom(str) {
    if (!str || typeof str !== 'string') return null;
    try {
      var doc = new DOMParser().parseFromString(str, 'image/svg+xml');
      var root = doc.documentElement;
      if (!root || root.nodeName.toLowerCase() === 'parsererror') return null;
      Array.prototype.slice.call(root.querySelectorAll('script,foreignObject')).forEach(
        function (n) { n.parentNode.removeChild(n); });
      Array.prototype.slice.call(root.querySelectorAll('*')).forEach(function (n) {
        Array.prototype.slice.call(n.attributes || []).forEach(function (at) {
          if (/^on/i.test(at.name) || /^(href|xlink:href)$/i.test(at.name) &&
              /^\s*javascript:/i.test(at.value)) n.removeAttribute(at.name);
        });
      });
      var imported = document.importNode(root, true);
      imported.removeAttribute('width');
      imported.removeAttribute('height');
      imported.setAttribute('class', 'a-svg');
      return imported;
    } catch (e) { return null; }
  }
  function details(summaryText, cls) {
    var d = el('details', { class: cls || 'a-det' });
    d.appendChild(el('summary', { text: summaryText }));
    return d;
  }
  function table(headers, rows, cls) {
    var t = el('table', { class: 'a-table ' + (cls || '') });
    var thead = el('thead'), tr = el('tr');
    headers.forEach(function (h) { tr.appendChild(el('th', { text: h })); });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = el('tbody');
    rows.forEach(function (r) {
      var row = el('tr');
      r.forEach(function (c) {
        row.appendChild(c && c.nodeType ? el('td', null, [c]) : el('td', { text: fmt(c) }));
      });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return t;
  }
  /* Sortable: click a header to sort the rows by that column, numeric when it parses. */
  function sortable(t) {
    var ths = t.querySelectorAll('thead th');
    Array.prototype.forEach.call(ths, function (th, i) {
      th.setAttribute('tabindex', '0');
      th.classList.add('a-sortable');
      var dir = 1;
      var go = function () {
        var tb = t.querySelector('tbody');
        var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'));
        rows.sort(function (a, b) {
          var x = a.children[i] ? a.children[i].textContent : '';
          var y = b.children[i] ? b.children[i].textContent : '';
          var nx = parseFloat(x), ny = parseFloat(y);
          if (!isNaN(nx) && !isNaN(ny)) return (nx - ny) * dir;
          return x.localeCompare(y) * dir;
        });
        dir = -dir;
        rows.forEach(function (r) { tb.appendChild(r); });
      };
      th.onclick = go;
      th.onkeydown = keyed(go);
    });
    return t;
  }

  /* ---------------------------------------------------------------- host tab plumbing */
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
    var pane = el('section', { class: 'layer-analysis tab-pane', 'data-layer': 'analysis',
                               'data-tab': 'analysis', id: 'tab-analysis', role: 'tabpanel',
                               'aria-label': 'Analysis' });
    var sib = pickPaneSibling();
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'analysis', 'data-layer': 'analysis', role: 'tab',
                             text: 'Analysis' });
      b.onclick = function () { switchTab('analysis'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  function readData() {
    var s = document.getElementById('layer-analysis-data');
    if (!s) return { error: 'missing' };
    try { return { data: JSON.parse(s.textContent) }; }
    catch (e) { return { error: 'unparseable', detail: String(e) }; }
  }

  ready(function () {
    var read = readData();
    var pane = mount();
    var head = el('header', { class: 'a-head' }, [el('h2', { text: 'Analysis' })]);
    pane.appendChild(head);
    if (read.error) {
      pane.appendChild(para(read.error === 'missing'
        ? 'This file carries an analysis layer marker but no data script; nothing to show.'
        : 'The analysis layer data did not parse: ' + (read.detail || ''), 'a-empty'));
      return;
    }
    var M = read.data || {};
    var src = M.source || {}, ctx = M.context || {}, mods = M.modules || {};
    var findings = (M.findings || []).slice();
    var figIndex = {};
    (M.figures || []).forEach(function (f) { if (f && f.id) figIndex[f.id] = f; });

    head.appendChild(el('p', { class: 'a-sub', text:
      (src.path || 'dataset') + ' — ' + (src.rows !== undefined ? src.rows + ' rows, ' : '') +
      (src.columns !== undefined ? src.columns + ' columns, ' : '') +
      'geometry ' + (src.geometry || '?') + (src.seed !== undefined ? ', seed ' + src.seed : '') }));

    /* ---------------- context strip */
    var strip = el('div', { class: 'a-strip', 'data-view': 'context' });
    strip.appendChild(el('h3', { text: 'What this pass inherits' }));
    var chips = el('div', { class: 'a-chips' });
    var P = ctx.partition || {};
    chips.appendChild(chip('basis: ' + ((ctx.basis || []).join(', ') || '—'), 'a-basis',
                           'the primitive columns the geometry layer settled'));
    if (P.label) {
      chips.appendChild(chip('label: ' + P.label + (P.task ? ' (' + P.task + ')' : ''), 'a-label'));
      if ((P.leakage || []).length)
        chips.appendChild(chip('leakage dropped: ' + P.leakage.join(', '), 'a-leak',
                               'columns that derive the label and must stay out of the features'));
      if (P.provenance) chips.appendChild(chip('partition ' + P.provenance, 'a-muted-chip'));
    } else {
      chips.appendChild(chip('no partition chosen', 'a-muted-chip'));
    }
    (ctx.derivations || []).forEach(function (d) {
      chips.appendChild(chip(d.column + ' ← ' + (d.body || []).join(', '), 'a-deriv',
                             d.formula || d.rule_id || ''));
    });
    if (ctx.time && ctx.time.column) chips.appendChild(chip('time: ' + ctx.time.column, 'a-time'));
    if (ctx.spatial && (ctx.spatial.columns || []).length)
      chips.appendChild(chip('spatial: ' + ctx.spatial.columns.map(function (c) {
        return c.column + (c.pair ? '/' + c.pair : ''); }).join(', ') +
        ' (crs ' + (ctx.spatial.crs || 'unknown') + ')', 'a-geo'));
    strip.appendChild(chips);
    if (ctx.reading) strip.appendChild(para(ctx.reading));
    pane.appendChild(strip);

    /* ---------------- findings board */
    var board = el('div', { class: 'a-board', 'data-view': 'findings' });
    board.appendChild(el('h3', { text: 'Findings' }));
    var bar = el('div', { class: 'a-filter' });
    var state = { severity: 'all', module: 'all' };
    function sel(label, opts, key) {
      var s = el('select', { 'aria-label': label });
      opts.forEach(function (o) { s.appendChild(el('option', { value: o, text: o })); });
      s.onchange = function () { state[key] = s.value; drawFindings(); };
      var w = el('label', { class: 'a-sel', text: label + ' ' });
      w.appendChild(s);
      return w;
    }
    var modNames = Object.keys(mods);
    bar.appendChild(sel('severity', ['all', 'high', 'medium', 'low'], 'severity'));
    bar.appendChild(sel('module', ['all'].concat(modNames), 'module'));
    board.appendChild(bar);
    var cards = el('div', { class: 'a-cards' });
    board.appendChild(cards);
    pane.appendChild(board);

    function methodLine(m) {
      if (!m) return null;
      var v = m.assumptions_checked;
      var cls = v === 'passed' ? 'a-ok' : v === 'violated' ? 'a-bad' : 'a-na';
      var row = el('div', { class: 'a-method' }, [
        el('span', { class: 'a-mname', text: m.name || 'method' }),
        chip('assumptions ' + (v || 'n/a'), cls,
             (m.assumptions || []).join('; '))]);
      if (m.switched_to) row.appendChild(chip('ran instead: ' + m.switched_to, 'a-swap'));
      if (m.correction) row.appendChild(chip(m.correction, 'a-muted-chip'));
      return row;
    }
    function kvTable(o) {
      var rows = [];
      Object.keys(o || {}).forEach(function (k) {
        var v = o[k];
        rows.push([k.replace(/_/g, ' '),
                   (v && typeof v === 'object') ? JSON.stringify(v) : fmt(v)]);
      });
      return rows.length ? sortable(table(['field', 'value'], rows, 'a-kv')) : null;
    }
    function candidateBlock(t) {
      var box = el('div', { class: 'a-cand' });
      box.appendChild(el('p', { class: 'a-candhead', text:
        (t.id ? t.id + ' · ' : '') + (t.op || 'step') + ' ' + ((t.columns || []).join(', ')) }));
      if (t.params) box.appendChild(el('pre', { class: 'a-params',
                                                text: JSON.stringify(t.params) }));
      if (t.rationale) box.appendChild(para(t.rationale, 'a-muted'));
      (t.alternatives || []).forEach(function (a) {
        box.appendChild(para('alternative: ' + (a.op || t.op) + ' ' +
          JSON.stringify(a.params || {}) + (a.when ? ' — when ' + a.when : ''), 'a-alt'));
      });
      return box;
    }
    function findingCard(f) {
      var card = el('article', { class: 'a-card a-sev-' + (f.severity || 'low'),
                                 'data-finding': f.id || '' });
      var h = el('div', { class: 'a-cardhead' }, [
        chip(f.severity || 'low', 'a-sev'),
        chip(f.module || '', 'a-mod'),
        el('h4', { text: (f.id ? f.id + ' — ' : '') + (f.title || '') })]);
      card.appendChild(h);
      if ((f.columns || []).length)
        card.appendChild(el('p', { class: 'a-cols',
                                   text: 'columns: ' + f.columns.join(', ') }));
      if (f.reading) card.appendChild(para(f.reading));
      var m = methodLine(f.method);
      if (m) card.appendChild(m);
      var evt = kvTable(f.evidence);
      if (evt) {
        var d = details('evidence');
        d.appendChild(evt);
        card.appendChild(d);
      }
      if (f.so_what) {
        var sw = el('dl', { class: 'a-sowhat' });
        Object.keys(f.so_what).forEach(function (k) {
          sw.appendChild(el('dt', { text: k.replace(/_/g, ' ') }));
          sw.appendChild(el('dd', { text: String(f.so_what[k]) }));
        });
        card.appendChild(sw);
      }
      (f.figures || []).forEach(function (fid) {
        var fg = figIndex[fid];
        if (!fg) return;
        var node = svgFrom(fg.svg);
        var wrap = el('figure', { class: 'a-fig' });
        if (node) wrap.appendChild(node);
        wrap.appendChild(el('figcaption', { text: fg.title || fid }));
        if (fg.alt) wrap.setAttribute('aria-label', fg.alt);
        card.appendChild(wrap);
      });
      var cands = f.transformation_candidates || [];
      if (cands.length) {
        var cd = details(cands.length + ' transformation candidate' + (cands.length > 1 ? 's' : ''));
        cands.forEach(function (t) { cd.appendChild(candidateBlock(t)); });
        cd.appendChild(para('These are proposals for /dataset-shaper, which applies them with ' +
                            'their provenance. This page changes no data.', 'a-muted'));
        card.appendChild(cd);
      }
      return card;
    }
    function drawFindings() {
      clear(cards);
      var list = findings.filter(function (f) {
        return (state.severity === 'all' || f.severity === state.severity) &&
               (state.module === 'all' || f.module === state.module);
      });
      list.sort(function (a, b) {
        return (SEV[b.severity] || 0) - (SEV[a.severity] || 0) ||
               String(a.id).localeCompare(String(b.id));
      });
      if (!list.length) {
        cards.appendChild(para(findings.length ? 'No finding matches this filter.'
          : 'This pass admitted no finding: nothing it measured changes a downstream decision. ' +
            'The module panels below carry the evidence anyway.', 'a-empty'));
        return;
      }
      list.forEach(function (f) { cards.appendChild(findingCard(f)); });
    }
    drawFindings();

    /* ---------------- module panels */
    var panels = el('div', { class: 'a-modules', 'data-view': 'modules' });
    panels.appendChild(el('h3', { text: 'Modules' }));
    Object.keys(mods).forEach(function (name) {
      var m = mods[name] || {};
      var d = details(name + (m.ran ? '' : ' — skipped'), 'a-det a-mod-panel');
      d.setAttribute('data-module', name);
      if (!m.ran) {
        d.appendChild(para(m.skipped_because || 'not run', 'a-muted'));
      } else {
        if (m.reading) d.appendChild(para(m.reading));
        var ev = m.evidence || {};
        Object.keys(ev).forEach(function (k) {
          var v = ev[k];
          if (Array.isArray(v) && v.length && typeof v[0] === 'object') {
            var cols = [];
            v.slice(0, 50).forEach(function (r) {
              Object.keys(r).forEach(function (c) {
                if (cols.indexOf(c) < 0 && typeof r[c] !== 'object') cols.push(c);
              });
            });
            if (!cols.length) return;
            var rows = v.slice(0, 200).map(function (r) {
              return cols.map(function (c) { return r[c]; });
            });
            var sub = details(k.replace(/_/g, ' ') + ' (' + v.length + ')');
            sub.appendChild(sortable(table(cols, rows)));
            d.appendChild(sub);
          } else if (v && typeof v === 'object' && !Array.isArray(v)) {
            var t2 = kvTable(v);
            if (t2) {
              var sub2 = details(k.replace(/_/g, ' '));
              sub2.appendChild(t2);
              d.appendChild(sub2);
            }
          } else if (!Array.isArray(v)) {
            d.appendChild(el('p', { class: 'a-kvline', text: k.replace(/_/g, ' ') + ': ' + fmt(v) }));
          }
        });
        (M.figures || []).filter(function (f) { return f.module === name; }).forEach(function (fg) {
          var node = svgFrom(fg.svg);
          if (!node) return;
          var wrap = el('figure', { class: 'a-fig' }, [node,
            el('figcaption', { text: fg.title || fg.id })]);
          d.appendChild(wrap);
        });
      }
      panels.appendChild(d);
    });
    pane.appendChild(panels);

    /* ---------------- transcript */
    var tr = el('div', { class: 'a-transcript', 'data-view': 'transcript' });
    tr.appendChild(el('h3', { text: 'Dialogue' }));
    var turns = M.transcript || [];
    if (!turns.length) {
      tr.appendChild(para('No dialogue turns in this pass. Re-run /data-lens with --continue to ' +
                          'ask the data a question and have the answer recorded here.', 'a-empty'));
    }
    turns.forEach(function (t) {
      var box = el('article', { class: 'a-turn' + (t.grounded === false ? ' a-ungrounded' : ''),
                                'data-turn': String(t.turn || '') });
      box.appendChild(el('h4', { text: (t.turn ? t.turn + '. ' : '') + (t.question || '') }));
      if (t.grounded === false) box.appendChild(chip('not grounded — no cell ran', 'a-bad'));
      var m = methodLine(t.method);
      if (m) box.appendChild(m);
      if (t.answer) box.appendChild(para(t.answer));
      (t.caveats || []).forEach(function (c) { box.appendChild(para('caveat: ' + c, 'a-alt')); });
      if (t.figure) {
        var node = svgFrom(figIndex[t.figure] ? figIndex[t.figure].svg : t.figure);
        if (node) box.appendChild(el('figure', { class: 'a-fig' }, [node]));
      }
      var runbox = el('div', { class: 'a-runbox' });
      var codeBlock = el('pre', { class: 'a-code', text: t.code || '' });
      var resBlock = el('pre', { class: 'a-result',
                                 text: t.result === undefined || t.result === null ? ''
                                       : JSON.stringify(t.result, null, 1) });
      var shown = false;
      var btn = el('button', { class: 'a-rerun', type: 'button',
                               text: 're-run (show the code and its stored result)' });
      btn.onclick = function () {
        shown = !shown;
        runbox.style.display = shown ? 'block' : 'none';
        btn.textContent = shown ? 'hide the code and result'
                                : 're-run (show the code and its stored result)';
      };
      runbox.style.display = 'none';
      runbox.appendChild(el('p', { class: 'a-muted', text:
        'This page cannot execute Python. The cell below is the exact code that ran, and the ' +
        'result is the one it returned; re-running it with data-lens/scripts/cell.py on the ' +
        'same run directory and seed reproduces it.' }));
      runbox.appendChild(codeBlock);
      if (resBlock.textContent) runbox.appendChild(resBlock);
      if (t.error) runbox.appendChild(el('pre', { class: 'a-err', text: String(t.error) }));
      box.appendChild(btn);
      box.appendChild(runbox);
      if ((t.transformation_candidates || []).length) {
        var cd = details('transformation candidates from this turn');
        t.transformation_candidates.forEach(function (c) { cd.appendChild(candidateBlock(c)); });
        box.appendChild(cd);
      }
      tr.appendChild(box);
    });
    pane.appendChild(tr);

    /* ---------------- stances and markers */
    if ((M.stances || []).length) {
      var st = details('stances recorded (' + M.stances.length + ')', 'a-det a-stances');
      M.stances.forEach(function (s) {
        st.appendChild(el('p', { class: 'a-stance', text:
          (s.id ? s.id + ' · ' : '') + (s.kind || '') + ' — ' + (s.assertion || '') +
          ((s.columns || []).length ? ' [' + s.columns.join(', ') + ']' : '') +
          (s.source ? ' (' + s.source + ')' : '') }));
      });
      pane.appendChild(st);
    }
    if ((M.markers || []).length) {
      var mk = details('run markers (' + M.markers.length + ')', 'a-det a-markers');
      M.markers.forEach(function (s) { mk.appendChild(el('p', { class: 'a-marker', text: s })); });
      pane.appendChild(mk);
    }
    if (M.handoff) {
      var ho = el('div', { class: 'a-handoff' });
      ho.appendChild(el('h3', { text: 'Hand-off' }));
      var ids = M.handoff.shaper_candidates || [];
      ho.appendChild(para(ids.length
        ? 'To /dataset-shaper, in this order: ' + ids.join(', ') + '. ' + (M.handoff.note || '')
        : 'No transformation candidate. ' + (M.handoff.note || '')));
      pane.appendChild(ho);
    }

    window.__analysis = {
      version: VERSION,
      findings: function () { return findings.slice(); },
      turns: function () { return (M.transcript || []).slice(); },
      modules: function () { return Object.keys(mods).map(function (k) {
        return { name: k, ran: !!mods[k].ran, skipped_because: mods[k].skipped_because || null }; }); },
      state: state
    };
  });
})();
