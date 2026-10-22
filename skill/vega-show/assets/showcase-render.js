/* /vega-show — `showcase` layer render script (contract: references/show-contract.md §8).
   Vanilla JS, no libraries, no network, no <img>, no <canvas>. Reads ONLY #layer-showcase-data.
   Mounts one tab (data-layer="showcase", labelled Showcase) with six surfaces, in this order:
     story      — the headline and the lead as the page renders them, the outline of the page's
                  sections with what each holds, the motif and polarity, the class roles as
                  swatches with their basis, the input groups with their member counts;
     bindings   — every number the page shows, traced: the fact key, its value, the record it
                  was read from and where the story uses it. Sortable and filterable;
     report     — linked to the compiled report or to the stage briefs (a warning: no report
                  was written), the roles bound and missing, the digests folded;
     open it    — the serve command, the URL and the container line, copyable. The page runs
                  on the bundle's own service and nowhere else;
     checks     — the build check and the served walk, its screenshots listed as paths, the
                  reason when the walk did not run;
     markers and readings — every marker of the run, a WARN marked as one, and the abstract.
   It computes nothing: every value was written by the stage's scripts. It is idempotent and
   shows an empty state when the data is missing. */
(function () {
  'use strict';
  if (document.querySelector('[data-layer="showcase"]')) return;

  var VERSION = 'vega-show/showcase-render@1';
  var SURFACES = ['story', 'bindings', 'report', 'open it', 'checks', 'markers and readings'];

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) {
      if (c) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  function isArr(v) { return Object.prototype.toString.call(v) === '[object Array]'; }
  function isObj(v) { return v !== null && typeof v === 'object' && !isArr(v); }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function num(v) {
    if (!isNum(v)) return String(v);
    if (v === Math.round(v)) return String(v);
    var a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : a >= 0.01 ? String(+v.toFixed(4))
      : String(+v.toPrecision(2));
  }
  /* a recorded value as the record holds it: a {value, ci} with its interval, a list counted */
  function val(v) {
    if (v === null || v === undefined || v === '') return '·';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (isNum(v)) return num(v);
    if (isArr(v) && !v.length) return 'none';
    if (isArr(v)) return v.length + ' item' + (v.length === 1 ? '' : 's') + ': '
      + v.slice(0, 6).map(val).join(', ') + (v.length > 6 ? ', …' : '');
    if (isObj(v) && 'value' in v) {
      var ci = isArr(v.ci) && v.ci.length === 2 ? ' [' + num(v.ci[0]) + ', ' + num(v.ci[1]) + ']' : '';
      return num(v.value) + ci + (isNum(v.n) ? ' (n ' + v.n + ')' : '');
    }
    if (isObj(v)) return JSON.stringify(v);
    return String(v);
  }
  function para(t, cls) { return el('p', { class: cls || 'sc-prose', text: t }); }
  function surface(name, title) {
    var s = el('section', { class: 'sc-surface', 'data-surface': name });
    s.appendChild(el('h3', { text: title }));
    return s;
  }
  function chip(label, value, cls) {
    return el('span', { class: 'sc-chip ' + (cls || ''), 'data-chip': label },
              [el('span', { class: 'sc-chip-k', text: label }), el('span', { text: val(value) })]);
  }
  function statusWord(value) {
    var s = String(value === null || value === undefined ? 'not recorded' : value);
    var cls = s === 'pass' || s === 'linked' ? 'sc-ok' : s === 'not-run' || s === 'brief-links' ? 'sc-warnword' : 'sc-notok';
    return el('span', { class: 'sc-status ' + cls, 'data-status': s, text: s });
  }
  function table(headers, cls) {
    var t = el('table', { class: cls || '' });
    var hr = el('tr');
    headers.forEach(function (h) { hr.appendChild(el('th', { text: h, scope: 'col' })); });
    t.appendChild(el('thead', null, [hr]));
    t.appendChild(el('tbody'));
    return t;
  }
  function body(t) { return t.querySelector('tbody'); }
  function details(summary) {
    var d = el('details', { class: 'sc-fold' });
    d.appendChild(el('summary', { text: summary }));
    return d;
  }

  /* ---------------- the host's tab bar */
  function pickTabContainer() {
    return document.querySelector('nav.tabs') || document.querySelector('nav[role="tablist"]')
        || document.querySelector('[data-tabs]') || null;
  }
  function switchTab(name) {
    document.querySelectorAll('nav.tabs button[data-tab], nav[role="tablist"] button[data-tab]')
      .forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-tab') === name); });
    document.querySelectorAll('section.tab-pane')
      .forEach(function (p) { p.classList.toggle('active', p.getAttribute('data-tab') === name); });
  }
  function mount() {
    var pane = el('section', { class: 'layer-showcase tab-pane', 'data-layer': 'showcase',
                               'data-tab': 'showcase', id: 'tab-showcase', role: 'tabpanel',
                               'aria-label': 'Showcase' });
    var panes = document.querySelectorAll('section.tab-pane');
    var sib = panes.length ? panes[panes.length - 1] : null;
    if (sib && sib.parentNode) sib.parentNode.insertBefore(pane, sib.nextSibling);
    else (document.querySelector('main') || document.body).appendChild(pane);
    var tabs = pickTabContainer();
    if (tabs) {
      var b = el('button', { 'data-tab': 'showcase', 'data-layer': 'showcase', role: 'tab',
                             type: 'button', text: 'Showcase' });
      b.onclick = function () { switchTab('showcase'); };
      tabs.appendChild(b);
    }
    return pane;
  }

  /* ---------------- 1. story */
  function storySurface(ST, SC) {
    var s = surface('story', 'The story the page tells');
    s.appendChild(el('p', { class: 'sc-headline', 'data-story': 'headline', text: val(ST.headline) }));
    s.appendChild(el('p', { class: 'sc-lead', 'data-story': 'lead', text: val(ST.lead) }));
    var chips = el('p', { class: 'sc-chips' }, [chip('motif', ST.motif, 'sc-motif'),
      chip('polarity', ST.polarity), chip('locale', ST.locale)]);
    s.appendChild(chips);
    var secs = isArr(ST.sections) ? ST.sections : [];
    var ol = el('ol', { class: 'sc-outline' });
    secs.forEach(function (sec) {
      var parts = isObj(sec.items) ? Object.keys(sec.items).map(function (k) {
        return k + ' ' + val(sec.items[k]);
      }) : [];
      ol.appendChild(el('li', { 'data-section': String(sec.id) }, [el('strong', { text: String(sec.id) }),
        document.createTextNode(parts.length ? ' — ' + parts.join(' · ') : '')]));
    });
    s.appendChild(el('h4', { text: 'Sections' }));
    s.appendChild(ol);
    rolesBlock(s, ST.class_roles);
    groupsBlock(s, ST.input_families);
    if (isArr(ST.unresolved) && ST.unresolved.length)
      s.appendChild(para('Placeholders the summary could not fill here (the page fills them from '
        + 'its own strings): ' + ST.unresolved.join(', ') + '.', 'sc-note'));
    s.appendChild(para('Showcase of ' + val((SC.bundle || {}).slug) + ' run ' + val((SC.bundle || {}).run)
      + ', template ' + val((SC.template || {}).version) + ', built ' + val(SC.created_at) + '.', 'sc-note'));
    return s;
  }
  function rolesBlock(s, CR) {
    var roles = isObj(CR) && isObj(CR.roles) ? CR.roles : {};
    var src = isObj(CR) && isObj(CR.source) ? CR.source : {};
    var names = Object.keys(roles);
    s.appendChild(el('h4', { text: 'Class roles' }));
    if (!names.length) { s.appendChild(para('This task has no classes, so no class carries a role.', 'sc-note')); return; }
    var ul = el('ul', { class: 'sc-roles' });
    names.forEach(function (c) {
      var r = String(roles[c]);
      ul.appendChild(el('li', { 'data-class': c, 'data-role': r }, [
        el('span', { class: 'sc-swatch sc-role-' + r.replace(/[^a-z]/g, ''), 'aria-hidden': 'true' }),
        el('span', { class: 'sc-class', text: c }), document.createTextNode(' — ' + r),
        el('span', { class: 'sc-basis', text: src[c] ? ' (' + src[c] + ')' : '' })]));
    });
    s.appendChild(ul);
  }
  function groupsBlock(s, IG) {
    var groups = isArr(IG) ? IG : [];
    s.appendChild(el('h4', { text: 'Input groups' }));
    if (!groups.length) { s.appendChild(para('The story names no input groups.', 'sc-note')); return; }
    var t = table(['group', 'label', 'members'], 'sc-groups');
    groups.forEach(function (g) {
      body(t).appendChild(el('tr', { 'data-group': String(g.id) }, [el('td', { class: 'sc-code', text: String(g.id) }),
        el('td', { text: val(g.label) }), el('td', { class: 'sc-num', text: val(g.members_n) })]));
    });
    s.appendChild(t);
  }

  /* ---------------- 2. bindings: sortable by any column, filtered by any text */
  function bindingsSurface(B) {
    var s = surface('bindings', 'Every number on the page, traced');
    s.appendChild(para('Each value the page shows is a fact read from the bundle’s pinned '
      + 'records; nothing below was computed for the page.', 'sc-note'));
    var input = el('input', { type: 'search', class: 'sc-filter', 'aria-label': 'filter the bindings',
                              placeholder: 'filter by key, value, record or use' });
    s.appendChild(el('p', null, [input]));
    var t = table(['key', 'value', 'record', 'used in'], 'sc-bindings');
    B.forEach(function (b) {
      var uses = isArr(b.used_in) ? b.used_in : [];
      body(t).appendChild(el('tr', { 'data-key': String(b.key) }, [
        el('td', { class: 'sc-code', text: String(b.key) }),
        el('td', { class: 'sc-num', text: val(b.value) }),
        el('td', { class: 'sc-code', text: val(b.ref) }),
        el('td', { class: 'sc-code', text: uses.length ? uses.join(', ') : 'not cited in the story text' })]));
    });
    sortable(t);
    input.oninput = function () {
      var q = String(input.value || '').toLowerCase();
      Array.prototype.forEach.call(body(t).querySelectorAll('tr'), function (tr) {
        tr.hidden = !!q && (tr.textContent || '').toLowerCase().indexOf(q) < 0;
      });
    };
    s.appendChild(t);
    s.appendChild(para(B.length + ' binding' + (B.length === 1 ? '' : 's') + '.', 'sc-note'));
    return s;
  }
  function sortable(t) {
    var ths = t.querySelectorAll('thead th');
    Array.prototype.forEach.call(ths, function (th, i) {
      th.setAttribute('aria-sort', 'none');
      th.setAttribute('tabindex', '0');
      function sort() {
        var dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
        Array.prototype.forEach.call(ths, function (o) { o.setAttribute('aria-sort', 'none'); });
        th.setAttribute('aria-sort', dir);
        var rs = Array.prototype.slice.call(body(t).querySelectorAll('tr'));
        rs.sort(function (x, y) {
          var a = (x.children[i] || {}).textContent || '', b = (y.children[i] || {}).textContent || '';
          return dir === 'ascending' ? a.localeCompare(b) : b.localeCompare(a);
        });
        rs.forEach(function (r) { body(t).appendChild(r); });
      }
      th.onclick = sort;
      th.onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); } };
    });
  }

  /* ---------------- 3. report */
  function reportSurface(R) {
    var s = surface('report', 'The report the code notes link to');
    var status = String(R.status || 'not recorded');
    var head = el('p', { class: 'sc-report-status', 'data-report-status': status }, [statusWord(status)]);
    s.appendChild(head);
    if (status === 'brief-links') {
      s.appendChild(para('No report was written: each code note links to the stage brief that '
        + 'documents its step. Run the report stage, then this stage again, to link the report.', 'sc-warn'));
      var briefs = isObj(R.briefs) ? R.briefs : {};
      var ul = el('ul', { class: 'sc-briefs' });
      Object.keys(briefs).forEach(function (k) {
        ul.appendChild(el('li', { 'data-brief': k }, [el('span', { class: 'sc-code', text: k + ': ' + val(briefs[k]) })]));
      });
      if (ul.children.length) s.appendChild(ul);
    } else {
      s.appendChild(para('The code notes link to the compiled report through '
        + (R.origin === 'proposed' ? 'roles proposed from its labels and confirmed' : 'the report’s own role map') + '.'));
    }
    [['bound', 'Roles bound'], ['missing', 'Roles missing']].forEach(function (pair) {
      var list = isArr(R[pair[0]]) ? R[pair[0]] : [];
      if (!list.length && pair[0] === 'bound' && status === 'brief-links') return;
      s.appendChild(para(pair[1] + ': ' + (list.length ? list.join(', ') : 'none') + '.',
        pair[0] === 'missing' && list.length ? 'sc-warn' : 'sc-prose'));
    });
    var dg = details('digests');
    ['pdf_digest', 'aux_digest', 'roles_digest'].forEach(function (k) {
      if (R[k]) dg.appendChild(el('p', { class: 'sc-code', text: k + ' ' + R[k] }));
    });
    if (dg.children.length > 1) s.appendChild(dg);
    return s;
  }

  /* ---------------- 4. open it */
  function openSurface(U) {
    var s = surface('open it', 'Open the page');
    s.appendChild(para('The page runs on the bundle’s own service and nowhere else: start the '
      + 'service from the bundle directory, then open the address.'));
    [['serve', 'serve'], ['url', 'address'], ['docker', 'container']].forEach(function (pair) {
      if (!U[pair[0]]) return;
      var pre = el('pre', { class: 'sc-snippet', 'data-usage': pair[0], text: String(U[pair[0]]) });
      var btn = el('button', { type: 'button', class: 'sc-copy', text: 'copy ' + pair[1] });
      btn.onclick = function () {
        try { navigator.clipboard.writeText(String(U[pair[0]])); btn.textContent = 'copied'; }
        catch (e) { btn.textContent = 'select the text to copy'; }
      };
      s.appendChild(el('div', { class: 'sc-use' }, [pre, btn]));
    });
    if (U.note) s.appendChild(para(String(U.note), 'sc-note'));
    return s;
  }

  /* ---------------- 5. checks */
  function checksSurface(C) {
    var s = surface('checks', 'What was checked');
    var b = isObj(C.build) ? C.build : {}, v = isObj(C.served) ? C.served : {};
    s.appendChild(el('p', { 'data-check': 'build' }, [document.createTextNode('Build check: '), statusWord(b.status),
      document.createTextNode(b.checked_at ? ' at ' + b.checked_at : '')]));
    var sv = el('div', { 'data-check': 'served' });
    sv.appendChild(el('p', null, [document.createTextNode('Served walk: '), statusWord(v.status),
      document.createTextNode(v.checked_at ? ' at ' + v.checked_at : '')]));
    if (v.status === 'not-run')
      sv.appendChild(para('The walk did not run: ' + val(v.why_not_run), 'sc-warn sc-why'));
    else {
      sv.appendChild(para('Views walked: ' + (isArr(v.views) && v.views.length ? v.views.join(', ') : 'none')
        + '. Page errors: ' + val(v.page_errors) + '. Unresolved placeholders: '
        + val(v.placeholders_unresolved) + '.'));
      if (isArr(v.links) && v.links.length) sv.appendChild(para('Links answered at their pinned digests: '
        + v.links.length + '.'));
    }
    var shots = isArr(v.screenshots) ? v.screenshots : [];
    if (shots.length) {
      var ul = el('ul', { class: 'sc-shots' });
      shots.forEach(function (p) { ul.appendChild(el('li', { class: 'sc-code', text: String(p) })); });
      sv.appendChild(ul);
    }
    s.appendChild(sv);
    return s;
  }

  /* ---------------- 6. markers and readings */
  function markersSurface(M, RD) {
    var s = surface('markers and readings', 'Markers and readings');
    s.appendChild(el('p', { class: 'sc-abstract', 'data-reading': 'abstract', text: val(RD.abstract) }));
    var ul = el('ul', { class: 'sc-markers' });
    M.forEach(function (m) {
      var t = String(m);
      ul.appendChild(el('li', { class: t.indexOf('WARN') === 0 ? 'sc-m-warn' : t.indexOf('ERROR') === 0 ? 'sc-m-error' : '',
                                text: t }));
    });
    s.appendChild(M.length ? ul : para('No markers were recorded.', 'sc-note'));
    return s;
  }

  function empty(pane, why) {
    pane.appendChild(para(why, 'sc-empty'));
    window.__showcase = { version: VERSION, story: function () { return null; },
                          bindings: function () { return []; } };
  }

  ready(function () {
    if (document.querySelector('[data-layer="showcase"]')) return;
    var s = document.getElementById('layer-showcase-data');
    var pane = mount();
    pane.appendChild(el('header', { class: 'sc-head' }, [el('h2', { text: 'Showcase' })]));
    var D = null;
    try { D = s ? JSON.parse(s.textContent) : null; } catch (e) { D = null; }
    if (!isObj(D)) {
      empty(pane, s ? 'The showcase layer data did not parse. The showcase directory beside the '
        + 'bundle, and its SHOWCASE.json, are still the record.'
        : 'This file carries a showcase layer marker but no data script.');
      return;
    }
    var ST = isObj(D.story) ? D.story : {}, SC = isObj(D.showcase) ? D.showcase : {};
    var B = isArr(D.bindings) ? D.bindings : [], R = isObj(D.report) ? D.report : {};
    var U = isObj(D.usage) ? D.usage : {}, C = isObj(D.checks) ? D.checks : {};
    var M = isArr(D.markers) ? D.markers : [], RD = isObj(D.readings) ? D.readings : {};
    var parts = [storySurface(ST, SC), bindingsSurface(B), reportSurface(R), openSurface(U),
                 checksSurface(C), markersSurface(M, RD)];
    parts.forEach(function (p, i) { p.setAttribute('data-order', String(i + 1)); pane.appendChild(p); });
    window.__showcase = {
      version: VERSION,
      surfaces: function () { return SURFACES.slice(); },
      story: function () { return ST; },
      bindings: function () { return B; }
    };
  });
})();
