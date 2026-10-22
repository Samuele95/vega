// A small DOM for the node tests that render the stepper (tests/stepper.test.js and the driver of
// vega-show/scripts/tests/test_page_math.py): the part of the document the page's builders call —
// createElement / createElementNS / createTextNode / createDocumentFragment, append and
// replaceChildren, attributes, className and classList, style.cssText, hidden, textContent,
// getElementById and querySelector / querySelectorAll by #id, .class and tag — enough to build
// every step's body and read its text. Not a browser: no layout, no events, no styles computed.
// Not served, not copied into a showcase.

class Node_ {
  constructor(nodeType) { Object.assign(this, { nodeType, childNodes: [], parentNode: null }); }
  append(...kids) {
    for (const k of kids) {
      const n = typeof k === "string" ? new Text_(k) : k;
      if (n.nodeType === 11) { for (const c of n.childNodes.slice()) this.append(c); n.childNodes = []; continue; }
      if (n.parentNode) n.parentNode.childNodes = n.parentNode.childNodes.filter((c) => c !== n);
      n.parentNode = this;
      this.childNodes.push(n);
    }
  }
  replaceChildren(...kids) { for (const c of this.childNodes) c.parentNode = null; this.childNodes = []; this.append(...kids); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { this.replaceChildren(new Text_(String(v))); }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get childElementCount() { return this.children.length; }
  // every element below this one, depth first
  descendants() { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  querySelectorAll(sel) { return this.descendants().filter((e) => e.matches(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

class Text_ extends Node_ {
  constructor(text) { super(3); this.data = text; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
  descendants() { return []; }
}

class Fragment_ extends Node_ { constructor() { super(11); } }

class ClassList_ {
  constructor(el) { this.el = el; }
  get list() { return this.el.className.split(/\s+/).filter(Boolean); }
  contains(c) { return this.list.includes(c); }
  add(...cs) { this.el.className = [...new Set([...this.list, ...cs])].join(" "); }
  remove(...cs) { this.el.className = this.list.filter((c) => !cs.includes(c)).join(" "); }
  toggle(c, on = !this.contains(c)) { if (on) this.add(c); else this.remove(c); return on; }
}

class Element_ extends Node_ {
  constructor(tag, ns = null) {
    super(1);
    Object.assign(this, { tagName: tag.toUpperCase(), localName: tag, namespaceURI: ns, attributes: {}, className: "", hidden: false,
      style: { cssText: "" }, onclick: null });
    this.classList = new ClassList_(this);
  }
  setAttribute(k, v) { if (k === "class") this.className = String(v); else if (k === "id") this.id = String(v); else this.attributes[k] = String(v); }
  getAttribute(k) { return k === "class" ? this.className : k === "id" ? this.id ?? null : this.attributes[k] ?? null; }
  get dataset() { return Object.fromEntries(Object.entries(this.attributes).filter(([k]) => k.startsWith("data-")).map(([k, v]) => [k.slice(5), v])); }
  closest(sel) { let n = this; while (n && n.nodeType === 1) { if (n.matches(sel)) return n; n = n.parentNode; } return null; }
  click() { if (this.onclick) this.onclick({ target: this }); }
  // one simple selector: #id, .class (several), tag, tag.class
  matches(sel) {
    const m = /^([a-z][a-z0-9-]*)?((?:[#.][\w-]+)*)$/i.exec(sel.trim());
    if (!m) throw new Error(`dom-shim: selector ${sel} is not supported`);
    if (m[1] && m[1].toLowerCase() !== this.localName.toLowerCase()) return false;
    return (m[2].match(/[#.][\w-]+/g) || []).every((p) => (p[0] === "#" ? this.id === p.slice(1) : this.classList.contains(p.slice(1))));
  }
}

class Document_ extends Node_ {
  constructor() {
    super(9);
    this.documentElement = new Element_("html");
    this.documentElement.lang = "en-US";
    this.body = new Element_("body");
    this.documentElement.append(this.body);
    this.append(this.documentElement);
  }
  createElement(tag) { return new Element_(tag); }
  createElementNS(ns, tag) { return new Element_(tag, ns); }
  createTextNode(text) { return new Text_(String(text)); }
  createDocumentFragment() { return new Fragment_(); }
  getElementById(id) { return this.querySelector(`#${id}`); }
}

// Installs a fresh document (and window, as the page's globals) on globalThis; returns it.
export function install() {
  const doc = new Document_();
  globalThis.document = doc;
  if (!globalThis.window) globalThis.window = globalThis;
  return doc;
}
