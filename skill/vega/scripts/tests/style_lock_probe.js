// style_lock_probe.js — the computed layer's reading (style_lock.py): __styleLock(spec) builds one
// specimen element per selector at the top of the body, reads its computed style in each theme,
// and restores the page. spec = {selectors, properties, themes, tokens}.
function __styleLock(spec) {
  const root = document.documentElement, before = root.getAttribute("data-theme");
  const freeze = document.createElement("style"), host = document.createElement("div");
  freeze.textContent = "*,*::before,*::after{transition:none!important}";
  document.head.appendChild(freeze);
  host.setAttribute("aria-hidden", "true");
  document.body.insertBefore(host, document.body.firstChild);
  const chOf = cs => {
    const s = document.createElement("span");
    s.textContent = "0".repeat(100);
    s.style.cssText = "position:absolute;white-space:nowrap;letter-spacing:normal;font-family:" + cs.fontFamily +
      ";font-size:" + cs.fontSize + ";font-weight:" + cs.fontWeight + ";font-style:" + cs.fontStyle;
    host.appendChild(s); const w = s.getBoundingClientRect().width / 100; s.remove(); return w;
  };
  const read = (cs, p) => {
    if (p === "padding" || p === "margin")
      return ["top", "right", "bottom", "left"].map(s => cs.getPropertyValue(p + "-" + s)).join(" ");
    const v = cs.getPropertyValue(p).trim();
    if (p !== "max-width" || !/^[\d.]+px$/.test(v) || Math.abs(parseFloat(v) - Math.round(parseFloat(v))) < 0.005) return v;
    const n = parseFloat(v) / chOf(cs);
    return Math.abs(n - Math.round(n)) < 0.02 ? Math.round(n) + "ch" : v;
  };
  const specimen = sel => {
    if (sel === "html" || sel === "body") return [sel === "html" ? root : document.body, null];
    let parent = host, top = null, el = null;
    for (const part of sel.replace(/\s*>\s*/g, " ").trim().split(/\s+/)) {
      const m = part.match(/^([a-z][a-z0-9]*)?(.*)$/i);
      el = document.createElement(m[1] || "div");
      for (const bit of m[2].match(/[.#][\w-]+|\[[^\]]+\]/g) || []) {
        if (bit[0] === ".") el.classList.add(bit.slice(1));
        else if (bit[0] === "#") el.id = bit.slice(1);
        else { const kv = bit.slice(1, -1).split("="); el.setAttribute(kv[0], (kv[1] || "").replace(/^["']|["']$/g, "")); }
      }
      parent.appendChild(el); top = top || el; parent = el;
    }
    return [el, top];
  };
  const out = { viewport: [innerWidth, innerHeight], themes: {},
                reduced_motion: matchMedia("(prefers-reduced-motion: reduce)").matches };
  try {
    for (const theme of spec.themes) {
      root.setAttribute("data-theme", theme);
      const styles = {}, tokens = {};
      for (const sel of spec.selectors) {
        const [el, top] = specimen(sel), cs = getComputedStyle(el), rec = {};
        for (const p of spec.properties) rec[p] = read(cs, p);
        styles[sel] = rec; if (top) top.remove();
      }
      const rcs = getComputedStyle(root);
      for (const t of spec.tokens) tokens[t] = rcs.getPropertyValue(t).trim();
      out.themes[theme] = { styles, tokens };
    }
  } finally {
    host.remove(); freeze.remove();
    if (before === null) root.removeAttribute("data-theme"); else root.setAttribute("data-theme", before);
  }
  return out;
}
