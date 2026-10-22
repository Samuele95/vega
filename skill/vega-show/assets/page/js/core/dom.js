// DOM helpers shared by every module: element lookup and creation, the browser store, the reduced
// motion preference and the CSS tokens.

export const $ = (id) => document.getElementById(id);

// el("div", { class: "bar", style: "…", "data-x": 1 }, child, "text", …): null, undefined and
// false children are skipped; anything that is not a node becomes a text node.
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "style") n.style.cssText = v; else n.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
}

// localStorage, when the browser allows it; a private window or a blocked store reads as empty.
export const store = Object.freeze({
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* the page works without it */ } },
});

export const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// The resolved value of a CSS custom property of the document, e.g. css("--accent").
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Replays the entrance animation of a node.
export const restart = (node) => { node.classList.remove("enter"); void node.offsetWidth; node.classList.add("enter"); };
