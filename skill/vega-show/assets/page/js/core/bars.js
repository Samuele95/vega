// Bars, the tooltip and the two-colour shading: the page's small charting vocabulary. Every label is
// the caller's; these only lay values out.
import { $, el } from "./dom.js";
import { CACHE } from "./state.js";

// One bar: name, track (value over max, optional interval bracket), label. `i` staggers the grow.
export function barRow({ name, v, max, label, selected = false, i = 0, ci = null, cls = "", fillCls = "" }) {
  const fill = el("div", { class: "fill " + fillCls, style: `width:${Math.max(0, v / max * 100)}%;animation-delay:${i * .045}s` });
  const track = el("div", { class: "track" }, fill);
  if (ci) track.append(el("div", { class: "ci", style: `left:${ci[0] / max * 100}%;width:${Math.max(.4, (ci[1] - ci[0]) / max * 100)}%` }));
  return el("div", { class: "bar" + (selected ? " sel" : "") + (cls ? " " + cls : "") }, el("div", { class: "name", title: name }, name), track, el("div", { class: "val" }, label));
}

// One signed term of a sum, on a diverging track: `val` is the list of nodes or strings shown.
export function termRow({ name, v, max, val, i = 0, cls = "" }) {
  return el("div", { class: "bar " + cls }, el("div", { class: "name", title: name }, name),
    el("div", { class: "track" }, el("div", { class: "fill " + (v >= 0 ? "pos" : "neg"), style: `width:${Math.min(50, Math.abs(v) / max * 50)}%;animation-delay:${i * .04}s` })),
    el("div", { class: "val" }, ...val));
}

// The k largest of `values` by magnitude, as diverging bars under a caption; tag(i) names entry i.
export function divergingBars({ title, values, k, tag }) {
  const top = values.map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, k);
  const mx = Math.max(1e-9, ...top.map((t) => Math.abs(t[1])));
  const box = el("div", {}, el("div", { class: "caps muted", style: "margin:8px 0 4px" }, title), el("div", { class: "bars diverge" }));
  top.forEach(([i, v], n) => box.lastChild.append(el("div", { class: "bar" },
    el("div", { class: "name", title: tag(i) }, tag(i)),
    el("div", { class: "track" }, el("div", { class: "fill " + (v >= 0 ? "pos" : "neg"), style: `width:${Math.abs(v) / mx * 50}%;animation-delay:${n * .04}s` })),
    el("div", { class: "val" }, (v >= 0 ? "+" : "−") + Math.abs(v).toPrecision(2)))));
  return box;
}

// The floating tooltip (#wtip) beside the pointer; nodes = null hides it.
export function tip(e, nodes) {
  const t = $("wtip");
  if (!nodes) { t.classList.remove("on"); return; }
  t.replaceChildren(...nodes); t.classList.add("on");
  const x = Math.min(innerWidth - 380, e.clientX + 14), y = Math.min(innerHeight - 70, e.clientY + 14);
  t.style.left = x + "px"; t.style.top = y + "px";
}

// [r, g, b] of a CSS colour, parsed once per colour.
export function rgbOf(color) {
  if (!CACHE.rgb.has(color)) {
    const c = document.createElement("canvas").getContext("2d");
    c.fillStyle = color;
    const h = c.fillStyle.replace("#", "");
    CACHE.rgb.set(color, [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16)));
  }
  return CACHE.rgb.get(color);
}

// A cell colour for v against the largest magnitude mx: palette.pos raises, palette.neg lowers,
// on a square-root scale so faint cells stay visible.
export function shade(v, mx, palette) {
  const [r, g, b] = rgbOf(v >= 0 ? palette.pos : palette.neg), a = Math.min(1, Math.sqrt(Math.abs(v) / Math.max(mx, 1e-12)));
  return `rgba(${r},${g},${b},${(0.04 + 0.92 * a).toFixed(3)})`;
}
