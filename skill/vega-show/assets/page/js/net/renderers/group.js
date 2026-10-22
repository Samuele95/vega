// A composite stage (D4 §7.2): its inside laid out as segments, drawn by their own renderers.
//   residual   each repeat's inside, then its ⊕; a skip arc carries the repeat's input over it into
//              the ⊕ (a projection, when there is one, rides on the arc). When the canvas is too
//              narrow for every repeat (D4 §7.3's overflow rule: ctx.collapse) the first alone is
//              drawn, titled "× n". One answer sets each arc's width by the skip's share of what the
//              ⊕ adds, ‖x‖ / (‖x‖ + ‖f(x)‖).
//   parallel   the branches stacked one above the other, all reading the same input, joined at the
//              right by ⊕ (sum) or ‖ (concat). One answer draws each branch's line into the merge as
//              wide as the branch's share of the merged output, and writes the share.
//   sequential (only when a caller did not lay it out inline) the inside, left to right.
// Its width is its children's — the inside and the merge node, which the service lists among them —
// plus GROUP_PAD (D4 §7.3). A focus on the group lights all of it; the pointer reaches the stage
// inside it.
import { css, el } from "../../core/dom.js";
import { t, tText, tn } from "../../core/text.js";
import { drawConnector, spreadRow, GROUP_PAD } from "../canvas.js";
import { groupParts, groupChild, lastStageOf } from "../segments.js";
import { stageNorm } from "../relevance.js";
import { tensorValues } from "../graph.js";

const PAD = GROUP_PAD / 2, MERGE = 18, GAP = 18, TITLE = 16, INSET = 10;
const MERGE_GLYPH = Object.freeze({ sum: "⊕", concat: "‖" });

// ---- what one answer says of a group
const extraOf = (ex, id) => (ex && ex.nodes && id && ex.nodes[id] && ex.nodes[id].extra) || {};
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const ratio = (a, b) => (finite(a) && finite(b) && a + b > 0 ? a / (a + b) : null);

// The skip's share of what a residual repeat's ⊕ adds, ‖x‖ / (‖x‖ + ‖f(x)‖) (D4 §7.2): the add node's
// own reading (skip_norm, branch_norm; inspect-contract §3.1), else the norms of the repeat's input
// (its projection's output when it has one) and of its inside's output. null when the explanation
// says neither.
export function skipShare(ex, seg, r) {
  const rep = seg.repeats[r], extra = extraOf(ex, rep.add);
  if (finite(extra.skip_norm) && finite(extra.branch_norm)) return ratio(extra.skip_norm, extra.branch_norm);
  const x = rep.projection ? stageNorm(ex, rep.projection, "out", tensorValues)
    : r === 0 ? stageNorm(ex, seg.group, "in", tensorValues) : stageNorm(ex, seg.repeats[r - 1].add, "out", tensorValues);
  const inner = lastStageOf(rep.segments);
  return ratio(x, inner ? stageNorm(ex, inner, "out", tensorValues) : null);
}

// Each branch's share of a parallel group's merged output (D4 §7.2): the merge node's reading
// (branch_norms), else the norms of each branch's output. null when the explanation says neither.
export function branchShares(ex, seg) {
  const own = extraOf(ex, seg.node).branch_norms;
  const norms = Array.isArray(own) && own.length === seg.branches.length && own.every(finite) ? own
    : seg.branches.map((b) => { const id = lastStageOf(b.segments); return id ? stageNorm(ex, id, "out", tensorValues) : null; });
  if (!norms.every(finite)) return null;
  const total = norms.reduce((a, v) => a + v, 0);
  return total > 0 ? norms.map((v) => v / total) : null;
}

// ---- the inside, as rows of items
// The repeats of a residual group drawn: every one, or the first alone when the canvas collapses
// repeated groups.
export const shownRepeats = (seg, ctx) => (ctx.collapse && seg.repeats.length > 1 ? [0] : seg.repeats.map((_, r) => r));

// The rows of the inside, each a list of items {seg, at: [part, index]} or {merge: repeat}: one row
// per branch (parallel); one row (residual) of each shown repeat's segments followed by its ⊕.
export function rowsOf(seg, ctx) {
  const items = (part) => groupParts(seg)[part].map((s, i) => ({ seg: s, at: [part, i] }));
  if (seg.mode === "residual") return [shownRepeats(seg, ctx).flatMap((r) => [...items(r), { merge: r }])];
  return groupParts(seg).map((_, k) => items(k));
}

const isMerge = (it) => it.merge !== undefined;
const sizeOf = (it, ctx) => (isMerge(it) ? { w: MERGE } : ctx.rendererFor(it.seg).measure(it.seg, ctx));
const rowWidth = (row, ctx) => row.reduce((a, it) => a + sizeOf(it, ctx).w, 0) + GAP * Math.max(0, row.length - 1);
// the acts of the child at [part, index], from the group's
const childActs = (acts, at) => (acts && acts.parts[at[0]] ? acts.parts[at[0]][at[1]] : null);

// One row of the inside: its items spread over the band, each segment laid out by its renderer.
function layoutRow(row, band, ctx, acts) {
  const boxes = spreadRow(row.map((it) => sizeOf(it, ctx)), band.x, band.w, GAP), y = band.y + band.h / 2;
  return row.map((it, i) => {
    const b = boxes[i];
    if (isMerge(it)) return { merge: it.merge, x: b.x + b.w / 2, y };
    const r = ctx.rendererFor(it.seg), box = { x: b.x, y: band.y, w: b.w, h: band.h };
    return { seg: it.seg, at: it.at, r, L: r.layout(it.seg, box, { ...ctx, acts: childActs(acts, it.at) }) };
  });
}

// ---- drawing
// The group's own share of the drawing's phase: its inside grows while the group is reached.
const localPhase = (st) => Math.max(0, Math.min(1, st.phase * st.n - st.i));
// The group's state in one answer (its acts()), or null.
const groupActs = (st) => (st.acts && st.acts[st.i] ? st.acts[st.i] : null);

function title(seg, ctx) {
  const collapsed = seg.mode === "residual" && shownRepeats(seg, ctx).length < seg.repeats.length;
  return tText(seg.mode === "parallel" ? "group.parallel" : "group.residual") + (collapsed ? " " + tText("group.repeat", { n: seg.repeats.length }) : "");
}

function drawFrame(g, seg, L, st) {
  const C = st.ctx.colors, on = st.focus.node(seg.group);
  g.globalAlpha = on ? .5 : .2; g.strokeStyle = C.line; g.setLineDash([4, 4]); g.lineWidth = 1;
  g.strokeRect(L.box.x + .5, L.box.y + .5, L.box.w - 1, L.box.h - 1);
  L.bandYs.slice(1).forEach((y0) => {                   // a faint line between two branches
    const y = y0 - GAP / 2;
    g.beginPath(); g.moveTo(L.box.x + PAD, y); g.lineTo(L.merge.x - MERGE, y); g.stroke();
  });
  g.setLineDash([]);
  g.globalAlpha = .85; g.fillStyle = C.muted; g.font = `10.5px ${css("--mono")}`; g.textAlign = "left"; g.textBaseline = "top";
  g.fillText(title(seg, st.ctx), L.box.x + 6, L.box.y + 4);
}

// The skip arcs of a residual group: from each shown repeat's input (the group's, or the ⊕ before it)
// over its inside into its ⊕, a projection riding on the arc; once answered, as wide as the skip's share.
function drawArcs(g, seg, L, st) {
  const C = st.ctx.colors, on = st.focus.node(seg.group), top = L.box.y - 10, acts = groupActs(st), phase = localPhase(st);
  let from = L.box.x + 2;
  L.bands[0].filter(isMerge).forEach((m) => {
    const share = acts && acts.share ? acts.share[m.merge] : null, span = m.x - from;
    g.globalAlpha = (on ? .7 : .3) * phase; g.strokeStyle = C.fg2; g.lineWidth = finite(share) ? .6 + 4.4 * share : 1.2;
    g.beginPath(); g.moveTo(from, m.y); g.bezierCurveTo(from + span / 3, top, m.x - span / 3, top, m.x, m.y - 8); g.stroke();
    if (seg.repeats[m.merge].projection) {                // at the arc's apex (t = ½)
      const x = from + span / 2, y = (2 * m.y - 8) / 8 + .75 * top;
      g.lineWidth = 1; g.fillStyle = C.bg; g.fillRect(x - 5, y - 5, 10, 10); g.strokeRect(x - 5, y - 5, 10, 10);
    }
    from = m.x + MERGE / 2;
  });
}

// A branch into the merge: the connector, or once answered one line as wide as the branch's share,
// with the share written above it.
function drawIntoMerge(g, band, L, st, share) {
  const last = band[band.length - 1], target = { in: [{ x: L.merge.x - MERGE / 2, y: L.merge.y }] };
  if (!last) return;
  if (!finite(share)) { drawConnector(g, last.L, target, st); return; }
  const C = st.ctx.colors, outs = last.L.out || [], x1 = L.merge.x - MERGE / 2;
  const x0 = outs.length ? Math.max(...outs.map((p) => p.x)) : last.L.box.x + last.L.box.w;
  const y0 = outs.length ? outs.reduce((a, p) => a + p.y, 0) / outs.length : last.L.box.y + last.L.box.h / 2;
  g.globalAlpha = .75 * st.phase; g.strokeStyle = C.acc; g.lineWidth = .6 + 4.4 * share;
  g.beginPath(); g.moveTo(x0, y0); g.bezierCurveTo((x0 + x1) / 2, y0, (x0 + x1) / 2, L.merge.y, x1, L.merge.y); g.stroke();
  g.globalAlpha = .85 * st.phase; g.fillStyle = C.fg2; g.font = `10px ${css("--mono")}`; g.textAlign = "right"; g.textBaseline = "bottom";
  g.fillText(tText("group.share", { p: share }), x1 - 2, y0 - 3);
}

export default {
  kinds: ["residual", "sequential", "parallel"], detail: false,

  measure(seg, ctx) {
    const par = seg.mode === "parallel", inside = Math.max(...rowsOf(seg, ctx).map((r) => rowWidth(r, ctx)), ctx.boxW);
    return { w: inside + (par ? GAP + MERGE : 0) + GROUP_PAD, minH: 80 * (par ? seg.branches.length : 1), grow: true };
  },

  layout(seg, box, ctx) {
    const rows = rowsOf(seg, ctx), n = rows.length, par = seg.mode === "parallel", mid = box.y + box.h / 2;
    const inner = { x: box.x + PAD, w: box.w - GROUP_PAD - (par ? GAP + MERGE : 0) };
    const bandH = (box.h - TITLE - GAP * (n - 1)) / n, inset = n > 1 ? INSET : 0;
    const bandYs = rows.map((_, k) => box.y + TITLE + k * (bandH + GAP));
    const bands = rows.map((row, k) => layoutRow(row, { ...inner, y: bandYs[k] + inset, h: bandH - 2 * inset }, ctx, ctx.acts || null));
    const flat = bands.flat(), merges = flat.filter(isMerge), kids = flat.filter((c) => !isMerge(c));
    const merge = par ? { x: box.x + box.w - PAD - MERGE / 2, y: mid } : merges.length ? merges[merges.length - 1] : null;
    const first = bands[0] && bands[0][0] && !isMerge(bands[0][0]) ? bands[0][0] : null, inPort = { x: box.x, y: merge ? merge.y : mid };
    const last = kids[kids.length - 1];
    return {
      box, bands, bandYs, merge: merge || { x: box.x + box.w - PAD, y: mid }, inPort,
      out: merge ? [{ x: merge.x + MERGE / 2, y: merge.y }] : last ? last.L.out : [{ x: box.x + box.w, y: mid }],
      in: par || !first ? [inPort] : first.L.in, inMatrix: par || !first ? null : first.L.inMatrix,
      captions: kids.flatMap((c) => c.L.captions || []),
    };
  },

  draw(g, seg, L, st) {
    const C = st.ctx.colors, hov = st.hover && st.hover.i === st.i ? st.hover.hit : null, phase = localPhase(st);
    const acts = groupActs(st), par = seg.mode === "parallel";
    drawFrame(g, seg, L, st);
    L.bands.forEach((band, k) => {
      const bandActs = band.map((c) => (isMerge(c) ? null : childActs(acts, c.at)));
      const sub = (i) => ({ ...st, phase, acts: bandActs, i, n: band.length, hover: hov && hov.band === k && hov.item === i ? { i, hit: hov.hit } : null });
      const one = { ...sub(0), i: 0, n: 1 };
      if (par && band[0]) drawConnector(g, { out: [L.inPort] }, band[0].L, one);
      band.forEach((c, i) => {
        const next = band[i + 1];
        if (!next) return;
        const a = isMerge(c) ? { out: [{ x: c.x + MERGE / 2, y: c.y }] } : c.L;
        const b = isMerge(next) ? { in: [{ x: next.x - MERGE / 2, y: next.y }] } : next.L;
        drawConnector(g, a, b, sub(i));
      });
      band.forEach((c, i) => { if (!isMerge(c)) { g.save(); c.r.draw(g, c.seg, c.L, sub(i)); g.restore(); } });
      if (par) drawIntoMerge(g, band, L, one, acts && acts.share ? acts.share[k] : null);
    });
    if (seg.mode === "residual") drawArcs(g, seg, L, st);
    const glyph = par ? MERGE_GLYPH[seg.merge] || MERGE_GLYPH.concat : MERGE_GLYPH.sum;
    const merges = par ? [L.merge] : seg.mode === "residual" ? L.bands[0].filter(isMerge) : [];
    g.globalAlpha = phase; g.fillStyle = C.fg; g.font = `15px ${css("--sans")}`; g.textAlign = "center"; g.textBaseline = "middle";
    merges.forEach((m) => g.fillText(glyph, m.x, m.y));
  },

  hit(seg, L, x, y) {
    for (let k = 0; k < L.bands.length; k++) for (let i = 0; i < L.bands[k].length; i++) {
      const c = L.bands[k][i];
      if (isMerge(c)) continue;
      const b = c.L.box;
      if (x < b.x - 10 || x > b.x + b.w + 10 || y < b.y - 16 || y > b.y + b.h + 16) continue;
      const inner = c.r.hit(c.seg, c.L, x, y);
      if (inner) return { band: k, item: i, at: c.at, hit: inner };
    }
    return x >= L.box.x && x <= L.box.x + L.box.w && y >= L.box.y && y <= L.box.y + L.box.h ? { group: seg.group } : null;
  },

  caption(seg, hit, ctx) {
    if (hit && hit.at) {
      const c = groupChild(seg, hit.at);
      return ctx.rendererFor(c).caption(c, hit.hit, ctx);
    }
    const node = ctx.graph.byId(seg.group);
    return el("div", {}, el("span", { class: "who" }, t(seg.mode === "parallel" ? "group.parallel" : "group.residual")),
      node && node.params ? el("span", { class: "muted" }, " · ", tn("card.stage_params", node.params)) : null,
      node && node.formula ? el("div", { class: "formula" }, node.formula) : null);
  },

  // One answer: every child's state by [part, index] (all repeats, whether drawn or not), and the
  // shares the arcs (residual) or the lines into the merge (parallel) are drawn with.
  acts(seg, ex, graph, ctx) {
    const parts = groupParts(seg).map((segs) => segs.map((s) => ctx.rendererFor(s).acts(s, ex, graph, ctx)));
    const share = seg.mode === "residual" ? seg.repeats.map((_, r) => skipShare(ex, seg, r)) : seg.mode === "parallel" ? branchShares(ex, seg) : null;
    const said = parts.some((p) => p.some((a) => a !== null)) || (share || []).some((v) => v !== null);
    return said ? { parts, share } : null;
  },
};
