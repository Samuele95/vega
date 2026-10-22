// Number formatting. Every figure comes from the service or the story; this only writes it. The
// locale is the document's language (html[lang]); integers are grouped as that locale groups them.

const locale = () => (globalThis.document && document.documentElement && document.documentElement.lang) || "en-US";

export const fmtInt = (n) => Number(n).toLocaleString(locale());

export function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return fmtInt(v);
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : a >= 0.01 ? String(+v.toFixed(4)) : String(+v.toPrecision(2));
}

// A value with its interval, {value, ci: [lo, hi]}: the interval alone, as "[lo, hi]".
export const fmtCi = (v) => (v && v.ci) ? `[${fmtNum(v.ci[0])}, ${fmtNum(v.ci[1])}]` : "";

// A story value: an array is counted; {value, ci} is the value with its interval unless `short`.
export const fmtVal = (v, short = false) => Array.isArray(v) ? fmtInt(v.length)
  : (v && typeof v === "object" && "value" in v) ? fmtNum(v.value) + (v.ci && !short ? " " + fmtCi(v) : "") : fmtNum(v);

export const scalar = (v) => (v && typeof v === "object" && "value" in v) ? v.value : v;

// "+0.123" / "−0.123" with a true minus sign.
export const signed = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
export const sg = (v, d = 2) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(d);
export const plain = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

// Subscript digits, for x₁ … xₙ.
export const subs = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);

export const pct = (v, d = 1) => `${(100 * v).toFixed(d)} %`;

export const pad2 = (n) => String(n).padStart(2, "0");

// Counts a node's number up to `target` in 900 ms (ease-out cubic); at once under reduced motion.
export function countUp(node, target, reduce = false) {
  if (typeof target !== "number" || reduce) { node.textContent = fmtNum(target); return; }
  const t0 = performance.now(), dur = 900, int = Number.isInteger(target);
  const tick = (t) => {
    const k = Math.max(0, Math.min(1, (t - t0) / dur)), e = 1 - Math.pow(1 - k, 3);
    node.textContent = int ? fmtInt(Math.round(target * e)) : fmtNum(target * e);
    if (k < 1) requestAnimationFrame(tick); else node.textContent = fmtNum(target);
  };
  requestAnimationFrame(tick);
}
