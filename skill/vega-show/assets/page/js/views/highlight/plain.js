// The code room's reader for a language it has no highlighter for (D4 §3.2): every line as it is,
// uncoloured (the room writes it as text, so nothing needs escaping), and a walk step's extent is
// the anchored line down to the next blank line, at most MAX_LINES lines. The members are
// python.js's: language, tokens(text), extent(lines, i).

export const MAX_LINES = 90;

export function tokens(text) {
  return String(text).split("\n").map((line) => (line ? [[null, line]] : []));
}

export function extent(lines, i) {
  let e = i + 1;
  while (e < lines.length && lines[e].trim()) e++;
  return [i, Math.min(e, i + MAX_LINES) - 1];
}

export default Object.freeze({ language: null, tokens, extent });
