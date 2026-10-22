// The code room's Python reader (D4 §3.2; the hand-built page's pyLine / codeBlock, 0d06aea
// index.html:2025–2070): enough colour to read by — comments, strings (a triple-quoted one may span
// lines), decorators, numbers, keywords, the names being defined and `self` — and the extent of a
// definition, so a walk step shows the whole function or class it is anchored in. The room loads
// one highlighter per page, by page.json#code.language (views/code.js); every highlighter has the
// same three members:
//   language          the language it reads
//   tokens(text)      one array per line of [class | null, text] runs, in order; the classes are
//                     page.css's tk-* (tk-c comment, tk-s string, tk-d decorator, tk-n number,
//                     tk-k keyword, tk-f defined name, tk-self), null is plain text; the runs of a
//                     line joined give the line back exactly
//   extent(lines, i)  [first, last] line indices of the definition whose header (or decorator) is
//                     lines[i]: its decorators, its header and every deeper-indented line under it,
//                     trailing blank lines left out, at most MAX_LINES lines

export const MAX_LINES = 90;

const KEYWORDS = Object.freeze(["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
  "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
  "raise", "return", "try", "while", "with", "yield"]);
const DEFINES = Object.freeze(["def", "class"]);
// comment | an opening triple quote | a one-line string | a decorator | a number | a name
const TOKEN = /(#.*$)|([rbfuRBFU]{0,2}(?:"""|'''))|([rbfuRBFU]{0,2}"(?:[^"\\]|\\.)*"|[rbfuRBFU]{0,2}'(?:[^'\\]|\\.)*')|(@[\w.]+)|\b(\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b/;

// The runs of one line; `st.open` carries an unclosed triple quote from one line to the next.
function lineTokens(line, st) {
  const out = [], plain = (s) => { if (s) out.push([null, s]); };
  let last = 0, prev = "";
  if (st.open) {
    const j = line.indexOf(st.open);
    if (j < 0) return line ? [["tk-s", line]] : [];
    out.push(["tk-s", line.slice(0, j + 3)]); last = j + 3; st.open = null;
  }
  const re = new RegExp(TOKEN.source, "g");
  re.lastIndex = last;
  for (let m = re.exec(line); m; m = re.exec(line)) {
    plain(line.slice(last, m.index)); last = re.lastIndex;
    if (m[2]) {
      const q = m[2].slice(-3), j = line.indexOf(q, re.lastIndex);
      if (j < 0) { out.push(["tk-s", line.slice(m.index)]); st.open = q; return out; }
      out.push(["tk-s", line.slice(m.index, j + 3)]); last = re.lastIndex = j + 3;
    } else if (m[6]) {
      const w = m[6];
      out.push([KEYWORDS.includes(w) ? "tk-k" : DEFINES.includes(prev) ? "tk-f" : w === "self" ? "tk-self" : null, w]);
      prev = w; continue;
    } else out.push([m[1] ? "tk-c" : m[3] ? "tk-s" : m[4] ? "tk-d" : "tk-n", m[0]]);
    prev = "";
  }
  plain(line.slice(last));
  return out;
}

export function tokens(text) {
  const st = { open: null };
  return String(text).split("\n").map((line) => lineTokens(line, st));
}

const indent = (s) => s.match(/^\s*/)[0].length;
const decorator = (s) => /^\s*@/.test(s);

export function extent(lines, i) {
  let a = i;
  while (a > 0 && decorator(lines[a - 1])) a--;
  let d = i;
  while (d < lines.length - 1 && decorator(lines[d])) d++;
  const base = indent(lines[d]);
  let e = d + 1;
  while (e < lines.length && (!lines[e].trim() || indent(lines[e]) > base)) e++;
  while (e > d + 1 && !lines[e - 1].trim()) e--;
  return [a, Math.min(e, a + MAX_LINES) - 1];
}

export default Object.freeze({ language: "python", tokens, extent });
