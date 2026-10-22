// Input families (D4 §15.3): the network's inputs gathered into the groups the story names, so the
// drawings show a handful of circles instead of every column. A story@2 lists each group's members
// (input_groups@2: every network input in exactly one group); the page checks that coverage again
// and, when it fails, draws one family of every input and says why in problems. A story@1 (a bundle
// packed before v2.3) matched the inputs with regular expressions: the first group whose pattern
// matches takes the input, and what no group takes forms one more family. Without groups, one
// family holds every input.
//
// A family: {id, label, text, members: [input names, network order], idx: [unit indices]}; idx are
// the columns of the first dense layer that reads the inputs (a numeric input is one unit, a nominal
// input its embedding's dims: graph.featureUnits()).
import { tText, data, storyPlain } from "../core/text.js";

const ALL_ID = "all", OTHER_ID = "other";

function family({ id, label, text, members, names, units }) {
  const order = names.filter((n) => members.includes(n));
  return { id, label: storyPlain(label), text: storyPlain(text), members: order, idx: order.flatMap((n) => units[names.indexOf(n)]) };
}

const allOf = (names, units) => family({ id: ALL_ID, label: tText("net.family.all"), text: "", members: names, names, units });

// story@2: explicit members. Each input in exactly one group, each member an input.
function explicit(groups, names, units) {
  const problems = [], seen = new Map();
  for (const g of groups) for (const m of g.members) {
    if (!names.includes(m)) problems.push(tText("net.family.problem.unknown", { group: data(g.id), name: data(m) }));
    else if (seen.has(m)) problems.push(tText("net.family.problem.twice", { name: data(m), first: data(seen.get(m)), second: data(g.id) }));
    else seen.set(m, g.id);
  }
  names.filter((n) => !seen.has(n)).forEach((n) => problems.push(tText("net.family.problem.none", { name: data(n) })));
  if (problems.length) return { families: [allOf(names, units)], problems };
  const families = groups.map((g) => family({ id: g.id, label: g.label, text: g.text, members: g.members, names, units }));
  return { families: families.filter((f) => f.members.length), problems };
}

// story@1: regular expressions, the first match wins, the rest is one more family.
function matched(groups, names, units) {
  const res = groups.map((g) => (g.match || []).map((m) => new RegExp(m)));
  const taken = groups.map(() => []), other = [];
  names.forEach((n) => { const k = res.findIndex((rs) => rs.some((r) => r.test(n))); (k >= 0 ? taken[k] : other).push(n); });
  const families = groups.map((g, k) => family({ id: g.id || `g${k}`, label: g.label, text: g.text, members: taken[k], names, units }))
    .filter((f) => f.members.length);
  if (other.length) families.push(family({ id: OTHER_ID, label: tText("net.family.other"), text: "", members: other, names, units }));
  return { families: families.length ? families : [allOf(names, units)], problems: [] };
}

// {families, problems} of the inputs `names` (network order), from the story's input_groups;
// units[i]: the unit indices of input i (default: input i is unit i).
export function familiesOf({ groups = null, names, units = null }) {
  const u = units || names.map((_, i) => [i]);
  if (!Array.isArray(groups) || !groups.length) return { families: [allOf(names, u)], problems: [] };
  return groups.every((g) => Array.isArray(g.members)) ? explicit(groups, names, u) : matched(groups, names, u);
}

// The family that holds an input, or null.
export const familyOf = (families, name) => families.find((f) => f.members.includes(name)) || null;
