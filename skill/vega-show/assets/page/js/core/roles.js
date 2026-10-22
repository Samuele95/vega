// Class roles (vega/class-roles@1, story.json#class_roles): the one place the page reads what a
// class means, so it colours from roles and never from class names.
//   neutral  the class the reader can take as the ordinary case (drawn in the accent)
//   alert    a class a person asked to see as an alert, or the anomalous side of an anomaly task
//            (drawn in the alert colour)
//   plain    everything else, and the default (drawn in the accent)
// The rules the page holds to, whatever the block says (vega-show's plan_rules refuses such a block
// before it ships; the page still never paints one):
//   - every role but plain has its basis (source[name]), one of the catalog's bases, and the basis
//     grants the role: `alert` only user:<turn>, show:<stance date>, registry:<lane> (a matrix
//     lane's registry, never a user's turn) or, on an anomaly lane,
//     task:anomaly; `neutral` every basis (vega:default and the structural ones grant nothing
//     else). A role its basis does not grant is drawn plain, and said in problems();
//   - no roles at all for the value, reconstruction and horizon view keys;
//   - at most one neutral for the classes, binary and anomaly view keys (the first, in the
//     network's output order, keeps it);
//   - only the lane's own names: its output classes (labels), or for anomaly the two sides, named
//     by the task's classes when a label exists and normal / anomalous otherwise.
// A problem is a machine code and the values of its placeholders, {code, params}: the code is a
// key of the strings table (roles.problem.*), whose English the page renders with
// tText(code, params). The closed sets mirror vega-show/scripts/catalog.json#roles
// (tests/roles.test.js holds them equal).

export const ROLES_SCHEMA = "vega/class-roles@1";
export const ROLE_SET = Object.freeze(["neutral", "alert", "plain"]);
export const DEFAULT_ROLE = "plain";
export const ALERT_REQUIRES = Object.freeze(["user:", "show:", "registry:", "task:anomaly"]);
export const BASIS_RE = /^(structural:(majority|negative)|task:anomaly|user:.+|show:\d{4}-\d{2}-\d{2}|registry:[A-Za-z0-9_-]+|vega:default)$/;
// What each basis grants (catalog roles.basis[*].grants), by the basis' kind.
export const GRANTS = Object.freeze({
  "structural:majority": Object.freeze(["neutral"]),
  "structural:negative": Object.freeze(["neutral"]),
  "task:anomaly": Object.freeze(["neutral", "alert"]),
  "user:": Object.freeze(["neutral", "alert", "plain"]),
  "show:": Object.freeze(["neutral", "alert", "plain"]),
  "registry:": Object.freeze(["neutral", "alert", "plain"]),
  "vega:default": Object.freeze(["neutral", "plain"]),
});
export const NO_ROLES = Object.freeze(["value", "reconstruction", "horizon"]);
export const ONE_NEUTRAL = Object.freeze(["classes", "binary", "anomaly"]);
export const ANOMALY_SIDES = Object.freeze(["normal", "anomalous"]);
const CLS = Object.freeze({ neutral: "role-neutral", alert: "role-alert", plain: "role-plain" });

// A problem: its code (a strings key) and its placeholders' values, as text (a value the block
// brings in its JSON form, so a string reads quoted and a missing one undefined).
const problem = (code, params) => Object.freeze({ code, params: Object.freeze(params) });
const shown = (value) => String(JSON.stringify(value));

// The lane's view key, from a Lane (viewKey) or page.json#lane (view_key).
export const viewKeyOf = (lane) => (lane && (lane.viewKey || lane.view_key)) || null;

// The roles a basis grants ([] for a basis outside the catalog's).
export function grantsOf(basis) {
  if (typeof basis !== "string" || !BASIS_RE.test(basis)) return [];
  const kind = ["user:", "show:", "registry:"].find((p) => basis.startsWith(p)) || basis;
  return GRANTS[kind] || [];
}

// The names the roles may cover on this lane.
export function roleNames(lane, network, block) {
  const vk = viewKeyOf(lane);
  if (NO_ROLES.includes(vk)) return [];
  const own = block && block.roles && typeof block.roles === "object" ? Object.keys(block.roles) : [];
  if (vk === "anomaly") return own.length === 2 ? own : ANOMALY_SIDES.slice();
  const task = network && network.task;
  const names = (task && task.outputs && task.outputs.names) || (network && network.outputs);
  return Array.isArray(names) && names.length ? names.slice() : own;
}

// One name's role as the block states it, checked against the rules; problems are pushed.
function checkedRole(name, block, vk, problems) {
  const role = block.roles[name], basis = block.source && block.source[name];
  if (!ROLE_SET.includes(role)) { problems.push(problem("roles.problem.role", { name, role: shown(role) })); return DEFAULT_ROLE; }
  if (role === DEFAULT_ROLE) return role;
  const granted = grantsOf(basis).includes(role) && (role !== "alert" || basis !== "task:anomaly" || vk === "anomaly");
  if (!granted) {
    problems.push(role === "alert" ? problem("roles.problem.alert", { name, basis: shown(basis) })
      : problem("roles.problem.basis", { name, role, basis: shown(basis) }));
    return DEFAULT_ROLE;
  }
  return role;
}

function readRoles(block, lane, network, problems) {
  const out = new Map();
  if (block === null || block === undefined) return out;
  if (typeof block !== "object" || block.schema !== ROLES_SCHEMA || !block.roles || typeof block.roles !== "object") {
    problems.push(problem("roles.problem.block", { schema: ROLES_SCHEMA }));
    return out;
  }
  const vk = viewKeyOf(lane), names = roleNames(lane, network, block);
  if (NO_ROLES.includes(vk)) {
    if (Object.keys(block.roles).length) problems.push(problem("roles.problem.no_roles", { view_key: vk }));
    return out;
  }
  for (const name of Object.keys(block.roles)) {
    if (!names.includes(name)) { problems.push(problem("roles.problem.name", { name })); continue; }
    out.set(name, checkedRole(name, block, vk, problems));
  }
  if (ONE_NEUTRAL.includes(vk)) {
    const neutral = names.filter((n) => out.get(n) === "neutral");
    neutral.slice(1).forEach((n) => { out.set(n, DEFAULT_ROLE); problems.push(problem("roles.problem.neutral", { name: n, view_key: vk })); });
  }
  return out;
}

export class RoleMap {
  // block: story.json#class_roles (or null); lane: a Lane or page.json#lane; network: /v1/network.
  constructor(block, lane, network) {
    this.issues = [];
    this.map = readRoles(block, lane, network, this.issues);
  }

  role(name) { return this.map.get(name) || DEFAULT_ROLE; }
  cls(name) { return CLS[this.role(name)]; }
  // The canvas colour of a name: the alert colour for an alert, the accent otherwise.
  color(name, C) { return this.role(name) === "alert" ? C.alert : C.acc; }
  hasAlert() { return [...this.map.values()].includes("alert"); }
  isNeutral(name) { return this.role(name) === "neutral"; }
  isAlert(name) { return this.role(name) === "alert"; }
  // Every problem, in the order found: [{code, params}].
  problems() { return this.issues.slice(); }
}
