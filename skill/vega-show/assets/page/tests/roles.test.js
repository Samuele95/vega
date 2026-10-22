// node --test: core/roles.js — class roles (class_roles@1, R-E5). The page colours from roles and
// never from class names: plain by default; alert only with a person's word (user:, show:) or, on
// an anomaly lane, the task's anomalous side; neutral from any basis; at most one neutral where the
// catalog says so; no roles on the view keys that answer with a number. A role its basis does not
// grant is drawn plain and said: a machine code, a key of the strings table, with its params; the
// table's English is what the page shows. The closed sets are vega-show's catalog (roles).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  RoleMap, ROLES_SCHEMA, ROLE_SET, DEFAULT_ROLE, ALERT_REQUIRES, BASIS_RE, GRANTS, NO_ROLES, ONE_NEUTRAL, ANOMALY_SIDES,
  grantsOf, roleNames, viewKeyOf,
} from "../js/core/roles.js";
import { STRINGS } from "../js/core/strings.en.js";
import { Text } from "../js/core/text.js";

const CATALOG = JSON.parse(readFileSync(fileURLToPath(new URL("../../../scripts/catalog.json", import.meta.url)), "utf8"));
const ROLES = CATALOG.roles;
const block = (roles, source) => ({ schema: ROLES_SCHEMA, roles, source, proposed: { neutral: null, rule: "none", alert: [], evidence: [] } });
const net = (...names) => ({ available: true, outputs: names });
const lane = (viewKey, shape = "row") => ({ viewKey, shape });
const C = { acc: "teal", alert: "coral" };
const plain = (o) => JSON.parse(JSON.stringify(o));
// A problem as the page says it: its code through the strings table, its params as placeholders.
const TEXT = new Text();
const said = (problems) => problems.map((p) => TEXT.tText(p.code, p.params));
const ROLES_CODES = ["block", "no_roles", "name", "role", "alert", "basis", "neutral"].map((c) => `roles.problem.${c}`);

test("the closed sets are the catalog's: roles, default, bases and what each grants, alert, covers", () => {
  assert.deepEqual([...ROLE_SET], ROLES.set);
  assert.equal(DEFAULT_ROLE, ROLES.default);
  assert.equal(BASIS_RE.source, ROLES.basis_pattern);
  assert.deepEqual([...ALERT_REQUIRES], ROLES.alert_requires);
  const kind = (b) => b.replace(/<.*>$/, "");
  assert.deepEqual(Object.keys(GRANTS).sort(), Object.keys(ROLES.basis).map(kind).sort());
  for (const [b, row] of Object.entries(ROLES.basis)) assert.deepEqual([...GRANTS[kind(b)]], row.grants, b);
  assert.deepEqual([...NO_ROLES].sort(), Object.entries(ROLES.covers).filter(([, v]) => v === null).map(([k]) => k).sort());
  assert.deepEqual([...ONE_NEUTRAL], ROLES.neutral_at_most_one);
  assert.deepEqual([...ANOMALY_SIDES], ROLES.anomaly_sides);
  assert.deepEqual(Object.keys(ROLES.covers).sort(), [...new Set(Object.values(CATALOG.view_keys).flatMap((v) => (typeof v === "string" ? [v] : Object.values(v))))].sort());
});

test("bases: what each grants; a basis outside the catalog grants nothing", () => {
  assert.deepEqual(grantsOf("user:turn 4"), ["neutral", "alert", "plain"]);
  assert.deepEqual(grantsOf("show:2026-09-01"), ["neutral", "alert", "plain"]);
  assert.deepEqual(grantsOf("task:anomaly"), ["neutral", "alert"]);
  assert.deepEqual(grantsOf("structural:majority"), ["neutral"]);
  assert.deepEqual(grantsOf("vega:default"), ["neutral", "plain"]);
  for (const bad of ["show:yesterday", "domain:security", "spec:§2", "user:", "structural:minority", null, 3]) assert.deepEqual(grantsOf(bad), [], String(bad));
});

test("without a block every class is plain, drawn in the accent, and nothing is an alert", () => {
  const roles = new RoleMap(null, lane("classes"), net("a", "b"));
  assert.equal(roles.role("a"), "plain");
  assert.equal(roles.cls("a"), "role-plain");
  assert.equal(roles.color("a", C), "teal");
  assert.equal(roles.hasAlert(), false);
  assert.equal(roles.isNeutral("a"), false);
  assert.deepEqual(roles.problems(), []);
});

test("alert needs a person's word: user: or show: grant it, vega:default and the structural bases do not", () => {
  const names = net("calm", "gale", "squall", "drizzle");
  const roles = new RoleMap(block(
    { calm: "neutral", gale: "alert", squall: "alert", drizzle: "alert" },
    { calm: "structural:majority", gale: "user:turn 3", squall: "show:2026-09-01", drizzle: "vega:default" }), lane("classes"), names);
  assert.equal(roles.role("calm"), "neutral");
  assert.equal(roles.cls("calm"), "role-neutral");
  assert.equal(roles.role("gale"), "alert");
  assert.equal(roles.cls("gale"), "role-alert");
  assert.equal(roles.color("gale", C), "coral");
  assert.equal(roles.role("squall"), "alert");
  assert.equal(roles.role("drizzle"), "plain");
  assert.equal(roles.color("drizzle", C), "teal");
  assert.equal(roles.hasAlert(), true);
  assert.equal(roles.isAlert("gale"), true);
  assert.deepEqual(plain(roles.problems()), [{ code: "roles.problem.alert", params: { name: "drizzle", basis: '"vega:default"' } }]);
  assert.deepEqual(said(roles.problems()), ['class_roles.drizzle: alert needs user:, show: or task:anomaly (basis "vega:default"); drawn plain']);
  const structural = new RoleMap(block({ calm: "alert" }, { calm: "structural:negative" }), lane("binary"), net("calm", "storm"));
  assert.equal(structural.role("calm"), "plain");
  const noBasis = new RoleMap(block({ calm: "alert" }, {}), lane("classes"), names);
  assert.equal(noBasis.role("calm"), "plain");
  assert.equal(noBasis.hasAlert(), false);
  assert.deepEqual(said(noBasis.problems()), ["class_roles.calm: alert needs user:, show: or task:anomaly (basis undefined); drawn plain"]);
  const neutral = new RoleMap(block({ calm: "neutral" }, { calm: "domain:weather" }), lane("classes"), names);
  assert.equal(neutral.role("calm"), "plain", "a basis outside the catalog grants no neutral");
  assert.deepEqual(plain(neutral.problems()), [{ code: "roles.problem.basis", params: { name: "calm", role: "neutral", basis: '"domain:weather"' } }]);
  assert.deepEqual(said(neutral.problems()), ['class_roles.calm: neutral needs a basis that grants it (basis "domain:weather"); drawn plain']);
});

test("task:anomaly grants the alert on an anomaly lane only, and the anomalous side keeps it", () => {
  const sides = block({ normal: "neutral", anomalous: "alert" }, { normal: "task:anomaly", anomalous: "task:anomaly" });
  const anomaly = new RoleMap(sides, lane("anomaly"), null);
  assert.equal(anomaly.role("normal"), "neutral");
  assert.equal(anomaly.role("anomalous"), "alert");
  assert.equal(anomaly.hasAlert(), true);
  const named = new RoleMap(block({ ok: "neutral", fraud: "alert" }, { ok: "task:anomaly", fraud: "task:anomaly" }), lane("anomaly"), null);
  assert.equal(named.role("fraud"), "alert", "the two sides are named by the task's classes when a label exists");
  const elsewhere = new RoleMap(block({ a: "alert" }, { a: "task:anomaly" }), lane("classes"), net("a", "b"));
  assert.equal(elsewhere.role("a"), "plain");
  assert.equal(elsewhere.problems()[0].code, "roles.problem.alert");
  assert.match(said(elsewhere.problems())[0], /class_roles\.a: alert needs/);
});

test("neutral: from any basis that grants it; at most one where the catalog says so, the first in output order", () => {
  const two = new RoleMap(block({ b: "neutral", a: "neutral" }, { a: "structural:majority", b: "user:turn 2" }), lane("classes"), net("a", "b", "c"));
  assert.equal(two.role("a"), "neutral");
  assert.equal(two.role("b"), "plain");
  assert.deepEqual(plain(two.problems()), [{ code: "roles.problem.neutral", params: { name: "b", view_key: "classes" } }]);
  assert.deepEqual(said(two.problems()), ["class_roles.b: a second neutral on the classes view key; drawn plain"]);
  const labels = new RoleMap(block({ a: "neutral", b: "neutral" }, { a: "vega:default", b: "vega:default" }), lane("labels"), net("a", "b", "c"));
  assert.equal(labels.role("a"), "neutral");
  assert.equal(labels.role("b"), "neutral");
  assert.deepEqual(labels.problems(), []);
  const ordinal = new RoleMap(block({ low: "neutral", mid: "neutral" }, { low: "user:t", mid: "user:t" }), lane("ordinal"), net("low", "mid", "high"));
  assert.equal(ordinal.role("mid"), "neutral");
});

test("value, reconstruction and horizon carry no roles", () => {
  for (const vk of NO_ROLES) {
    const roles = new RoleMap(block({ x: "alert" }, { x: "user:t" }), lane(vk), net("x"));
    assert.equal(roles.role("x"), "plain", vk);
    assert.equal(roles.hasAlert(), false, vk);
    assert.deepEqual(plain(roles.problems()), [{ code: "roles.problem.no_roles", params: { view_key: vk } }]);
    assert.deepEqual(said(roles.problems()), [`class_roles: the ${vk} view key carries no roles`]);
    assert.deepEqual(roleNames(lane(vk), net("x"), null), []);
  }
});

test("only the lane's own names: the network's outputs, the task's names, or the anomaly sides", () => {
  assert.deepEqual(roleNames(lane("classes"), net("a", "b"), null), ["a", "b"]);
  assert.deepEqual(roleNames(lane("classes"), { task: { outputs: { names: ["x", "y"] } }, outputs: ["a"] }, null), ["x", "y"]);
  assert.deepEqual(roleNames(lane("classes"), null, block({ q: "neutral" }, { q: "user:t" })), ["q"], "without a network, the block's own names");
  assert.deepEqual(roleNames(lane("anomaly"), null, null), ["normal", "anomalous"]);
  const roles = new RoleMap(block({ a: "neutral", zz: "alert" }, { a: "user:t", zz: "user:t" }), lane("classes"), net("a", "b"));
  assert.equal(roles.role("zz"), "plain");
  assert.deepEqual(plain(roles.problems()), [{ code: "roles.problem.name", params: { name: "zz" } }]);
  assert.deepEqual(said(roles.problems()), ["class_roles.zz: not a name of this lane"]);
});

test("a block of another schema, or an unknown role, is drawn plain and said", () => {
  const other = new RoleMap({ schema: "vega/class-roles@2", roles: { a: "alert" }, source: { a: "user:t" } }, lane("classes"), net("a"));
  assert.equal(other.role("a"), "plain");
  assert.deepEqual(plain(other.problems()), [{ code: "roles.problem.block", params: { schema: ROLES_SCHEMA } }]);
  assert.deepEqual(said(other.problems()), ["class_roles: the block is not vega/class-roles@1; every class is plain"]);
  const unknown = new RoleMap(block({ a: "danger" }, { a: "user:t" }), lane("classes"), net("a"));
  assert.equal(unknown.role("a"), "plain");
  assert.deepEqual(plain(unknown.problems()), [{ code: "roles.problem.role", params: { name: "a", role: '"danger"' } }]);
  assert.match(said(unknown.problems())[0], /unknown role "danger"/);
});

test("every problem code is a key of the strings table, said with every placeholder filled", () => {
  for (const code of ROLES_CODES) assert.equal(typeof STRINGS[code], "string", code);
  const seen = [
    ...new RoleMap({ schema: "x", roles: {} }, lane("classes"), net("a")).problems(),
    ...new RoleMap(block({ x: "alert" }, { x: "user:t" }), lane("value"), net("x")).problems(),
    ...new RoleMap(block({ a: "neutral", b: "neutral", c: "alert", d: "danger", e: "neutral", zz: "plain" },
      { a: "user:t", b: "user:t", c: "vega:default", e: "domain:x" }), lane("classes"), net("a", "b", "c", "d", "e")).problems(),
  ];
  assert.deepEqual([...new Set(seen.map((p) => p.code))].sort(), [...ROLES_CODES].sort());
  for (const p of seen) {
    assert.ok(Object.isFrozen(p) && Object.isFrozen(p.params), p.code);
    assert.ok(Object.values(p.params).every((v) => typeof v === "string"), p.code);
  }
  const tx = new Text();
  seen.forEach((p) => tx.tText(p.code, p.params));
  assert.deepEqual(tx.missing(), []);
  assert.deepEqual(tx.unresolved(), []);
});

test("the lane: a Lane (viewKey) or page.json#lane (view_key)", () => {
  assert.equal(viewKeyOf({ viewKey: "binary" }), "binary");
  assert.equal(viewKeyOf({ view_key: "value" }), "value");
  assert.equal(viewKeyOf(null), null);
  const fromPage = new RoleMap(block({ no: "neutral", yes: "alert" }, { no: "structural:negative", yes: "user:t" }), { view_key: "binary", shape: "row" }, net("no", "yes"));
  assert.equal(fromPage.role("no"), "neutral");
  assert.equal(fromPage.role("yes"), "alert");
});
