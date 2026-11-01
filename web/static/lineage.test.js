"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { build } = require("./lineage.js");

const root = (id = "p", extra = {}) => ({ key: `project:${id}`, type: "project", row: { project_id: id }, ...extra });
const run = (id, parent, extra = {}) => ({ key: `run:${id}`, type: "run", runId: id,
  row: { run_id: id, ...(parent === undefined ? {} : { parent_run_id: parent }) }, ...extra });
const get = (model, id) => model.nodes.find(node => node.runId === id);
const links = (model, kind) => model.edges.filter(edge => edge.kind === kind).map(edge => [edge.from, edge.to]);
const geometry = model => model.nodes.map(node => [node.key, node.depth, node.iterationDepth, node.treeRow, node.treeParentKey]);
function frozen(value) {
  if (value && typeof value === "object") { Object.values(value).forEach(frozen); Object.freeze(value); }
  return value;
}
function shuffled(values, seed) {
  const result = values.slice();
  for (let i = result.length - 1; i > 0; i--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const j = seed % (i + 1); [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
const canonical = model => ({ nodes: model.nodes.slice().sort((a, b) => a.key.localeCompare(b.key)), edges: model.edges });

test("loads as a browser global without CommonJS or DOM", () => {
  const ctx = vm.createContext({});
  vm.runInContext(fs.readFileSync(`${__dirname}/lineage.js`, "utf8"), ctx);
  assert.equal(typeof ctx.RSILineage.build, "function");
  assert.equal(JSON.stringify(ctx.RSILineage.build([], [])), '{"nodes":[],"edges":[]}');
  assert.ok(Object.isFrozen(ctx.RSILineage));
});

test("empty input and a project without runs have finite zero geometry", () => {
  assert.deepEqual(build(), { nodes: [], edges: [] });
  const model = build([root()]);
  assert.deepEqual(geometry(model), [["project:p", 0, 0, 0, null]]);
  assert.equal(model.nodes[0].ancestryStatus, "root");
  assert.equal(model.nodes[0].ancestryReason, null);
});

test("initial runs attach only to the project and occupy separate sibling rows", () => {
  const model = build([root(), run("b", undefined, { layoutOrder: 1 }), run("a", undefined, { layoutOrder: 0 })]);
  assert.deepEqual(links(model, "membership"), [["project:p", "run:a"], ["project:p", "run:b"]]);
  for (const id of ["a", "b"]) {
    const node = get(model, id);
    assert.equal(node.depth, 1); assert.equal(node.iterationDepth, 0);
    assert.equal(node.ancestryStatus, "initial"); assert.equal(node.ancestryReason, null);
    assert.equal(node.treeParentKey, "project:p");
  }
  assert.equal(get(model, "a").treeRow, 0); assert.equal(get(model, "b").treeRow, 1);
  assert.equal(model.nodes[0].treeRow, 0.5);
});

test("explicit chain increases only valid parent hops and removes child root memberships", () => {
  const nodes = [root(), run("a"), run("b", "a"), run("c", "b")];
  const oldEdges = nodes.slice(1).map(node => ({ from: "project:p", to: node.key, kind: "membership" }));
  const model = build(nodes, oldEdges);
  assert.deepEqual(links(model, "membership"), [["project:p", "run:a"]]);
  assert.deepEqual(links(model, "run-parent"), [["run:a", "run:b"], ["run:b", "run:c"]]);
  assert.deepEqual(model.nodes.map(node => node.depth), [0, 1, 2, 3]);
  assert.deepEqual(model.nodes.map(node => node.iterationDepth), [0, 0, 1, 2]);
  assert.deepEqual(model.nodes.map(node => node.treeRow), [0, 0, 0, 0]);
  assert.equal(get(model, "c").parentRunId, "b");
  assert.equal(get(model, "c").ancestryStatus, "known");
});

test("branch layout separates leaves and siblings and centers each parent on its leaf span", () => {
  const model = build([root(), run("a", undefined, { layoutOrder: 0 }), run("z", undefined, { layoutOrder: 1 }),
    run("c", "a", { layoutOrder: 2 }), run("b", "a", { layoutOrder: 1 }),
    run("e", "b", { layoutOrder: 1 }), run("d", "b", { layoutOrder: 0 }), run("f", "c")]);
  assert.deepEqual(["d", "e", "f", "z"].map(id => get(model, id).treeRow), [0, 1, 2, 3]);
  assert.deepEqual(["a", "b", "c"].map(id => get(model, id).treeRow), [1, 0.5, 2]);
  assert.equal(model.nodes[0].treeRow, 1.5);
  assert.equal(get(model, "b").depth, 2); assert.equal(get(model, "d").depth, 3);
});

test("a missing declared parent is unknown, while its valid descendants keep bounded hops", () => {
  const model = build([root(), run("a", "absent"), run("b", "a"), run("c", "b")]);
  assert.equal(get(model, "a").ancestryReason, "missing-parent");
  assert.deepEqual(get(model, "a").missing, ["absent"]);
  assert.deepEqual(["a", "b", "c"].map(id => get(model, id).ancestryStatus), ["unknown", "unknown", "unknown"]);
  assert.deepEqual(["a", "b", "c"].map(id => get(model, id).depth), [1, 2, 3]);
  assert.deepEqual(["a", "b", "c"].map(id => get(model, id).iterationDepth), [0, 1, 2]);
  assert.equal(get(model, "c").ancestryReason, "ancestor-unknown");
  assert.deepEqual(links(model, "membership"), [["project:p", "run:a"]]);
});

test("self-parent is unknown with no self edge", () => {
  const model = build([root(), run("self", "self"), run("child", "self")]);
  assert.equal(get(model, "self").ancestryReason, "self-parent");
  assert.equal(get(model, "self").depth, 1); assert.equal(get(model, "self").iterationDepth, 0);
  assert.deepEqual(links(model, "run-parent"), [["run:self", "run:child"]]);
  assert.deepEqual(links(model, "membership"), [["project:p", "run:self"]]);
  assert.equal(get(model, "child").ancestryStatus, "unknown");
});

test("all members of a cycle reset to the root, independently of traversal entry", () => {
  const nodes = [root(), run("a", "b"), run("b", "c"), run("c", "a"), run("d", "c"), run("e", "d"), run("valid")];
  for (let seed = 1; seed <= 12; seed++) {
    const model = build(shuffled(nodes, seed));
    for (const id of ["a", "b", "c"]) {
      assert.equal(get(model, id).ancestryReason, "cycle");
      assert.equal(get(model, id).depth, 1); assert.equal(get(model, id).iterationDepth, 0);
      assert.equal(get(model, id).treeParentKey, "project:p");
    }
    assert.deepEqual(links(model, "run-parent"), [["run:c", "run:d"], ["run:d", "run:e"]]);
    assert.equal(get(model, "d").depth, 2); assert.equal(get(model, "e").depth, 3);
    assert.equal(get(model, "e").ancestryReason, "ancestor-unknown");
    assert.equal(get(model, "valid").ancestryStatus, "initial");
  }
});

test("disconnected cycles and self cycles do not infect independent branches", () => {
  const model = build([root(), run("a", "b"), run("b", "a"), run("c", "d"), run("d", "c"),
    run("e", "e"), run("initial"), run("valid", "initial")]);
  assert.equal(links(model, "run-parent").length, 1);
  assert.equal(get(model, "valid").depth, 2);
  assert.equal(get(model, "valid").ancestryStatus, "known");
  assert.equal(get(model, "e").ancestryReason, "self-parent");
});

test("multiple sources create reference edges without adding depth or changing tree rows", () => {
  const nodes = [root(), run("a"), run("b", "a"), run("c", "b"), run("initial"), run("child", "b")];
  const before = build(nodes);
  nodes[4].row.source_run_ids = ["a", "b", "c", "a", "initial", "missing", null, 7];
  nodes[5].row.source_run_ids = ["a", "b", "c"];
  const model = build(nodes);
  assert.deepEqual(geometry(model), geometry(before));
  assert.equal(get(model, "initial").ancestryStatus, "initial");
  assert.deepEqual(get(model, "initial").missing, ["missing"]);
  assert.equal(get(model, "initial").sourceReferences.find(ref => ref.runId === "initial").status, "self");
  assert.deepEqual(links(model, "run-source"), [["run:a", "run:child"], ["run:b", "run:child"],
    ["run:c", "run:child"], ["run:a", "run:initial"], ["run:b", "run:initial"], ["run:c", "run:initial"]]);
  assert.equal(get(model, "child").depth, 3); assert.equal(get(model, "child").iterationDepth, 2);
});

test("reference-only cycles never become ancestry cycles", () => {
  const model = build([root(), run("a", undefined, { row: { run_id: "a", source_run_ids: ["b"] } }),
    run("b", undefined, { row: { run_id: "b", source_run_ids: ["a"] } })]);
  assert.deepEqual(model.nodes.slice(1).map(node => [node.depth, node.iterationDepth, node.ancestryStatus]),
    [[1, 0, "initial"], [1, 0, "initial"]]);
  assert.equal(links(model, "run-source").length, 2);
});

test("duplicate run IDs with unique graph keys make references ambiguous", () => {
  const model = build([root(), run("dup", undefined, { key: "first" }), run("dup", undefined, { key: "second" }),
    run("child", "dup", { row: { run_id: "child", parent_run_id: "dup", source_run_ids: ["dup"] } })]);
  assert.equal(get(model, "child").ancestryReason, "ambiguous-parent");
  assert.deepEqual(get(model, "child").ambiguous, ["dup"]);
  assert.deepEqual(links(model, "run-parent"), []); assert.deepEqual(links(model, "run-source"), []);
  assert.equal(get(model, "child").depth, 1);
});

test("duplicate graph keys never emit an ambiguous endpoint or collapse input nodes", () => {
  const model = build([root(), run("dup"), run("dup"), run("child", "dup")]);
  assert.equal(model.nodes.length, 4);
  assert.deepEqual(model.nodes.slice(1, 3).map(node => node.ancestryReason), ["duplicate-key", "duplicate-key"]);
  assert.equal(get(model, "child").ancestryReason, "ambiguous-parent");
  assert.ok(model.edges.every(edge => edge.from !== "run:dup" && edge.to !== "run:dup"));
  assert.equal(new Set(model.nodes.slice(1).map(node => node.treeRow)).size, 3);
});

test("row is completely authoritative, including an absent parent or absent sources", () => {
  const model = build([root(), run("a"), run("b", undefined, {
    snapshot: { run_id: "b", parent_run_id: "a", source_run_ids: ["a"] },
  }), run("c", "a", { snapshot: { run_id: "c", parent_run_id: "b" } })]);
  assert.equal(get(model, "b").depth, 1); assert.equal(get(model, "b").parentRunId, null);
  assert.deepEqual(get(model, "b").sourceReferences, []);
  assert.deepEqual(links(model, "run-parent"), [["run:a", "run:c"]]);
});

test("snapshot is used only when row is unavailable, with record run_id as identity fallback", () => {
  const model = build([root(), { key: "a", type: "run", snapshot: { run_id: "a" } },
    { key: "b", type: "run", snapshot: { run_id: "b", parent_run_id: "a", source_run_ids: ["a"] } }]);
  assert.deepEqual(links(model, "run-parent"), [["a", "b"]]);
  assert.deepEqual(links(model, "run-source"), [["a", "b"]]);
  assert.equal(model.nodes[2].iterationDepth, 1);
});

test("status, accepted flags, scores, timestamps and old edges never infer continuation", () => {
  const model = build([root(), run("a", undefined, { row: { run_id: "a", status: "accepted", score: 1, applied: true } }),
    run("b", undefined, { row: { run_id: "b", status: "completed", base_revision: "a", updated_at: "2099-01-01" } }),
    run("c", "a", { row: { run_id: "c", parent_run_id: "a", status: "failed", score: -1 } })],
  [{ from: "run:a", to: "run:b", kind: "run-parent" }, { from: "run:a", to: "run:b", kind: "time" }]);
  assert.equal(get(model, "b").ancestryStatus, "initial");
  assert.equal(get(model, "b").iterationDepth, 0);
  assert.deepEqual(links(model, "run-parent"), [["run:a", "run:c"]]);
  assert.equal(get(model, "c").ancestryStatus, "known");
  assert.equal(model.edges.some(edge => edge.kind === "time"), false);
});

test("project ownership confines both parents and references even with matching IDs elsewhere", () => {
  const nodes = [root("p"), root("q"), run("same", undefined, { key: "p:same", projectId: "p" }),
    run("same", undefined, { key: "q:same", projectId: "q" }),
    run("outside", undefined, { projectId: "q" }),
    run("child", "same", { projectId: "p", row: { run_id: "child", parent_run_id: "same", source_run_ids: ["same", "outside"] } }),
    run("missing", "outside", { projectId: "p" })];
  const model = build(nodes, [{ from: "run:outside", to: "run:child", kind: "run-source" }]);
  assert.deepEqual(links(model, "run-parent"), [["p:same", "run:child"]]);
  assert.deepEqual(links(model, "run-source"), [["p:same", "run:child"]]);
  assert.deepEqual(get(model, "child").missing, ["outside"]);
  assert.equal(get(model, "missing").ancestryReason, "missing-parent");
  assert.equal(get(model, "missing").treeParentKey, "project:p");
});

test("membership can establish project scope without adding root edges to valid children", () => {
  const nodes = [root("p"), root("q"), run("same", undefined, { key: "p:same" }),
    run("same", undefined, { key: "q:same" }), run("child", "same")];
  const model = build(nodes, [{ from: "project:p", to: "p:same", kind: "membership" },
    { from: "project:q", to: "q:same", kind: "membership" },
    { from: "project:p", to: "run:child", kind: "membership" }]);
  assert.deepEqual(links(model, "run-parent"), [["p:same", "run:child"]]);
  assert.equal(links(model, "membership").length, 2);
  assert.equal(get(model, "child").treeParentKey, "p:same");
});

test("projectKey and row project metadata establish scope; uncertain ownership stays unknown", () => {
  const model = build([root("p"), root("q"), run("a", undefined, { projectKey: "project:p" }),
    run("b", "a", { row: { run_id: "b", parent_run_id: "a", project: { project_id: "p" } } }),
    run("c", "b", { row: { run_id: "c", parent_run_id: "b", project_id: "p" } }),
    run("unassigned"), run("unknown", undefined, { projectId: "absent" })]);
  assert.equal(get(model, "c").depth, 3);
  for (const id of ["unassigned", "unknown"]) {
    assert.equal(get(model, id).ancestryReason, "unknown-project");
    assert.equal(get(model, id).treeParentKey, null);
    assert.equal(model.edges.some(edge => edge.to === `run:${id}`), false);
  }
  const conflicting = build([root("p"), root("q"), run("a", undefined, { projectId: "p" })],
    [{ from: "project:q", to: "run:a", kind: "membership" }]);
  assert.equal(get(conflicting, "a").ancestryStatus, "unknown");
  assert.equal(get(conflicting, "a").treeParentKey, null);
});

test("ambiguous project ownership is never resolved by picking the first root", () => {
  const model = build([root("p", { key: "first" }), root("p", { key: "second" }), run("a", undefined, { projectId: "p" })]);
  assert.equal(get(model, "a").ancestryReason, "ambiguous-project");
  assert.deepEqual(model.edges, []);
  const memberships = build([root("p"), root("q"), run("a")], [
    { from: "project:p", to: "run:a", kind: "membership" }, { from: "project:q", to: "run:a", kind: "membership" }]);
  assert.equal(get(memberships, "a").ancestryReason, "ambiguous-project");
});

test("an explicitly missing project cannot fall back to the only visible project", () => {
  const model = build([root(), run("a", undefined, { projectId: "elsewhere" }), run("b", "a")]);
  assert.equal(get(model, "a").treeParentKey, null);
  assert.equal(get(model, "a").ancestryReason, "unknown-project");
  assert.equal(get(model, "b").ancestryReason, "missing-parent");
});

test("input permutations give identical lineage, row layout and edges", () => {
  const nodes = [root(), run("a", undefined, { layoutOrder: 9 }), run("b", "a", { layoutOrder: 2 }),
    run("c", "a", { layoutOrder: 2 }), run("d", "b", { layoutOrder: 4 }),
    run("e", "c", { row: { run_id: "e", parent_run_id: "c", source_run_ids: ["d", "a"] } }),
    run("x", "y"), run("y", "x"), run("missing", "absent")];
  const edges = nodes.slice(1).map(node => ({ from: "project:p", to: node.key, kind: "membership" }));
  const expected = canonical(build(nodes, edges));
  for (let seed = 1; seed <= 40; seed++) assert.deepEqual(canonical(build(shuffled(nodes, seed), shuffled(edges, seed + 90))), expected);
  assert.ok(get(expected, "b").treeRow < get(expected, "c").treeRow);
});

test("source ordering and duplicate declarations do not affect output", () => {
  const a = [root(), run("a"), run("b"), run("c", undefined, { row: { run_id: "c", source_run_ids: ["b", "a", "b"] } })];
  const b = structuredClone(a); b[3].row.source_run_ids = ["a", "b"];
  const first = build(a), second = build(b);
  assert.deepEqual(first.edges, second.edges);
  assert.deepEqual(get(first, "c").sourceReferences, get(second, "c").sourceReferences);
  assert.deepEqual(geometry(first), geometry(second));
});

test("matching edge decoration survives, stale parent/source/membership edges do not", () => {
  const model = build([root(), run("a"), run("b", "a")], [
    { from: "run:a", to: "run:b", kind: "run-parent", evidence: ["declared"], custom: 7 },
    { from: "project:p", to: "run:a", kind: "membership", evidence: ["owner"] },
    { from: "project:p", to: "run:b", kind: "membership" },
    { from: "run:b", to: "run:a", kind: "run-parent" },
    { from: "run:a", to: "run:b", kind: "run-source" },
  ]);
  assert.equal(model.edges.length, 2);
  assert.deepEqual(model.edges.find(edge => edge.kind === "run-parent").evidence, ["declared"]);
  assert.equal(model.edges.find(edge => edge.kind === "run-parent").custom, 7);
});

test("build is pure, preserves input order and arbitrary fields, and is repeatable", () => {
  const nodes = frozen([run("b", "a", { x: 23, y: 40, missing: ["stale"], ambiguous: ["stale"], ancestryReason: "stale" }),
    root(), run("a")]);
  const edges = frozen([{ from: "project:p", to: "run:b", kind: "membership", evidence: ["x"] }]);
  const model = build(nodes, edges);
  assert.deepEqual(model.nodes.map(node => node.key), nodes.map(node => node.key));
  assert.deepEqual([model.nodes[0].x, model.nodes[0].y], [23, 40]);
  assert.equal(model.nodes[0].row, nodes[0].row);
  assert.notEqual(model.nodes[0], nodes[0]);
  assert.deepEqual(model.nodes[0].missing, []); assert.deepEqual(model.nodes[0].ambiguous, []);
  assert.deepEqual(build(model.nodes, model.edges), model);
  assert.deepEqual(nodes[0].missing, ["stale"]);
});

test("invalid parent values are unknown; null/empty parents are initial and malformed sources are ignored", () => {
  for (const value of [9, false, [], {}, "  "]) {
    const model = build([root(), run("a", value)]);
    assert.equal(get(model, "a").ancestryReason, "invalid-parent");
    assert.equal(get(model, "a").depth, 1);
  }
  for (const value of [undefined, null, ""]) {
    const model = build([root(), run("a", value, { row: { run_id: "a", parent_run_id: value, source_run_ids: "a" } })]);
    assert.equal(get(model, "a").ancestryStatus, "initial");
    assert.deepEqual(get(model, "a").sourceReferences, []);
  }
});

test("missing run identity stays unknown and identifiers use safe exact map keys", () => {
  const model = build([root(), { key: "no-id", type: "run", row: {} }, run("__proto__"), run("constructor", "__proto__")]);
  assert.equal(model.nodes[1].ancestryReason, "missing-run-id");
  assert.equal(get(model, "constructor").depth, 2);
  assert.deepEqual(links(model, "run-parent"), [["run:__proto__", "run:constructor"]]);
});

test("50,000-run chain resolves without recursion or quadratic ancestor expansion", () => {
  const size = 50000;
  const nodes = [root()];
  for (let i = size - 1; i >= 0; i--) nodes.push(run(String(i), i ? String(i - 1) : undefined, { layoutOrder: i }));
  const model = build(nodes);
  assert.equal(model.nodes.length, size + 1);
  assert.equal(model.edges.length, size);
  assert.equal(links(model, "membership").length, 1);
  assert.equal(get(model, String(size - 1)).depth, size);
  assert.equal(get(model, String(size - 1)).iterationDepth, size - 1);
  assert.ok(model.nodes.every(node => node.treeRow === 0));
});

test("a large branching tree has one distinct row per leaf and centers the root", () => {
  const size = 16000, nodes = [root()];
  for (let i = 0; i < size; i++) nodes.push(run(String(i), i ? String(Math.floor((i - 1) / 3)) : undefined, { layoutOrder: i }));
  const model = build(shuffled(nodes, 27));
  const leaves = model.nodes.filter(node => node.type === "run" && Number(node.runId) * 3 + 1 >= size);
  assert.equal(new Set(leaves.map(node => node.treeRow)).size, leaves.length);
  assert.ok(leaves.every(node => Number.isInteger(node.treeRow)));
  assert.equal(model.nodes.find(node => node.type === "project").treeRow, (leaves.length - 1) / 2);
  assert.equal(links(model, "membership").length, 1);
  assert.equal(model.edges.length, size);
  assert.ok(model.nodes.every(node => Number.isFinite(node.depth) && Number.isFinite(node.treeRow)));
});

test("a large cycle with later descendants resets cycle depth before counting descendants", () => {
  const cycleSize = 12000, tailSize = 2000, nodes = [root()];
  for (let i = 0; i < cycleSize; i++) nodes.push(run(`cycle${i}`, `cycle${(i + 1) % cycleSize}`));
  for (let i = 0; i < tailSize; i++) nodes.push(run(`tail${i}`, i ? `tail${i - 1}` : "cycle0"));
  const model = build(shuffled(nodes, 8));
  const cycles = model.nodes.filter(node => node.runId?.startsWith("cycle"));
  assert.ok(cycles.every(node => node.depth === 1 && node.iterationDepth === 0 && node.ancestryReason === "cycle"));
  assert.equal(get(model, `tail${tailSize - 1}`).depth, tailSize + 1);
  assert.equal(get(model, `tail${tailSize - 1}`).iterationDepth, tailSize);
  assert.equal(get(model, `tail${tailSize - 1}`).ancestryReason, "ancestor-unknown");
  assert.equal(links(model, "membership").length, cycleSize);
  assert.equal(links(model, "run-parent").length, tailSize);
});
