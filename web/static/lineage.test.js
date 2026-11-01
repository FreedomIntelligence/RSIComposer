"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

// Run the actual frontend functions without starting polling or requiring a DOM library.
const source = fs.readFileSync(`${__dirname}/app.js`, "utf8");
const context = vm.createContext({
  document: { getElementById: () => ({ addEventListener() {} }), addEventListener() {} },
});
vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), context);
function evaluate(expression, fixture) {
  context.fixture = fixture;
  return JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
}
const model = fixture => evaluate("lineageModel(fixture)", fixture);
const decision = (id, extra = {}) => ({ decision_id: id, kind: "directive", mode: "train", ...extra });
function relations(graph, kind) {
  const ids = new Map(graph.nodes.map(n => [n.key, n.row.decision_id || n.row.job_id || n.row.candidate_id]));
  return graph.edges.filter(e => !kind || e.kind === kind).map(e => [ids.get(e.from), ids.get(e.to), e.kind]);
}

test("explicit multiple parents and forward references resolve only within the snapshot", () => {
  const graph = model({ decisions: [
    decision("child", { parent_decision_id: "a", source_decision_ids: ["a", "b", "outside-run", null, 7] }),
    decision("b"), decision("a"),
  ] });
  assert.deepEqual(relations(graph, "explicit").sort(), [["a", "child", "explicit"], ["b", "child", "explicit"]]);
  assert.equal(graph.nodes.length, 3);
  assert.deepEqual(graph.nodes.find(n => n.row.decision_id === "child").missing, ["outside-run"]);
  const anotherRun = model({ decisions: [decision("child", { parent_decision_id: "a" })] });
  assert.deepEqual(anotherRun.edges, []);
});

test("ambiguous parent IDs and self references never create guessed edges", () => {
  const graph = model({ decisions: [decision("dup"), decision("dup"), decision("child", { source_decision_ids: ["dup", "child"] })] });
  assert.deepEqual(graph.edges, []);
  assert.deepEqual(graph.nodes.find(n => n.row.decision_id === "child").ambiguous, ["dup"]);
});

test("only known, strict timestamp order produces dashed temporal edges", () => {
  const graph = model({ decisions: [
    decision("a", { timestamp: "2026-11-01T00:00:00Z" }),
    decision("b", { timestamp: "2026-11-01T00:01:00Z" }),
  ] });
  assert.deepEqual(relations(graph), [["a", "b", "time"]]);
  for (const timestamp of [undefined, "invalid", "2026-11-01T00:00:00Z"]) {
    assert.deepEqual(model({ decisions: [decision("a", { timestamp: "2026-11-01T00:00:00Z" }), decision("b", { timestamp })] }).edges, []);
  }
});

test("matching names, rounds, candidate IDs and revisions never imply parenthood", () => {
  const shared = { round: 1, candidate_id: "same", actor_revision: "r1", reason: "同一研究目标" };
  assert.deepEqual(model({ decisions: [decision("a", shared), decision("b", shared)] }).edges, []);
});

test("experiment and candidate feedback require a unique original directive", () => {
  const graph = model({ decisions: [decision("a", { directive_id: "dir-a", job_id: "claimed" }),
    { decision_id: "review", kind: "review", directive_id: "dir-a", action: "continue" }],
  jobs: [{ job_id: "job", directive_id: "dir-a" }, { job_id: "claimed", directive_id: "other" }],
  candidates: [{ candidate_id: "candidate", directive_id: "dir-a" }, { candidate_id: "unlinked" }],
  });
  assert.deepEqual(relations(graph, "directive").sort(), [["a", "candidate", "directive"], ["a", "job", "directive"]]);
  assert.equal(graph.nodes.length, 6);
  const ambiguous = model({ decisions: [decision("a", { directive_id: "same" }), decision("b", { directive_id: "same" })], jobs: [{ job_id: "job", directive_id: "same" }] });
  assert.deepEqual(ambiguous.edges, []);
});

test("repeated finish compresses records while retaining every parent reference", () => {
  const finishes = Array.from({ length: 197 }, (_, i) => ({ decision_id: `f${i}`, kind: "finish", round: 1, status: "completed", reason: "本轮结束", parent_decision_id: i % 2 ? "a" : "b" }));
  const graph = model({ status: "running", decisions: [...finishes, decision("a"), decision("b")] });
  const finish = graph.nodes.find(n => n.rows?.length > 1);
  assert.equal(graph.nodes.length, 3);
  assert.equal(finish.rows.length, 197);
  assert.deepEqual(graph.edges.filter(e => e.to === finish.key).map(e => e.kind), ["explicit", "explicit"]);
  // Changing the recorded reason starts a separate group.
  assert.equal(model({ decisions: [finishes[0], { ...finishes[1], reason: "另一结论" }] }).nodes.length, 2);
});

test("empty and malformed inputs stay empty; IDs and prose remain inert text", () => {
  assert.deepEqual(model({}), { nodes: [], edges: [] });
  assert.deepEqual(model({ decisions: [null, false, 7], jobs: "invalid" }), { nodes: [], edges: [] });
  const text = '<img src=x onerror="alert(1)">';
  const graph = model({ decisions: [decision(text, { reason: text }), decision("child", { parent_decision_id: text })] });
  assert.equal(graph.nodes.find(n => n.row.decision_id === text).summary, text);
  assert.equal(graph.edges.length, 1);
  assert.equal(evaluate("projectMeta(fixture)", { name: "qwen-parity-rsi" }).project_id, "unassigned");
  assert.equal(evaluate("projectMeta(fixture)", {}).name, "未归属");
});
