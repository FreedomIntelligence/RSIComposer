"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { layout } = require("./decisions.js");
const source = fs.readFileSync(`${__dirname}/decisions.js`, "utf8");
const row = (id, second = 0, extra = {}) => ({ decision_id: id,
  timestamp: new Date(Date.UTC(2026, 10, 1, 0, 0, second)).toISOString(), ...extra });
const get = (model, key) => model.nodes.find(node => node.key === key);
const links = (model, kind) => model.edges.filter(edge => edge.kind === kind).map(edge => [edge.from, edge.to]);
const decisions = model => model.nodes.filter(node => node.type === "decision");
function freeze(value) {
  if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
function valid(model) {
  const byKey = new Map(model.nodes.map(node => [node.key, node]));
  assert.equal(byKey.size, model.nodes.length);
  for (const node of model.nodes) {
    assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y));
    assert.ok(node.x >= 18 && node.x + 160 < model.width);
    assert.ok(node.y >= 18 && node.y + 32 < model.height);
  }
  const structural = new Map();
  for (const edge of model.edges) {
    assert.ok(byKey.has(edge.from) && byKey.has(edge.to));
    assert.notEqual(edge.from, edge.to);
    if (edge.kind === "source") continue;
    assert.ok(byKey.get(edge.from).x < byKey.get(edge.to).x);
    assert.ok(!structural.has(edge.to));
    structural.set(edge.to, edge.from);
  }
  for (const node of decisions(model)) {
    const visited = new Set();
    let key = node.key;
    while (key !== "root") {
      assert.ok(!visited.has(key)); visited.add(key);
      assert.ok(structural.has(key)); key = structural.get(key);
    }
  }
  // A label occupies at most 160px horizontally and one 16px line below its dot.
  for (let i = 0; i < model.nodes.length; i++) {
    for (let j = i + 1; j < model.nodes.length; j++) {
      const a = model.nodes[i], b = model.nodes[j];
      assert.ok(Math.abs(a.x - b.x) >= 176 || Math.abs(a.y - b.y) >= 36);
    }
  }
}

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.attributes = {}; this.style = {}; this.listeners = {}; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...children) { this.children.push(...children); }
  addEventListener(name, listener) { this.listeners[name] = listener; }
}
function harness() {
  const document = { createElementNS(namespace, tag) {
    assert.equal(namespace, "http://www.w3.org/2000/svg"); return new Element(tag);
  } };
  const context = vm.createContext({ document });
  vm.runInContext(source, context);
  return context.RSIDecisions;
}
const renderedNodes = svg => svg.children.filter(node => node.attributes.class?.split(" ").includes("decision-node"));
const tag = (node, name) => node.children.filter(child => child.tagName === name);

test("CommonJS and browser global load without a DOM, timers, or dependencies", () => {
  const context = vm.createContext({});
  vm.runInContext(source, context);
  assert.equal(typeof context.RSIDecisions.layout, "function");
  assert.equal(typeof context.RSIDecisions.render, "function");
  assert.ok(Object.isFrozen(context.RSIDecisions));
  assert.deepEqual(JSON.parse(JSON.stringify(context.RSIDecisions.layout())), layout());
});

test("empty and invalid inputs retain a fixed, unselectable root and finite minimum size", () => {
  for (const input of [undefined, null, [], {}, [null, undefined, 3, "x", []]]) {
    const model = layout(input);
    assert.deepEqual(model.nodes, [{ key: "root", index: -1, row: null, x: 40, y: 100, type: "root" }]);
    assert.deepEqual(model.edges, []);
    assert.ok(model.height >= 240); valid(model);
  }
});

test("timestamps use actual instants, not lexical strings or input order", () => {
  const rows = [row("late", 3), row("early", 0, { timestamp: "2026-11-01T08:00:01+08:00" }), row("middle", 2)];
  const model = layout(freeze(rows));
  assert.deepEqual(decisions(model).map(node => [node.key, node.index]), [["early", 1], ["middle", 2], ["late", 0]]);
  assert.deepEqual(links(model, "sequence"), [["root", "early"], ["early", "middle"], ["middle", "late"]]);
  assert.equal(get(model, "early").row, rows[1]);
  assert.deepEqual(model, layout(rows)); valid(model);
});

test("epoch, numeric, tied and missing timestamps have deterministic ordering", () => {
  const model = layout([{ decision_id: "unknown" }, { decision_id: "bad", timestamp: "invalid" },
    { decision_id: "zero", timestamp: 0 }, { decision_id: "negative", timestamp: -1 },
    { decision_id: "tie", timestamp: "1970-01-01T00:00:00Z" }, { decision_id: "null", timestamp: null }]);
  assert.deepEqual(decisions(model).map(node => node.key), ["negative", "zero", "tie", "unknown", "bad", "null"]);
  valid(model);
});

test("real parent branches occupy distinct rows and override chronological predecessors", () => {
  const model = layout([row("a"), row("b", 1, { parent_decision_id: "a" }),
    row("c", 2, { parent_decision_id: "a" }), row("d", 3, { parent_decision_id: "b" }),
    row("e", 4, { parent_decision_id: "b" }), row("f", 5, { parent_decision_id: "c" })]);
  assert.deepEqual(links(model, "parent"), [["a", "b"], ["a", "c"], ["b", "d"], ["b", "e"], ["c", "f"]]);
  assert.deepEqual(links(model, "sequence"), [["root", "a"]]);
  assert.equal(get(model, "b").x, get(model, "c").x);
  assert.notEqual(get(model, "b").y, get(model, "c").y);
  assert.equal(get(model, "root").y, 100); valid(model);
});

test("missing or malformed parents fall back to sequence without fabricated references", () => {
  const model = layout([row("a", 0, { parent_decision_id: "absent" }),
    row("b", 1, { parent_decision_id: "other" }), row("c", 2, { parent_decision_id: 8 }),
    row("d", 3, { parent_decision_id: "  " })]);
  assert.deepEqual(links(model, "parent"), []);
  assert.deepEqual(links(model, "sequence"), [["root", "a"], ["a", "b"], ["b", "c"], ["c", "d"]]);
  valid(model);
});

test("all explicit cycle members, including self cycles, downgrade to sequence", () => {
  const model = layout([row("a", 0, { parent_decision_id: "b" }), row("b", 1, { parent_decision_id: "c" }),
    row("c", 2, { parent_decision_id: "a" }), row("child", 3, { parent_decision_id: "c" }),
    row("self", 4, { parent_decision_id: "self" })]);
  assert.deepEqual(links(model, "parent"), [["c", "child"]]);
  assert.deepEqual(links(model, "sequence"), [["root", "a"], ["a", "b"], ["b", "c"], ["child", "self"]]);
  valid(model);
});

test("future explicit parents survive contradictory timestamps without sequence cycles", () => {
  const model = layout([row("a", 0, { parent_decision_id: "c" }), row("b", 1), row("c", 2)]);
  assert.deepEqual(links(model, "parent"), [["c", "a"]]);
  assert.deepEqual(links(model, "sequence"), [["a", "b"], ["root", "c"]]);
  valid(model);
});

test("duplicate IDs get unique keys, preserve rows, and cannot resolve ambiguous references", () => {
  const rows = [row("dup", 0), row("dup", 1), row("child", 2,
    { parent_decision_id: "dup", source_decision_ids: ["dup"] })];
  const model = layout(rows), keys = decisions(model).map(node => node.key);
  assert.equal(new Set(keys).size, 3);
  assert.ok(!keys.includes("dup"));
  assert.deepEqual(links(model, "parent"), []);
  assert.deepEqual(links(model, "source"), []);
  assert.deepEqual(links(model, "sequence"), [["root", keys[0]], [keys[0], keys[1]], [keys[1], "child"]]);
  assert.deepEqual(keys, decisions(layout(rows)).map(node => node.key)); valid(model);
});

test("fallback keys survive reordering and mutable feedback; collisions and reserved root IDs are safe", () => {
  const a = row(undefined, 0, { mode: "train" }), b = row(undefined, 1, { mode: "evaluate" });
  const model = layout([a, b]), keys = decisions(model).map(node => node.key);
  assert.deepEqual(keys, decisions(layout([{ ...b, feedback: "changed" }, a])).map(node => node.key));
  const collision = layout([a, a, row(keys[0], 2), row("root", 3), row("child", 4, { parent_decision_id: "root" })]);
  const rootRecord = decisions(collision).find(node => node.row.decision_id === "root");
  assert.notEqual(rootRecord.key, "root");
  assert.deepEqual(links(collision, "parent"), [[rootRecord.key, "child"]]); valid(collision);
});

test("explicit sources are deduplicated references and never change tree geometry", () => {
  const rows = [row("a"), row("b", 1, { parent_decision_id: "a" }), row("c", 2, { parent_decision_id: "a" })];
  const model = layout([rows[0], rows[1], { ...rows[2], source_decision_ids: ["b", "a", "b", "c", "absent", null] }]);
  assert.deepEqual(links(model, "source"), [["b", "c"], ["a", "c"]]);
  const geometry = m => m.nodes.map(node => [node.key, node.x, node.y]);
  assert.deepEqual(geometry(model), geometry(layout(rows))); valid(model);
  assert.deepEqual(links(layout([row("a", 0, { source_decision_ids: "b" }), row("b", 1)]), "source"), []);
});

test("200 sequential nodes retain horizontal extent and branches never overlap labels", () => {
  const chain = layout(Array.from({ length: 200 }, (_, i) => row(`d${i}`, i)));
  assert.equal(chain.nodes.length, 201); assert.equal(chain.edges.length, 200);
  assert.ok(chain.width > 40000); assert.equal(chain.height, 240); valid(chain);
  const branches = layout([row("a"), ...Array.from({ length: 199 }, (_, i) =>
    row(`b${i}`, i + 1, { parent_decision_id: "a" }))]);
  assert.equal(new Set(decisions(branches).slice(1).map(node => node.y)).size, 199);
  valid(branches);
});

test("render returns SVG with intrinsic dimensions, readable labels, selection and full titles", () => {
  const api = harness(), rows = [row("b", 1), row("a")], calls = [];
  const label = (record, index) => { calls.push([record, index]); return `${index + 1} 很长的决策名称与完整上下文标签`; };
  const svg = api.render({ rows, selected: "b", label, startLabel: "开始" });
  assert.equal(svg.tagName, "svg");
  assert.ok(Number(svg.attributes.width) > 400);
  assert.equal(svg.style.minWidth, `${svg.attributes.width}px`);
  assert.equal(svg.attributes.height, "240");
  assert.equal(svg.style.overflow, undefined);
  assert.deepEqual(calls, [[rows[1], 1], [rows[0], 0]]);
  for (const node of renderedNodes(svg)) {
    assert.ok(node.attributes.class.includes("flow-choice"));
    assert.equal(node.attributes.role, "button"); assert.equal(node.attributes.tabindex, "0");
    assert.equal(tag(node, "text")[0].attributes["font-size"], "16");
    assert.ok(tag(node, "text")[0].textContent.endsWith("…"));
    assert.equal(tag(node, "title")[0].textContent, node.attributes["aria-label"]);
    assert.equal(tag(node, "circle")[0].attributes.r, "18");
    assert.equal(tag(node, "circle")[0].attributes.fill, "transparent");
  }
  const [a, b] = renderedNodes(svg);
  assert.equal(a.attributes["aria-pressed"], "false");
  assert.equal(b.attributes["aria-pressed"], "true");
  assert.equal(tag(a, "circle")[1].attributes.r, "6");
  assert.equal(tag(b, "circle")[1].attributes.r, "8");
  assert.equal(tag(b, "circle")[1].attributes.fill, "var(--orange, #b45b35)");
  const root = svg.children.find(node => node.attributes.class === "decision-root");
  assert.equal(root.attributes.transform, "translate(40 100)");
  assert.equal(root.attributes.role, undefined); assert.equal(root.attributes.tabindex, undefined);
  assert.deepEqual(root.listeners, {}); assert.equal(tag(root, "title")[0].textContent, "开始");
});

test("click, Enter and Space select with key, original row and original index", () => {
  const api = harness(), record = row("a"), calls = [];
  const node = renderedNodes(api.render({ rows: [null, record], onSelect: (...args) => calls.push(args) }))[0];
  node.listeners.click();
  let prevented = 0;
  for (const key of ["Enter", " ", "Spacebar", "Escape", "ArrowRight"]) {
    node.listeners.keydown({ key, preventDefault() { prevented++; } });
  }
  node.listeners.keydown({ key: " ", repeat: true, preventDefault() { prevented++; } });
  assert.equal(prevented, 4); assert.equal(calls.length, 4);
  for (const call of calls) assert.deepEqual(call, ["a", record, 1]);
  assert.doesNotThrow(() => renderedNodes(api.render({ rows: [record] }))[0].listeners.click());
});

test("render uses straight lines, dashed source references, and optional translated edge titles", () => {
  const api = harness(), rows = [row("a"), row("b", 1, { parent_decision_id: "a", source_decision_ids: ["a"] })];
  const titles = { parent: "明确父决策", sequence: "时间顺序（非因果）", source: "来源引用" };
  const svg = api.render({ rows, edgeLabel: kind => titles[kind] });
  const lines = tag(svg, "line"); assert.equal(lines.length, 3);
  for (const line of lines) {
    const kind = line.attributes["data-kind"];
    assert.equal(tag(line, "title")[0].textContent, titles[kind]);
    assert.equal(line.attributes["stroke-dasharray"], kind === "source" ? "4 4" : undefined);
  }
  const defaults = tag(api.render({ rows }), "line").map(line => tag(line, "title")[0].textContent);
  assert.deepEqual(defaults, ["Sequence (not causality)", "Declared parent", "Source reference"]);
  assert.deepEqual(tag(svg, "path"), []); assert.deepEqual(tag(svg, "animate"), []);
});

test("labels remain text even when markup is supplied, including empty and Unicode labels", () => {
  const api = harness();
  for (const label of ["<script>alert(1)</script>", "😀".repeat(30), "", "短标签"]) {
    const node = renderedNodes(api.render({ rows: [row("a")], label: () => label }))[0];
    assert.equal(tag(node, "title")[0].textContent, label);
    assert.equal(node.attributes["aria-label"], label);
    assert.ok(Array.from(tag(node, "text")[0].textContent).length <= 14);
    assert.equal(node.innerHTML, undefined);
  }
});
