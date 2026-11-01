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
    if(!structural.has(edge.to))structural.set(edge.to,[]);
    structural.get(edge.to).push(edge.from);
  }
  for (const node of decisions(model)) {
    const visited = new Set();
    const pending=[node.key];
    while(pending.length){
      const key=pending.pop();if(key==="root"||visited.has(key))continue;visited.add(key);
      assert.ok(structural.has(key));pending.push(...structural.get(key));
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
  constructor(tag) {
    this.tagName = tag; this.children = []; this.attributes = {}; this.style = {}; this.listeners = {};
    this.isConnected = true; this.rect = { left: 0, top: 0, width: 960, height: 420 };
    this.captures = new Set();
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...children) { this.children.push(...children); for (const child of children) child.parentNode = this; }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  removeEventListener(name, listener) { if (this.listeners[name] === listener) delete this.listeners[name]; }
  setPointerCapture(id) { this.captures.add(id); }
  releasePointerCapture(id) { this.captures.delete(id); }
  getBoundingClientRect() { return this.rect; }
}
function harness(options = {}) {
  const document = { hidden: false, listeners: {}, addEventListener(name, callback) { this.listeners[name] = callback; },
    removeEventListener(name, callback) { if (this.listeners[name] === callback) delete this.listeners[name]; },
    createElementNS(namespace, tag) {
    assert.equal(namespace, "http://www.w3.org/2000/svg"); return new Element(tag);
  } };
  const context = vm.createContext({ document, ...options });
  vm.runInContext(source, context);
  return context.RSIDecisions;
}
const descendants = node => node.children.flatMap(child => [child, ...descendants(child)]);
const renderedNodes = svg => descendants(svg).filter(node => node.attributes.class?.split(" ").includes("decision-node"));
const tag = (node, name) => descendants(node).filter(child => child.tagName === name);
const circle = (node, className) => tag(node, "circle").find(c => c.attributes.class === className);

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
  assert.deepEqual(links(model, "dependency"), [["a", "b"], ["a", "c"], ["b", "d"], ["b", "e"], ["c", "f"]]);
  assert.deepEqual(links(model, "sequence"), [["root", "a"]]);
  assert.equal(get(model, "b").x, get(model, "c").x);
  assert.notEqual(get(model, "b").y, get(model, "c").y);
  assert.equal(get(model, "root").y, 100); valid(model);
});

test("diamond dependencies retain both parents, longest-path layers and centered fork/rejoin geometry", () => {
  const rows=freeze([row("a",3,{depends_on_decision_ids:[]}),row("b",2,{depends_on_decision_ids:["a"]}),row("c",1,{depends_on_decision_ids:["a"]}),row("d",0,{depends_on_decision_ids:["b","c"]})]);
  const model=layout(rows);
  assert.equal(links(model,"dependency").length,4);
  for(const edge of [["a","b"],["a","c"],["b","d"],["c","d"]])assert.ok(links(model,"dependency").some(pair=>JSON.stringify(pair)===JSON.stringify(edge)));
  assert.deepEqual(links(model,"sequence"),[["root","a"]]);
  assert.deepEqual(["a","b","c","d"].map(key=>get(model,key).x),[240,440,440,640]);
  const middle=(get(model,"b").y+get(model,"c").y)/2;
  assert.equal(get(model,"a").y,middle);assert.equal(get(model,"d").y,middle);assert.notEqual(get(model,"b").y,get(model,"c").y);
  assert.deepEqual(layout(rows),model);valid(model);
  const extended=layout([...rows,row("e",4,{depends_on_decision_ids:["a","d"]})]);
  assert.deepEqual(links(extended,"dependency").filter(([,to])=>to==="e"),[["a","e"],["d","e"]]);
  assert.equal(get(extended,"e").x,840);valid(extended);
});

test("two confirmed adoption sources both survive and take precedence over dependency and reference duplicates", () => {
  const model=layout([row("a",0,{depends_on_decision_ids:[]}),row("b",1,{depends_on_decision_ids:[]}),row("c",2,{parent_decision_id:"a",depends_on_decision_ids:["a","b","b"],adopted_decision_ids:["a","b","a"],source_decision_ids:["a","b"]})]);
  assert.deepEqual(links(model,"adoption"),[["a","c"],["b","c"]]);assert.deepEqual(links(model,"dependency"),[]);assert.deepEqual(links(model,"source"),[]);
  assert.deepEqual(links(model,"sequence"),[["root","a"],["root","b"]]);
  assert.equal(get(model,"a").x,get(model,"b").x);assert.equal(get(model,"c").x,get(model,"a").x+200);valid(model);
});

test("explicitly empty lineage keeps independent decisions parallel; only entirely legacy data chains by order", () => {
  for(const field of ["depends_on_decision_ids","adopted_decision_ids"]){
    // One modern record opts the mixed catalog into explicit lineage too.
    const model=layout([row("a",0,{[field]:[]}),row("b",1),row("c",2,{[field]:[]})]);
    assert.deepEqual(links(model,"sequence"),[["root","a"],["root","b"],["root","c"]]);
    assert.equal(new Set(decisions(model).map(n=>n.x)).size,1);assert.equal(new Set(decisions(model).map(n=>n.y)).size,3);
    assert.deepEqual(links(model,"dependency"),[]);assert.deepEqual(links(model,"adoption"),[]);valid(model);
  }
  const legacy=layout([row("a"),row("b",1),row("c",2)]);
  assert.deepEqual(links(legacy,"sequence"),[["root","a"],["a","b"],["b","c"]]);assert.deepEqual(links(legacy,"dependency"),[]);valid(legacy);
});

test("DAG cycles discard only internal structural edges and retain valid outside parents and descendants", () => {
  const rows=[row("outside",0,{depends_on_decision_ids:[]}),row("a",1,{depends_on_decision_ids:["outside","b"]}),row("b",2,{adopted_decision_ids:["a"]}),row("child",3,{depends_on_decision_ids:["b"]}),row("self",4,{adopted_decision_ids:["self"]})];
  const model=layout(rows);
  assert.deepEqual(links(model,"dependency"),[["outside","a"],["b","child"]]);assert.deepEqual(links(model,"adoption"),[]);
  assert.deepEqual(links(model,"sequence"),[["root","outside"],["root","b"],["root","self"]]);valid(model);
  const cycle=layout([row("a",0,{depends_on_decision_ids:["c"]}),row("b",1,{depends_on_decision_ids:["a"]}),row("c",2,{adopted_decision_ids:["b"]})]);
  assert.deepEqual(cycle.edges.map(e=>[e.from,e.to,e.kind]),[["root","a","sequence"],["root","b","sequence"],["root","c","sequence"]]);valid(cycle);
});

test("unknown, self, ambiguous and malformed lineage never invent parents or hide records", () => {
  const rows=[row("dup",0,{depends_on_decision_ids:[]}),row("dup",1,{depends_on_decision_ids:[]}),row("unknown",2,{depends_on_decision_ids:["missing",null,7,{},"unknown"],adopted_decision_ids:["dup","missing"],source_decision_ids:["dup","unknown","missing"]}),row("bad",3,{depends_on_decision_ids:"dup",adopted_decision_ids:{id:"dup"}})];
  const model=layout(freeze(rows));assert.equal(model.nodes.length,5);assert.equal(model.edges.length,4);
  assert.ok(model.edges.every(e=>e.from==="root"&&e.kind==="sequence"));assert.equal(new Set(model.nodes.map(n=>n.key)).size,5);
  assert.deepEqual(layout(rows),model);valid(model);
});

test("reference cycles and references to failed or later nodes do not affect dependency layers", () => {
  const base=[row("failed",0,{status:"failed",depends_on_decision_ids:[]}),row("a",1,{depends_on_decision_ids:[]}),row("b",2,{depends_on_decision_ids:["a"]})];
  const model=layout([{...base[0],source_decision_ids:["b"]},{...base[1],source_decision_ids:["failed","b"]},{...base[2],source_decision_ids:["failed"]}]);
  assert.deepEqual(links(model,"source"),[["b","failed"],["failed","a"],["b","a"],["failed","b"]]);
  const geometry=m=>m.nodes.map(n=>[n.key,n.x,n.y]);assert.deepEqual(geometry(model),geometry(layout(base)));valid(model);
});

function sourceClear(model,positions=new Map(model.nodes.map(node=>[node.key,node]))) {
  for(const edge of model.edges.filter(e=>e.kind==="source")){
    const a=positions.get(edge.from),b=positions.get(edge.to),dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy;
    assert.ok(length>0);
    for(const [key,p] of positions){
      if(key===edge.from||key===edge.to)continue;
      const t=((p.x-a.x)*dx+(p.y-a.y)*dy)/length;
      if(t>0&&t<1)assert.ok(Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy)>=18,`${edge.from} -> ${edge.to} crosses ${key}`);
    }
  }
}

test("source-only fan-in and fan-out separate straight references without moving dependency layers",()=>{
  const base=[row("a",0,{depends_on_decision_ids:[],status:"failed"}),row("b",1,{depends_on_decision_ids:[]}),row("c",2,{depends_on_decision_ids:[]}),row("parent",3,{depends_on_decision_ids:[]}),row("child",4,{depends_on_decision_ids:["parent"]})];
  const original=layout(base);
  for(const refs of [{b:["a"],c:["a"]},{c:["a","b"]}]){
    const rows=base.map(record=>({...record,...(refs[record.decision_id]?{source_decision_ids:refs[record.decision_id]}:{})})),model=layout(freeze(rows));
    assert.equal(links(model,"source").length,2);assert.deepEqual(links(model,"dependency"),[["parent","child"]]);
    for(const key of ["root","parent","child"])assert.deepEqual([get(model,key).x,get(model,key).y],[get(original,key).x,get(original,key).y]);
    assert.equal(get(model,"child").x-get(model,"parent").x,200);
    sourceClear(model);valid(model);assert.deepEqual(layout(rows),model);
  }
});

test("missing or malformed parents attach to root without fabricated references", () => {
  const model = layout([row("a", 0, { parent_decision_id: "absent" }),
    row("b", 1, { parent_decision_id: "other" }), row("c", 2, { parent_decision_id: 8 }),
    row("d", 3, { parent_decision_id: "  " })]);
  assert.deepEqual(links(model, "dependency"), []);
  assert.deepEqual(links(model, "sequence"), [["root", "a"], ["root", "b"], ["root", "c"], ["root", "d"]]);
  valid(model);
});

test("all explicit cycle members, including self cycles, attach to root while descendants survive", () => {
  const model = layout([row("a", 0, { parent_decision_id: "b" }), row("b", 1, { parent_decision_id: "c" }),
    row("c", 2, { parent_decision_id: "a" }), row("child", 3, { parent_decision_id: "c" }),
    row("self", 4, { parent_decision_id: "self" })]);
  assert.deepEqual(links(model, "dependency"), [["c", "child"]]);
  assert.deepEqual(links(model, "sequence"), [["root", "a"], ["root", "b"], ["root", "c"], ["root", "self"]]);
  valid(model);
});

test("future explicit parents survive contradictory timestamps without sequence cycles", () => {
  const model = layout([row("a", 0, { parent_decision_id: "c" }), row("b", 1), row("c", 2)]);
  assert.deepEqual(links(model, "dependency"), [["c", "a"]]);
  assert.deepEqual(links(model, "sequence"), [["a", "b"], ["root", "c"]]);
  valid(model);
});

test("duplicate IDs get unique keys, preserve rows, and cannot resolve ambiguous references", () => {
  const rows = [row("dup", 0), row("dup", 1), row("child", 2,
    { parent_decision_id: "dup", source_decision_ids: ["dup"] })];
  const model = layout(rows), keys = decisions(model).map(node => node.key);
  assert.equal(new Set(keys).size, 3);
  assert.ok(!keys.includes("dup"));
  assert.deepEqual(links(model, "dependency"), []);
  assert.deepEqual(links(model, "source"), []);
  assert.deepEqual(links(model, "sequence"), [["root", keys[0]], [keys[0], keys[1]], ["root", "child"]]);
  assert.deepEqual(keys, decisions(layout(rows)).map(node => node.key)); valid(model);
});

test("fallback keys survive reordering and mutable feedback; collisions and reserved root IDs are safe", () => {
  const a = row(undefined, 0, { mode: "train" }), b = row(undefined, 1, { mode: "evaluate" });
  const model = layout([a, b]), keys = decisions(model).map(node => node.key);
  assert.deepEqual(keys, decisions(layout([{ ...b, feedback: "changed" }, a])).map(node => node.key));
  const collision = layout([a, a, row(keys[0], 2), row("root", 3), row("child", 4, { parent_decision_id: "root" })]);
  const rootRecord = decisions(collision).find(node => node.row.decision_id === "root");
  assert.notEqual(rootRecord.key, "root");
  assert.deepEqual(links(collision, "dependency"), [[rootRecord.key, "child"]]); valid(collision);
});

test("explicit sources are deduplicated references and never change tree geometry", () => {
  const rows = [row("a"), row("b", 1, { parent_decision_id: "a" }), row("c", 2, { parent_decision_id: "a" })];
  const model = layout([rows[0], rows[1], { ...rows[2], source_decision_ids: ["b", "a", "b", "c", "absent", null] }]);
  assert.deepEqual(links(model, "source"), [["b", "c"]]);
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

test("render returns responsive 420px SVG, readable labels, selection halo and full titles", () => {
  const api = harness(), rows = [row("b", 1), row("a")], calls = [];
  const label = (record, index) => { calls.push([record, index]); return `${index + 1} 很长的决策名称与完整上下文标签`; };
  const svg = api.render({ rows, selected: "b", label, startLabel: "开始" });
  assert.equal(svg.tagName, "svg");
  assert.equal(svg.attributes.width, "100%");
  assert.equal(svg.style.width, "100%");
  assert.equal(svg.attributes.height, "420");
  assert.equal(svg.attributes.viewBox, "0 0 960 420");
  assert.equal(svg.style.overflow, undefined);
  assert.deepEqual(calls, [[rows[1], 1], [rows[0], 0]]);
  for (const node of renderedNodes(svg)) {
    assert.ok(node.attributes.class.includes("flow-choice"));
    assert.equal(node.attributes.role, "button"); assert.equal(node.attributes.tabindex, "0");
    assert.equal(tag(node, "text")[0].attributes["font-size"], "16");
    assert.equal(tag(node, "text")[0].attributes["text-anchor"], "middle");
    assert.equal(tag(node, "text")[0].attributes.x, "0");
    assert.equal(tag(tag(node, "text")[0], "tspan").length, 2);
    assert.equal(tag(node, "text")[0].attributes.textLength, undefined);
    assert.equal(tag(node, "title")[0].textContent, node.attributes["aria-label"]);
    assert.equal(tag(node, "circle")[0].attributes.r, "18");
    assert.equal(tag(node, "circle")[0].attributes.fill, "transparent");
  }
  const [a, b] = renderedNodes(svg);
  assert.equal(a.attributes["aria-pressed"], "false");
  assert.equal(b.attributes["aria-pressed"], "true");
  assert.equal(circle(a, "decision-dot").attributes.r, "6");
  assert.equal(circle(b, "decision-dot").attributes.r, "8");
  assert.equal(circle(b, "decision-dot").attributes.fill, "var(--orange, #b45b35)");
  assert.equal(circle(b, "decision-halo").attributes.r, "13");
  const root = descendants(svg).find(node => node.attributes.class === "decision-root");
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
  const api = harness(), rows = [row("a"), row("b", 1, { parent_decision_id: "a" }),row("c",2,{depends_on_decision_ids:["a"],adopted_decision_ids:["b"],source_decision_ids:["b"]}),row("d",3,{source_decision_ids:["a"]})];
  const titles = { dependency: "明确依赖", adoption:"确认采纳", sequence: "时间顺序（非因果）", source: "来源引用" };
  const svg = api.render({ rows, edgeLabel: kind => titles[kind] });
  const lines = tag(svg, "line"); assert.equal(lines.length, 6);
  for (const line of lines) {
    const kind = line.attributes["data-kind"];
    assert.equal(tag(line, "title")[0].textContent, titles[kind]);
    assert.equal(line.attributes["stroke-dasharray"], kind === "source" ? "4 4" : undefined);
    assert.equal(line.attributes.stroke,kind==="adoption"?"var(--orange, #b45b35)":"var(--muted, #8a7568)");
    if(kind==="dependency")assert.equal(line.attributes["stroke-width"],"1");
  }
  assert.ok(Number(lines.find(l=>l.attributes["data-kind"]==="sequence").attributes.opacity)<Number(lines.find(l=>l.attributes["data-kind"]==="dependency").attributes.opacity));
  const defaults = tag(api.render({ rows }), "line").map(line => tag(line, "title")[0].textContent);
  assert.deepEqual(defaults, ["Sequence (not causality)", "Declared dependency", "Declared dependency", "Confirmed adoption", "Sequence (not causality)", "Source reference (not dependency or adoption)"]);
  assert.deepEqual(tag(svg, "path"), []); assert.deepEqual(tag(svg, "animate"), []);
});

test("labels remain text even when markup is supplied, including empty and Unicode labels", () => {
  const api = harness();
  for (const label of ["<script>alert(1)</script>", "😀".repeat(30), "", "短标签"]) {
    const node = renderedNodes(api.render({ rows: [row("a")], label: () => label }))[0];
    assert.equal(tag(node, "title")[0].textContent, label);
    assert.equal(node.attributes["aria-label"], label);
    const text = tag(node, "text")[0];
    assert.ok(text.children.length <= 2);
    assert.equal(text.attributes.textLength, undefined);
    assert.equal(text.attributes.lengthAdjust, undefined);
    assert.equal(node.innerHTML, undefined);
  }
});

function interactive({ reduced = false, raf = true } = {}) {
  const frames = new Map(), observers = [], canceled = [], listeners = {};
  let next = 0;
  const document = { hidden: false, listeners, createElementNS: (_, name) => new Element(name),
    addEventListener(type, callback) { listeners[type] = callback; },
    removeEventListener(type, callback) { if (listeners[type] === callback) delete listeners[type]; } };
  class ResizeObserver {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { this.disconnected = true; }
  }
  const api = harness({ document, ResizeObserver, matchMedia: () => ({ matches: reduced }),
    ...(raf ? { requestAnimationFrame(callback) { frames.set(++next, callback); return next; },
      cancelAnimationFrame(id) { canceled.push(id); frames.delete(id); } } : {}) });
  function step() {
    const pending = [...frames]; frames.clear();
    for (const [, callback] of pending) callback();
  }
  function drain() {
    let count = 0;
    while (frames.size && count++ < 100) step();
    assert.equal(frames.size, 0, "RAF must have a finite budget");
    return count;
  }
  return { api, document, observers, frames, canceled, step, drain };
}
function pointer(target, type, x, y, extra = {}) {
  const event = { type, target, clientX: x, clientY: y, pointerId: 1, button: 0,
    preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...extra };
  target.listeners[type]?.(event);
  return event;
}
const coordinates = p => [p.x, p.y];
function startDrag(svg, view, index = 0) {
  const node = renderedNodes(svg)[index], key = node.attributes["data-key"], position = view.positions.get(key);
  const x = view.camera.x + position.x * view.camera.scale, y = view.camera.y + position.y * view.camera.scale;
  pointer(node, "pointerdown", x, y);
  return { node, key, x, y };
}

function renderedDAG(svg,view,rows) {
  const model=layout(rows),lines=tag(svg,"line");assert.equal(lines.length,model.edges.length);
  model.edges.forEach((edge,i)=>{
    const from=view.positions.get(edge.from),to=view.positions.get(edge.to),line=lines[i];
    assert.equal(line.attributes["data-kind"],edge.kind);
    assert.deepEqual(["x1","y1","x2","y2"].map(key=>Number(line.attributes[key])),[from.x,from.y,to.x,to.y]);
    if(edge.kind!=="source")assert.ok(to.x-from.x>=60-1e-8,`${edge.from} -> ${edge.to} must keep every parent constraint`);
  });
  assert.deepEqual(coordinates(view.positions.get("root")),[40,100]);
}

test("diamond drag and settling preserve every dependency/adoption edge, root and all parent constraints", () => {
  const rows=[row("a",0,{depends_on_decision_ids:[],source_decision_ids:["join"]}),row("b",1,{depends_on_decision_ids:["a"]}),row("c",2,{depends_on_decision_ids:["a"]}),row("join",3,{depends_on_decision_ids:["b","c"],adopted_decision_ids:["b","c"]}),row("end",4,{depends_on_decision_ids:["a","join"]})];
  const h=interactive(),view={};const svg=h.api.render({rows,view});
  assert.deepEqual(links(layout(rows),"adoption"),[["b","join"],["c","join"]]);renderedDAG(svg,view,rows);
  for(const [key,dx,dy] of [["b",5000,120],["c",-5000,-90],["join",-5000,180],["end",-5000,30]]){
    const index=renderedNodes(svg).findIndex(n=>n.attributes["data-key"]===key),d=startDrag(svg,view,index);
    pointer(d.node,"pointermove",d.x+dx,d.y+dy);renderedDAG(svg,view,rows);
    for(let i=0;i<3;i++){h.step();renderedDAG(svg,view,rows);}
    pointer(d.node,"pointerup",d.x+dx,d.y+dy);h.drain();renderedDAG(svg,view,rows);
  }
  svg.zoomBy(1.5);pointer(svg,"pointerdown",100,100);pointer(svg,"pointermove",-600,800);pointer(svg,"pointerup",-600,800);
  const camera={...view.camera},positions=[...view.positions].map(([key,p])=>[key,...coordinates(p)]);svg.dispose();
  const restored=h.api.render({rows,view});assert.deepEqual({...view.camera},camera);assert.deepEqual([...view.positions].map(([key,p])=>[key,...coordinates(p)]),positions);renderedDAG(restored,view,rows);
  restored.fit();renderedDAG(restored,view,rows);restored.reset();renderedDAG(restored,view,rows);restored.dispose();
});

test("new second-parent evidence repairs cached horizontal constraints without losing manual placement or camera", () => {
  const rows=[row("a",0,{depends_on_decision_ids:[]}),row("b",1,{depends_on_decision_ids:[]}),row("join",2,{depends_on_decision_ids:["a"]})];
  const h=interactive(),view={};const svg=h.api.render({rows,view});
  const d=startDrag(svg,view,1);pointer(d.node,"pointermove",d.x+900,d.y+80);pointer(d.node,"pointerup",d.x+900,d.y+80);h.drain();svg.zoomBy(1.25);
  const b=coordinates(view.positions.get("b")),joinY=view.positions.get("join").y,camera={...view.camera};assert.ok(b[0]>view.positions.get("join").x);svg.dispose();
  const updated=[rows[0],rows[1],{...rows[2],depends_on_decision_ids:["a","b"]}];
  const restored=h.api.render({rows:updated,view});
  assert.equal(links(layout(updated),"dependency").length,2);assert.deepEqual(coordinates(view.positions.get("b")),b);assert.equal(view.positions.get("join").y,joinY);assert.deepEqual({...view.camera},camera);assert.ok(view.pinned.has("b"));renderedDAG(restored,view,updated);restored.dispose();
});

test("new source references separate cached parallel nodes with straight lines and preserve camera and pinned placement",()=>{
  const base=[row("a",0,{depends_on_decision_ids:[]}),row("b",1,{depends_on_decision_ids:[]}),row("c",2,{depends_on_decision_ids:[]})];
  for(const pinned of [false,true]){
    const h=interactive(),view={},initial=h.api.render({rows:base,view});initial.zoomBy(1.25);
    const camera={...view.camera},b=coordinates(view.positions.get("b"));if(pinned)view.pinned.add("b");initial.dispose();
    const rows=[base[0],base[1],{...base[2],source_decision_ids:["a","b"]}],svg=h.api.render({rows,view});
    assert.deepEqual({...view.camera},camera);if(pinned){assert.ok(view.pinned.has("b"));assert.deepEqual(coordinates(view.positions.get("b")),b);}
    assert.equal(tag(svg,"path").length,0);assert.equal(tag(svg,"line").filter(line=>line.attributes["data-kind"]==="source").length,2);
    sourceClear(layout(rows),view.positions);renderedDAG(svg,view,rows);
    const saved=[...view.positions].map(([key,p])=>[key,...coordinates(p)]);svg.dispose();
    const restored=h.api.render({rows,view});assert.deepEqual([...view.positions].map(([key,p])=>[key,...coordinates(p)]),saved);assert.deepEqual({...view.camera},camera);sourceClear(layout(rows),view.positions);restored.dispose();
  }
});

test("default graph has vertical variation and readable scale; explicit fit/reset are synchronous", () => {
  const h = interactive(), view = {}, rows = Array.from({ length: 200 }, (_, i) => row(`d${i}`, i));
  const svg = h.api.render({ rows, view });
  assert.equal(h.frames.size, 0); assert.ok(view.camera.scale >= 0.85);
  assert.ok(new Set([...view.positions.values()].map(p => p.y)).size > 1);
  const keys = new Set(layout(rows).edges.map(e => JSON.stringify([e.from, e.to, e.kind])));
  assert.equal(tag(svg, "line").length, keys.size);
  svg.fit(); assert.ok(view.camera.scale < 0.1);
  const fitScale = view.camera.scale;
  h.observers[0].callback(); assert.equal(view.camera.scale, fitScale);
  for (const p of view.positions.values()) {
    assert.ok((p.x - 96) * view.camera.scale + view.camera.x >= 31.9);
    assert.ok((p.x + 96) * view.camera.scale + view.camera.x <= 928.1);
  }
  svg.reset(); assert.ok(view.camera.scale >= 0.85); assert.equal(h.frames.size, 0);
  svg.dispose();
});

test("node dragging elastically moves neighbors, fixes root, persists manual placement and suppresses click", () => {
  const h = interactive(), view = {}, calls = []; let idle = 0;
  const rows = [row("a"), row("b", 1), row("c", 2)];
  const svg = h.api.render({ rows, view, onSelect: key => calls.push(key), onIdle: () => idle++ });
  const root = coordinates(view.positions.get("root")), before = coordinates(view.positions.get("b"));
  const d = startDrag(svg, view);
  assert.equal(view.interacting, true);
  pointer(d.node, "pointermove", d.x + 40, d.y + 150);
  assert.equal(view.animating, true); assert.equal(h.frames.size, 1);
  const placed = coordinates(view.positions.get("a"));
  h.step();
  assert.notDeepEqual(coordinates(view.positions.get("b")), before);
  assert.deepEqual(coordinates(view.positions.get("root")), root);
  pointer(d.node, "pointerup", d.x + 40, d.y + 150);
  assert.equal(view.interacting, false); assert.equal(view.animating, true);
  assert.equal(d.node.captures.size, 0); assert.equal(idle, 0);
  d.node.listeners.click({ detail: 1 }); assert.deepEqual(calls, []);
  assert.equal(h.drain(), 18); assert.equal(idle, 1); assert.equal(view.animating, false);
  assert.deepEqual(coordinates(view.positions.get("a")), placed);
  assert.deepEqual(coordinates(view.positions.get("root")), root);
  d.node.listeners.keydown({ key: "Enter", preventDefault() {} }); assert.deepEqual(calls, ["a"]);
  const savedCamera = { ...view.camera }, savedPositions = [...view.positions].map(([key, p]) => [key, ...coordinates(p)]);
  svg.dispose(); assert.equal(idle, 1);
  const restored = h.api.render({ rows, view, label: () => "translated" });
  assert.deepEqual({ ...view.camera }, savedCamera);
  assert.deepEqual([...view.positions].map(([key, p]) => [key, ...coordinates(p)]), savedPositions);
  assert.equal(h.frames.size, 0);
  restored.reset(); assert.notDeepEqual(coordinates(view.positions.get("a")), placed);
  restored.dispose();
});

test("tiny pointer movements still click; pointercancel stops animation without accidental selection", () => {
  const h = interactive(), view = {}, calls = []; let idle = 0;
  const svg = h.api.render({ rows: [row("a"), row("b", 1)], view,
    onSelect: key => calls.push(key), onIdle: () => idle++ });
  let d = startDrag(svg, view);
  const original = coordinates(view.positions.get("a"));
  pointer(d.node, "pointermove", d.x + 2, d.y + 1);
  pointer(d.node, "pointerup", d.x + 2, d.y + 1);
  assert.deepEqual(coordinates(view.positions.get("a")), original);
  d.node.listeners.click({ detail: 1 }); assert.deepEqual(calls, ["a"]);
  d = startDrag(svg, view);
  pointer(d.node, "pointermove", d.x, d.y + 50);
  pointer(d.node, "pointercancel", d.x, d.y + 50);
  assert.equal(h.frames.size, 0); assert.equal(view.interacting, false); assert.equal(view.animating, false);
  d.node.listeners.click({ detail: 1 }); assert.deepEqual(calls, ["a"]);
  assert.equal(idle, 2); svg.dispose();
});

test("background pan is unlimited in both directions and restored cameras remain run-local", () => {
  const h = interactive(), view = { camera: { x: 11, y: -23, scale: 1.5 } };
  const svg = h.api.render({ rows: [row("a")], view });
  pointer(svg, "pointerdown", 400, 200); pointer(svg, "pointermove", -1600, -1800);
  pointer(svg, "pointerup", -1600, -1800);
  assert.deepEqual({ ...view.camera }, { x: -1989, y: -2023, scale: 1.5 });
  pointer(svg, "pointerdown", 100, 100); pointer(svg, "pointermove", 4100, 4100);
  pointer(svg, "pointerup", 4100, 4100);
  assert.deepEqual({ ...view.camera }, { x: 2011, y: 1977, scale: 1.5 });
  assert.equal(view.interacting, false); assert.equal(h.frames.size, 0);
  const saved = { ...view.camera }; svg.dispose();
  const restored = h.api.render({ rows: [row("a")], view });
  assert.deepEqual({ ...view.camera }, saved);
  const otherView = {}; const other = h.api.render({ rows: [row("a")], view: otherView });
  assert.notDeepEqual({ ...otherView.camera }, saved);
  restored.dispose(); other.dispose();
});

test("same-size rerenders preserve an auto-fit camera when new decisions arrive", () => {
  const h = interactive(), view = {};
  const initial = h.api.render({ rows: [row("a")], view });
  assert.equal(view.autoFit, true); assert.equal(view.viewportWidth, 960);
  const camera = { ...view.camera }, a = coordinates(view.positions.get("a"));
  initial.dispose();
  const updated = h.api.render({ rows: [row("a"), ...Array.from({ length: 30 }, (_, i) => row(`d${i}`, i + 1))], view });
  assert.deepEqual({ ...view.camera }, camera);
  assert.deepEqual(coordinates(view.positions.get("a")), a);
  h.observers[1].callback(); assert.deepEqual({ ...view.camera }, camera);
  updated.rect.width = 720; h.observers[1].callback();
  assert.equal(view.viewportWidth, 720); assert.notDeepEqual({ ...view.camera }, camera);
  updated.dispose();
});

test("Ctrl/Meta wheel zooms at pointer while ordinary wheel retains native scrolling", () => {
  const h = interactive(), view = { camera: { x: 0, y: 0, scale: 1 } };
  const svg = h.api.render({ rows: [row("a")], view });
  const before = { ...view.camera };
  const ordinary = pointer(svg, "wheel", 200, 150, { deltaY: 100 });
  assert.equal(ordinary.prevented, undefined); assert.deepEqual({ ...view.camera }, before);
  for (const modifier of ["ctrlKey", "metaKey"]) {
    const world = [(200 - view.camera.x) / view.camera.scale, (150 - view.camera.y) / view.camera.scale];
    const event = pointer(svg, "wheel", 200, 150, { deltaY: -80, [modifier]: true });
    assert.equal(event.prevented, true);
    assert.ok(Math.abs((200 - view.camera.x) / view.camera.scale - world[0]) < 1e-8);
    assert.ok(Math.abs((150 - view.camera.y) / view.camera.scale - world[1]) < 1e-8);
  }
  const scale = view.camera.scale; svg.zoomBy(2); assert.equal(view.camera.scale, scale * 2);
  const valid = { ...view.camera }; svg.zoomBy(NaN); svg.zoomBy(-1);
  assert.deepEqual({ ...view.camera }, valid); svg.dispose();
});

test("resize and CTM conversion preserve world drag distances and a manual camera", () => {
  const h = interactive(), view = { camera: { x: 50, y: 20, scale: 2 } };
  const svg = h.api.render({ rows: [row("a"), row("b", 1)], view });
  svg.rect = { left: 100, top: 200, width: 480, height: 210 };
  // Screen CTM includes a transformed ancestor; use its inverse rather than rect alone.
  let inversions = 0;
  svg.getScreenCTM = () => ({ inverse() { inversions++; return {}; } });
  svg.createSVGPoint = () => ({ x: 0, y: 0,
    matrixTransform() { return { x: (this.x - 100) * 2, y: (this.y - 200) * 2 }; } });
  const node = renderedNodes(svg)[0], before = coordinates(view.positions.get("a"));
  pointer(node, "pointerdown", 300, 300); pointer(node, "pointermove", 310, 330);
  assert.deepEqual(coordinates(view.positions.get("a")), [before[0] + 10, before[1] + 30]);
  assert.equal(inversions, 2); pointer(node, "pointerup", 310, 330); h.drain();
  const camera = { ...view.camera };
  svg.rect.width = 720; h.observers[0].callback();
  assert.equal(svg.attributes.viewBox, "0 0 720 420");
  assert.deepEqual({ ...view.camera }, camera); svg.dispose();
});

test("dispose cancels RAF, observer, capture and all listeners without onIdle or state loss", () => {
  const h = interactive(), view = {}; let idle = 0;
  const svg = h.api.render({ rows: [row("a"), row("b", 1)], view, onIdle: () => idle++ });
  const d = startDrag(svg, view); pointer(d.node, "pointermove", d.x, d.y + 80);
  const callback = [...h.frames.values()][0], positions = [...view.positions].map(([k, p]) => [k, ...coordinates(p)]);
  svg.dispose(); svg.dispose(); callback();
  assert.equal(idle, 0); assert.equal(h.frames.size, 0); assert.ok(h.canceled.length > 0);
  assert.equal(view.interacting, false); assert.equal(view.animating, false);
  assert.equal(h.observers[0].disconnected, true); assert.equal(d.node.captures.size, 0);
  for (const node of [svg, ...descendants(svg)]) assert.deepEqual(node.listeners, {});
  assert.deepEqual(h.document.listeners, {});
  assert.deepEqual([...view.positions].map(([k, p]) => [k, ...coordinates(p)]), positions);
  const camera = { ...view.camera }; svg.fit(); svg.zoomBy(2); svg.reset();
  assert.deepEqual({ ...view.camera }, camera);
});

test("hidden or detached SVG stops RAF; reduced motion and missing RAF still permit dragging", () => {
  for (const reason of ["hidden", "detached"]) {
    const h = interactive(), view = {}; let idle = 0;
    const svg = h.api.render({ rows: [row("a"), row("b", 1)], view, onIdle: () => idle++ });
    const d = startDrag(svg, view); pointer(d.node, "pointermove", d.x, d.y + 80);
    if (reason === "hidden") { h.document.hidden = true; h.document.listeners.visibilitychange(); }
    else { svg.isConnected = false; h.step(); }
    assert.equal(h.frames.size, 0); assert.equal(view.animating, false); assert.equal(view.interacting, false);
    assert.equal(h.observers[0].disconnected, true); assert.equal(idle, 0);
  }
  for (const options of [{ reduced: true }, { raf: false }]) {
    const h = interactive(options), view = {};
    const svg = h.api.render({ rows: [row("a"), row("b", 1)], view });
    const d = startDrag(svg, view), before = coordinates(view.positions.get("a"));
    pointer(d.node, "pointermove", d.x, d.y + 80); pointer(d.node, "pointerup", d.x, d.y + 80);
    assert.notDeepEqual(coordinates(view.positions.get("a")), before);
    assert.deepEqual(coordinates(view.positions.get("root")), [40, 100]);
    assert.equal(h.frames.size, 0); assert.equal(view.animating, false); svg.dispose();
  }
});
