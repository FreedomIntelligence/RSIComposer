"use strict";

/**
 * layout(rows) -> {nodes, edges, width, height}; no DOM access or input mutation.
 * Nodes: {key, index, row, x, y, type}; root has key "root", index -1, row null.
 * Edges: {from, to, kind}, where kind is parent, sequence (NOT causality), or
 * source (reference only). Only unique, nonempty string IDs resolve references.
 * Unique IDs are keys, except the reserved root ID; other keys use stable record
 * fields, with occurrence suffixes for indistinguishable records/collisions.
 * Valid timestamps sort ascending, ties keep input order; unknowns come last.
 * Explicit parent cycles fall back to sequence. If a chronological predecessor
 * would close a cycle with valid parents, its sequence edge starts at root.
 * Sources never affect layout. Root stays fixed; branch leaves have separate
 * lanes. Intrinsic SVG width allows the caller's CSS to supply scrolling.
 *
 * render({rows, selected, onSelect, label, startLabel, edgeLabel}) -> SVG element.
 * selected is a node key; label(row, originalIndex) returns its full label.
 * onSelect(key, row, originalIndex) fires for click, Enter, or Space.
 * edgeLabel(kind) optionally supplies a translated edge title.
 * Selection is controlled by the caller: rerender with the new selected key.
 * No dependencies; browser global RSIDecisions and CommonJS exports.
 */
(function (root) {
  const NS = "http://www.w3.org/2000/svg";
  const X = 40, Y = 100, COLUMN = 200, LANE = 72, LABEL_WIDTH = 160;
  const identifier = value => typeof value === "string" && value.trim() !== "";

  function timestamp(value) {
    const time = typeof value === "number" ? value
      : typeof value === "string" && value.trim() ? Date.parse(value) : NaN;
    return Number.isFinite(time) ? time : Infinity;
  }

  function fallback(row) {
    // Deliberately excludes input position and mutable feedback/status fields.
    return "decision:" + JSON.stringify([row.timestamp, row.kind, row.round,
      row.directive_id, row.mode, row.operation, row.action, row.job_id, row.reason]);
  }

  function layout(rows = []) {
    const records = (Array.isArray(rows) ? rows : []).map((row, index) => ({ row, index }))
      .filter(({ row }) => row && typeof row === "object" && !Array.isArray(row));
    const ids = new Map();
    for (const record of records) {
      const id = record.row.decision_id;
      if (identifier(id)) {
        if (!ids.has(id)) ids.set(id, []);
        ids.get(id).push(record);
      }
    }
    const used = new Set(["root", ...ids.keys()]);
    for (const record of records) {
      const id = record.row.decision_id;
      let key = id;
      if (!identifier(id) || id === "root" || ids.get(id).length !== 1) {
        const base = fallback(record.row);
        key = base;
        let occurrence = 1;
        while (used.has(key)) key = `${base}#${occurrence++}`;
        used.add(key);
      }
      record.key = key;
    }
    records.sort((a, b) => {
      const at = timestamp(a.row.timestamp), bt = timestamp(b.row.timestamp);
      return (at === bt ? 0 : at < bt ? -1 : 1) || a.index - b.index;
    });
    const rank = new Map(records.map((record, i) => [record, i]));
    const resolve = id => identifier(id) && ids.get(id)?.length === 1
      ? rank.get(ids.get(id)[0]) : -1;
    const parent = records.map(record => resolve(record.row.parent_decision_id));

    // Cut every explicit cycle member, retaining valid links to descendants.
    const state = new Uint8Array(records.length);
    for (let start = 0; start < records.length; start++) {
      if (state[start]) continue;
      const path = [];
      let current = start;
      while (current >= 0 && !state[current]) {
        state[current] = 1;
        path.push(current);
        current = parent[current];
      }
      if (current >= 0 && state[current] === 1) {
        for (let j = path.indexOf(current); j < path.length; j++) parent[path[j]] = -1;
      }
      for (const index of path) state[index] = 2;
    }
    const kinds = parent.map(p => p >= 0 ? "parent" : "sequence");
    for (let i = 0; i < records.length; i++) {
      if (parent[i] >= 0) continue;
      let previous = i - 1, ancestor = previous;
      while (ancestor >= 0 && ancestor !== i) ancestor = parent[ancestor];
      if (ancestor === i) previous = -1;
      parent[i] = previous;
    }

    const nodes = [{ key: "root", index: -1, row: null, x: X, y: Y, type: "root" },
      ...records.map(({ key, index, row }) => ({ key, index, row, x: 0, y: 0, type: "decision" }))];
    const children = nodes.map(() => []);
    for (let i = 0; i < records.length; i++) children[parent[i] + 1].push(i + 1);
    const first = [], last = [];
    let lane = 0;
    const stack = [[0, false]];
    while (stack.length) {
      const [i, exit] = stack.pop(), node = nodes[i];
      if (exit) {
        first[i] = first[children[i][0]];
        last[i] = last[children[i][children[i].length - 1]];
        if (i) node.y = Y + (first[i] + last[i]) / 2 * LANE;
      } else if (!children[i].length) {
        first[i] = last[i] = lane++;
        if (i) node.y = Y + first[i] * LANE;
      } else {
        stack.push([i, true]);
        for (let j = children[i].length - 1; j >= 0; j--) {
          const child = children[i][j];
          nodes[child].x = node.x + COLUMN;
          stack.push([child, false]);
        }
      }
    }
    const edges = [];
    records.forEach((record, i) => {
      edges.push({ from: nodes[parent[i] + 1].key, to: record.key, kind: kinds[i] });
      const sources = record.row.source_decision_ids;
      for (const id of new Set(Array.isArray(sources) ? sources : [])) {
        const source = resolve(id);
        if (source >= 0 && source !== i) {
          edges.push({ from: records[source].key, to: record.key, kind: "source" });
        }
      }
    });
    let width = X + LABEL_WIDTH + 24, height = 240;
    for (const node of nodes) {
      width = Math.max(width, node.x + LABEL_WIDTH + 24);
      height = Math.max(height, node.y + 56);
    }
    return { nodes, edges, width, height };
  }

  function render({ rows = [], selected = null, onSelect, label, startLabel = "Start", edgeLabel } = {}) {
    const model = layout(rows);
    function element(tag, attributes = {}, text) {
      const node = root.document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
      if (text !== undefined) node.textContent = String(text);
      return node;
    }
    const svg = element("svg", { class: "decision-map", width: model.width, height: model.height,
      viewBox: `0 0 ${model.width} ${model.height}`, role: "group", "aria-label": startLabel });
    // Prevent flex containers or broad svg CSS from shrinking the intrinsic map.
    svg.style.minWidth = `${model.width}px`;
    const byKey = new Map(model.nodes.map(node => [node.key, node]));
    for (const edge of model.edges) {
      const from = byKey.get(edge.from), to = byKey.get(edge.to);
      const attributes = { class: `decision-edge decision-edge-${edge.kind}`,
        "data-kind": edge.kind, x1: from.x, y1: from.y, x2: to.x, y2: to.y,
        stroke: "currentColor", "stroke-width": 1, opacity: 0.4 };
      if (edge.kind === "source") attributes["stroke-dasharray"] = "4 4";
      const line = element("line", attributes);
      const defaultTitle = edge.kind === "sequence" ? "Sequence (not causality)"
        : edge.kind === "source" ? "Source reference" : "Declared parent";
      line.append(element("title", {}, typeof edgeLabel === "function" ? edgeLabel(edge.kind) ?? defaultTitle : defaultTitle));
      svg.append(line);
    }
    for (const node of model.nodes) {
      const isRoot = node.type === "root", active = !isRoot && node.key === selected;
      const fullLabel = String(isRoot ? startLabel : (typeof label === "function"
        ? label(node.row, node.index) : node.row.operation || node.row.action || node.row.mode
          || node.row.kind || node.row.decision_id || `Decision ${node.index + 1}`) ?? "");
      const group = element("g", { class: isRoot ? "decision-root" : `decision-node flow-choice${active ? " is-selected" : ""}`,
        "data-key": node.key, transform: `translate(${node.x} ${node.y})` });
      group.append(element("title", {}, fullLabel));
      if (!isRoot) {
        group.style.cursor = "pointer";
        group.setAttribute("role", "button");
        group.setAttribute("tabindex", "0");
        group.setAttribute("aria-label", fullLabel);
        group.setAttribute("aria-pressed", String(active));
        const select = () => { if (typeof onSelect === "function") onSelect(node.key, node.row, node.index); };
        group.addEventListener("click", select);
        group.addEventListener("keydown", event => {
          if (["Enter", " ", "Spacebar"].includes(event.key)) {
            event.preventDefault();
            if (!event.repeat) select();
          }
        });
      }
      if (!isRoot) group.append(element("circle", { r: 18, fill: "transparent", "pointer-events": "all" }));
      group.append(element("circle", { r: active ? 8 : 6, fill: active ? "var(--orange, #b45b35)" : "currentColor" }));
      // Bound text by both character count and SVG textLength for wide glyphs.
      const characters = Array.from(fullLabel.replace(/\s+/g, " "));
      const brief = characters.length > 14 ? characters.slice(0, 13).join("") + "…" : characters.join("");
      const text = element("text", { x: -8, y: 32, "font-size": 16, "font-family": "sans-serif" }, brief);
      const estimated = Array.from(brief).reduce((sum, char) => sum + (char.codePointAt(0) > 255 ? 16 : 10), 0);
      if (estimated > 0) {
        text.setAttribute("textLength", String(Math.min(estimated, LABEL_WIDTH)));
        text.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      group.append(text);
      svg.append(group);
    }
    return svg;
  }

  const api = Object.freeze({ layout, render });
  root.RSIDecisions = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
