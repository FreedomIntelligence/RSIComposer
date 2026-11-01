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
 * lanes. This pure layout is independent of the interactive renderer.
 *
 * render({rows, selected, onSelect, label, startLabel, edgeLabel, view, onIdle})
 * returns a responsive 420px SVG with dispose(), fit(), zoomBy(factor), reset().
 * view is caller-owned per run: positions Map<key,{x,y}>, camera {x,y,scale},
 * pinned Set<key>, interacting/animating booleans. Reuse it across rerenders.
 * Dragged nodes stay pinned until reset. Only layout edges are ever drawn.
 * onIdle() follows interaction/settling; dispose() NEVER calls it.
 * No idle RAF/timer: a drag starts at most 24 frames, release at most 18 frames.
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

  function render({ rows = [], selected = null, onSelect, label, startLabel = "Start", edgeLabel,
    view = {}, onIdle } = {}) {
    const model = layout(rows), HEIGHT = 420, document = root.document;
    const finitePoint = p => p && Number.isFinite(p.x) && Number.isFinite(p.y);
    if (!view.positions?.get || !view.positions?.set) view.positions = new Map();
    if (!view.pinned?.has || !view.pinned?.add) view.pinned = new Set();
    const positions = view.positions, pinned = view.pinned;
    const restoredCamera = finitePoint(view.camera) && Number.isFinite(view.camera.scale) && view.camera.scale > 0;
    if (!restoredCamera) view.camera = { x: 0, y: 0, scale: 1 };
    if (view.autoFit === undefined) view.autoFit = !restoredCamera;
    view.interacting = view.animating = false;
    let width = Number.isFinite(view.viewportWidth) && view.viewportWidth > 0 ? view.viewportWidth : 960;
    let viewportReady = restoredCamera, disposed = false, frame = null, heat = 0, gesture = null, suppressed = null;
    let observer = null;
    const removers = [], groups = new Map(), lines = [];
    const reducedMotion = () => root.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
    function element(tag, attributes = {}, text) {
      const node = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
      if (text !== undefined) node.textContent = String(text);
      return node;
    }
    function listen(target, type, callback, options) {
      if (!target?.addEventListener) return;
      target.addEventListener(type, callback, options);
      removers.push(() => target.removeEventListener?.(type, callback, options));
    }
    const svg = element("svg", { class: "decision-map", width: "100%", height: HEIGHT,
      viewBox: `0 0 ${width} ${HEIGHT}`, preserveAspectRatio: "none", role: "group", "aria-label": startLabel });
    svg.style.width = "100%"; svg.style.height = `${HEIGHT}px`;
    svg.style.touchAction = "none";
    const plane = element("g", { class: "decision-plane" }); svg.append(plane);
    const hash = key => {
      let value = 2166136261;
      for (let i = 0; i < key.length; i++) value = Math.imul(value ^ key.charCodeAt(i), 16777619);
      return (value >>> 0) / 4294967296;
    };
    for (const node of model.nodes) {
      const depth = Math.round((node.x - X) / COLUMN);
      node.home = { x: node.x, y: node.type === "root" ? Y
        : Y + (node.y - Y) * 1.8 + (depth % 2 ? -64 : 64) + hash(node.key) * 20 - 10 };
      const saved = positions.get(node.key);
      const point = node.type !== "root" && finitePoint(saved) ? saved : node.home;
      node.x = point.x; node.y = point.y;
      node.anchor = { x: node.x, y: node.y }; node.vx = node.vy = 0;
    }
    const byKey = new Map(model.nodes.map(node => [node.key, node]));
    const structural = [];
    for (const edge of model.edges) {
      const from = byKey.get(edge.from), to = byKey.get(edge.to);
      const attributes = { class: `decision-edge decision-edge-${edge.kind}`, "data-kind": edge.kind,
        stroke: "var(--muted, #8a7568)", "stroke-width": 1, "vector-effect": "non-scaling-stroke", opacity: 0.45 };
      if (edge.kind === "source") attributes["stroke-dasharray"] = "4 4";
      const line = element("line", attributes);
      const defaultTitle = edge.kind === "sequence" ? "Sequence (not causality)"
        : edge.kind === "source" ? "Source reference" : "Declared parent";
      line.append(element("title", {}, typeof edgeLabel === "function" ? edgeLabel(edge.kind) ?? defaultTitle : defaultTitle));
      plane.append(line); lines.push({ line, from, to });
      if (edge.kind !== "source") structural.push({ from, to, dx: to.x - from.x, dy: to.y - from.y });
    }
    function notifyIdle() {
      if (!disposed && !view.interacting && !view.animating && typeof onIdle === "function") onIdle();
    }
    function cache() {
      for (const node of model.nodes) positions.set(node.key, { x: node.x, y: node.y });
    }
    function paint() {
      for (const node of model.nodes) groups.get(node.key)?.setAttribute("transform", `translate(${node.x} ${node.y})`);
      for (const { line, from, to } of lines) {
        for (const [name, value] of Object.entries({ x1: from.x, y1: from.y, x2: to.x, y2: to.y })) line.setAttribute(name, String(value));
      }
      const camera = view.camera;
      plane.setAttribute("transform", `translate(${camera.x} ${camera.y}) scale(${camera.scale})`);
      cache();
    }
    function stopAnimation() {
      if (frame !== null) root.cancelAnimationFrame?.(frame);
      frame = null; heat = 0; view.animating = false;
      for (const node of model.nodes) node.vx = node.vy = 0;
    }
    function release() {
      const previous = gesture; gesture = null; view.interacting = false;
      if (previous?.element.releasePointerCapture && previous.pointerId !== undefined) {
        try { previous.element.releasePointerCapture(previous.pointerId); } catch { /* Capture may already be lost. */ }
      }
    }
    function cancelInteraction() {
      if (gesture?.moved) suppressed = gesture.node?.key ?? null;
      release(); stopAnimation(); cache();
    }
    function simulate() {
      for (const n of model.nodes) {
        n.fx = (n.anchor.x - n.x) * 0.035; n.fy = (n.anchor.y - n.y) * 0.018;
      }
      for (const { from, to, dx, dy } of structural) {
        const fx = (to.x - from.x - dx) * 0.035, fy = (to.y - from.y - dy) * 0.035;
        from.fx += fx; from.fy += fy; to.fx -= fx; to.fy -= fy;
      }
      // Local label separation, bounded to nearby grid cells rather than all pairs.
      const grid = new Map();
      for (const n of model.nodes) {
        const gx = Math.floor(n.x / 200), gy = Math.floor(n.y / 200);
        for (let x = gx - 1; x <= gx + 1; x++) for (let y = gy - 1; y <= gy + 1; y++) {
          for (const other of grid.get(`${x},${y}`) || []) {
            const dx = n.x - other.x, dy = n.y - other.y;
            if (Math.abs(dx) < 180 && Math.abs(dy) < 64) {
              const force = (1 - Math.abs(dy) / 64) * 2.5 * (dy < 0 ? -1 : 1);
              n.fy += force; other.fy -= force;
            }
          }
        }
        const key = `${gx},${gy}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(n);
      }
      for (const n of model.nodes) {
        if (n.type === "root" || pinned.has(n.key) || gesture?.node === n) continue;
        n.vx = (n.vx + n.fx) * 0.72; n.vy = (n.vy + n.fy) * 0.72;
        n.x += Math.max(-10, Math.min(10, n.vx)); n.y += Math.max(-12, Math.min(12, n.vy));
        constrain(n);
      }
      paint();
    }
    function constrain(node) {
      let left = X + 60, right = Infinity;
      for (const edge of structural) {
        if (edge.to === node) left = Math.max(left, edge.from.x + 60);
        if (edge.from === node) right = Math.min(right, edge.to.x - 60);
      }
      if (left <= right) node.x = Math.max(left, Math.min(right, node.x));
    }
    function tick() {
      frame = null;
      if (disposed) return;
      if (document.hidden || svg.isConnected === false) { svg.dispose(); return; }
      if (reducedMotion()) { stopAnimation(); notifyIdle(); return; }
      simulate();
      if (--heat > 0) frame = root.requestAnimationFrame(tick);
      else { view.animating = false; notifyIdle(); }
    }
    function reheat(frames = 24) {
      if (disposed || document.hidden || svg.isConnected === false) return;
      if (reducedMotion() || typeof root.requestAnimationFrame !== "function") {
        simulate(); return;
      }
      heat = frames; view.animating = true;
      if (frame === null) frame = root.requestAnimationFrame(tick);
    }
    function point(event) {
      // Invert SVG screen CTM: works with CSS sizing, page scroll, and transforms.
      try {
        const matrix = svg.getScreenCTM?.();
        if (matrix && svg.createSVGPoint) {
          const p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
          const converted = p.matrixTransform(matrix.inverse());
          if (finitePoint(converted)) return converted;
        }
      } catch { /* Detached SVGs and minimal test DOMs use the viewBox fallback. */ }
      const rect = svg.getBoundingClientRect?.();
      return { x: (event.clientX - (rect?.left || 0)) * width / (rect?.width || width),
        y: (event.clientY - (rect?.top || 0)) * HEIGHT / (rect?.height || HEIGHT) };
    }
    function begin(event, node, target) {
      if (disposed || document.hidden || gesture || (event.button !== undefined && event.button !== 0)) return;
      stopAnimation(); suppressed = null;
      const p = point(event);
      gesture = { node, element: target, pointerId: event.pointerId, start: p,
        clientX: event.clientX, clientY: event.clientY, x: node?.x ?? view.camera.x,
        y: node?.y ?? view.camera.y, scale: view.camera.scale, moved: false };
      view.interacting = true;
      event.stopPropagation?.();
      try { target.setPointerCapture?.(event.pointerId); } catch { /* Optional in test DOMs. */ }
    }
    function move(event) {
      if (!gesture || disposed || (event.pointerId !== undefined && event.pointerId !== gesture.pointerId)) return;
      if (document.hidden || svg.isConnected === false) { svg.dispose(); return; }
      event.stopPropagation?.();
      const g = gesture, p = point(event);
      if (Math.hypot(event.clientX - g.clientX, event.clientY - g.clientY) > 4) g.moved = true;
      if (!g.moved) return;
      event.preventDefault?.(); view.autoFit = false;
      if (g.node) {
        g.node.x = g.x + (p.x - g.start.x) / g.scale;
        g.node.y = g.y + (p.y - g.start.y) / g.scale;
        constrain(g.node); pinned.add(g.node.key);
        g.node.anchor = { x: g.node.x, y: g.node.y };
        paint(); reheat();
      } else {
        view.camera.x = g.x + p.x - g.start.x; view.camera.y = g.y + p.y - g.start.y; paint();
      }
    }
    function finish(event = {}) {
      if (!gesture || (event.pointerId !== undefined && event.pointerId !== gesture.pointerId)) return;
      const g = gesture;
      if (g.moved && g.node) suppressed = g.node.key;
      release();
      if (g.moved && g.node && event.type !== "pointercancel" && event.type !== "lostpointercapture") reheat(18);
      else stopAnimation();
      cache(); notifyIdle();
    }
    for (const node of model.nodes) {
      const isRoot = node.type === "root", active = !isRoot && node.key === selected;
      const fullLabel = String(isRoot ? startLabel : (typeof label === "function"
        ? label(node.row, node.index) : node.row.operation || node.row.action || node.row.mode
          || node.row.kind || node.row.decision_id || `Decision ${node.index + 1}`) ?? "");
      const group = element("g", { class: isRoot ? "decision-root" : `decision-node flow-choice${active ? " is-selected" : ""}`,
        "data-key": node.key });
      group.append(element("title", {}, fullLabel));
      if (!isRoot) {
        group.style.cursor = "grab";
        group.setAttribute("role", "button"); group.setAttribute("tabindex", "0");
        group.setAttribute("aria-label", fullLabel); group.setAttribute("aria-pressed", String(active));
        const select = () => { if (!disposed && typeof onSelect === "function") onSelect(node.key, node.row, node.index); };
        listen(group, "click", event => {
          if (suppressed === node.key && event?.detail !== 0) { suppressed = null; return; }
          select();
        });
        listen(group, "keydown", event => {
          if (["Enter", " ", "Spacebar"].includes(event.key)) {
            event.preventDefault(); if (!event.repeat) { suppressed = null; select(); }
          }
        });
        listen(group, "pointerdown", event => begin(event, node, group));
        listen(group, "pointermove", move);
        for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) listen(group, type, finish);
        group.append(element("circle", { r: 18, fill: "transparent", "pointer-events": "all" }));
      }
      if (active) group.append(element("circle", { class: "decision-halo", r: 13,
        fill: "var(--orange, #b45b35)", opacity: 0.18, "pointer-events": "none" }));
      group.append(element("circle", { class: "decision-dot", r: active ? 8 : 6,
        fill: active ? "var(--orange, #b45b35)" : "var(--muted, #8a7568)" }));
      const characters = Array.from(fullLabel.replace(/\s+/g, " "));
      const brief = characters.length > 14 ? characters.slice(0, 13).join("") + "…" : characters.join("");
      const text = element("text", { x: 0, y: 32, "text-anchor": "middle", "font-size": 16, "font-family": "sans-serif",
        fill: "var(--muted, #8a7568)" }, brief);
      const estimated = Array.from(brief).reduce((sum, char) => sum + (char.codePointAt(0) > 255 ? 16 : 10), 0);
      if (estimated > 0) {
        text.setAttribute("textLength", String(Math.min(estimated, LABEL_WIDTH)));
        text.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      group.append(text); plane.append(group); groups.set(node.key, group);
    }
    listen(svg, "pointerdown", event => {
      // Node events stop bubbling; root and labels also never start background pan.
      let target = event.target;
      while (target && target !== svg) {
        if (target === groups.get("root") || [...groups.values()].includes(target)) return;
        target = target.parentNode;
      }
      begin(event, null, svg);
    });
    listen(svg, "pointermove", move);
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) listen(svg, type, finish);
    function zoom(factor, center = { x: width / 2, y: HEIGHT / 2 }) {
      if (!Number.isFinite(factor) || factor <= 0 || disposed) return;
      const camera = view.camera, scale = Math.max(0.005, Math.min(6, camera.scale * factor));
      const ratio = scale / camera.scale;
      camera.x = center.x - (center.x - camera.x) * ratio;
      camera.y = center.y - (center.y - camera.y) * ratio;
      camera.scale = scale; view.autoFit = false; paint();
    }
    listen(svg, "wheel", event => {
      if (disposed || (!event.ctrlKey && !event.metaKey)) return;
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? HEIGHT : 1);
      zoom(Math.exp(-Math.max(-200, Math.min(200, delta)) * 0.002), point(event)); notifyIdle();
    }, { passive: false });
    function fitCamera(initial = false) {
      let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
      for (const n of model.nodes) {
        left = Math.min(left, n.x - LABEL_WIDTH / 2 - 16); right = Math.max(right, n.x + LABEL_WIDTH / 2 + 16);
        top = Math.min(top, n.y - 24); bottom = Math.max(bottom, n.y + 44);
      }
      const scale = Math.max(initial ? 0.85 : 0.005, Math.min(1, (width - 64) / (right - left), (HEIGHT - 64) / (bottom - top)));
      view.camera.scale = scale;
      // Large graphs open at the fixed start; fit() can explicitly show all nodes.
      view.camera.x = initial && (right - left) * scale > width - 64 ? 32 - left * scale : (width - (left + right) * scale) / 2;
      view.camera.y = initial && (bottom - top) * scale > HEIGHT - 64 ? HEIGHT / 2 - Y * scale : (HEIGHT - (top + bottom) * scale) / 2;
    }
    function resize() {
      if (disposed) return;
      const measured = svg.getBoundingClientRect?.().width || svg.clientWidth || svg.parentElement?.clientWidth;
      if (Number.isFinite(measured) && measured > 0) width = measured;
      const changed = Number.isFinite(view.viewportWidth) && view.viewportWidth !== width;
      view.viewportWidth = width;
      svg.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);
      if (view.autoFit && !view.interacting && !view.animating && (!viewportReady || changed)) fitCamera(!view.fitAll);
      viewportReady = true;
      paint();
    }
    svg.dispose = () => {
      if (disposed) return;
      disposed = true; cancelInteraction(); observer?.disconnect();
      for (const remove of removers.splice(0)) remove();
    };
    svg.fit = () => { if (disposed) return; cancelInteraction(); view.autoFit = true; view.fitAll = true; fitCamera(); paint(); notifyIdle(); };
    svg.zoomBy = factor => { if (disposed) return; zoom(factor); notifyIdle(); };
    svg.reset = () => {
      if (disposed) return;
      cancelInteraction(); pinned.clear();
      for (const n of model.nodes) {
        n.x = n.home.x; n.y = n.home.y; n.anchor = { x: n.x, y: n.y };
      }
      for (const edge of structural) { edge.dx = edge.to.x - edge.from.x; edge.dy = edge.to.y - edge.from.y; }
      view.autoFit = true; view.fitAll = false; fitCamera(true); paint(); notifyIdle();
    };
    listen(document, "visibilitychange", () => { if (document.hidden) svg.dispose(); });
    if (typeof root.ResizeObserver === "function") {
      observer = new root.ResizeObserver(() => {
        if (svg.isConnected === false) { svg.dispose(); return; }
        resize();
      });
      observer.observe(svg);
    }
    resize();
    return svg;
  }

  const api = Object.freeze({ layout, render });
  root.RSIDecisions = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
