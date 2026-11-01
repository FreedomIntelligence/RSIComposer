"use strict";

/**
 * layout(rows) -> {nodes, edges, width, height}; no DOM access or input mutation.
 * Nodes: {key, index, row, x, y, type}; root has key "root", index -1, row null.
 * Edges: {from, to, kind}: dependency, adoption, source (reference only), or
 * sequence (NOT causality; also connects independent nodes to root). Only unique,
 * nonempty string IDs resolve references. Adoption wins for the same pair.
 * Unique IDs are keys, except the reserved root ID; other keys use stable record
 * fields, with occurrence suffixes for indistinguishable records/collisions.
 * Valid timestamps sort ascending, ties keep input order; unknowns come last.
 * depends_on_decision_ids and adopted_decision_ids opt the dataset into explicit
 * lineage, even when empty: independent nodes attach to root. Legacy data alone
 * uses chronological sequence fallback. parent_decision_id remains a dependency.
 * Cyclic structural edges are removed; valid outside edges survive. References
 * never affect topology. Longest-path layers center forks and multi-parent joins.
 * Root stays fixed. This pure layout is independent of the interactive renderer.
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

  function separateReferences(nodes,edges,pinned=new Set()) {
    const byKey=new Map(nodes.map(node=>[node.key,node])),fixed=new Set(["root",...pinned]);
    // Keep every real DAG layer and legacy sequence intact. Only independent
    // reference nodes may move sideways; references never become dependencies.
    for(const edge of edges)if(edge.kind!=="source"&&edge.from!=="root"){
      fixed.add(edge.from);fixed.add(edge.to);
    }
    const references=edges.filter(edge=>edge.kind==="source"),moves=new Map();
    const obstructs=(a,b,p)=>{
      const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy;
      if(!length)return false;
      const t=((p.x-a.x)*dx+(p.y-a.y)*dy)/length;
      return t>0&&t<1&&Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy)<24;
    };
    for(let pass=0;pass<4;pass++){
      let changed=false;
      for(const edge of references){
        const a=byKey.get(edge.from),b=byKey.get(edge.to);
        for(const obstacle of nodes){
          if(obstacle===a||obstacle===b||!obstructs(a,b,obstacle))continue;
          const movable=[obstacle,b,a].find(node=>!fixed.has(node.key)&&(moves.get(node.key)||0)<4);
          if(!movable)continue;
          movable.x+=80;
          // A side step must not trade a crossed dot for overlapping labels.
          while(nodes.some(other=>other!==movable&&Math.abs(other.x-movable.x)<180&&Math.abs(other.y-movable.y)<60))movable.y+=LANE;
          moves.set(movable.key,(moves.get(movable.key)||0)+1);changed=true;
        }
      }
      if(!changed)break;
    }
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
    // New lineage fields describe a DAG, including an explicitly empty list.
    // Source references remain independent of topology, including on legacy rows.
    const modern = records.some(({row}) => Object.hasOwn(row, "depends_on_decision_ids") || Object.hasOwn(row, "adopted_decision_ids"));
    const parents = records.map(() => new Map());
    records.forEach(({row}, to) => {
      const add = (id, kind) => {
        const from = resolve(id);
        if (from < 0 || from === to) return;
        if (!parents[to].has(from) || kind === "adoption") parents[to].set(from, kind);
      };
      add(row.parent_decision_id, "dependency");
      for (const id of Array.isArray(row.depends_on_decision_ids) ? row.depends_on_decision_ids : []) add(id, "dependency");
      for (const id of Array.isArray(row.adopted_decision_ids) ? row.adopted_decision_ids : []) add(id, "adoption");
    });
    const children = records.map(() => []);
    parents.forEach((incoming, to) => { for (const from of incoming.keys()) children[from].push(to); });
    // Iterative Kosaraju: remove only edges inside a strongly connected component.
    // Other parents of a cycle member and its valid descendants remain visible.
    const visited = new Set(), order = [];
    for (let start = 0; start < records.length; start++) {
      if (visited.has(start)) continue;
      visited.add(start);
      const stack = [[start, 0]];
      while (stack.length) {
        const top = stack[stack.length - 1], next = children[top[0]][top[1]++];
        if (next === undefined) { order.push(top[0]); stack.pop(); }
        else if (!visited.has(next)) { visited.add(next); stack.push([next, 0]); }
      }
    }
    const component = new Map(); let componentID = 0;
    for (const start of order.reverse()) {
      if (component.has(start)) continue;
      const stack = [start]; component.set(start, componentID);
      while (stack.length) {
        for (const parent of parents[stack.pop()].keys()) if (!component.has(parent)) {
          component.set(parent, componentID); stack.push(parent);
        }
      }
      componentID++;
    }
    parents.forEach((incoming, to) => {
      for (const from of incoming.keys()) if (component.get(from) === component.get(to)) incoming.delete(from);
    });
    const hasBackward = parents.some((incoming, to) => [...incoming.keys()].some(from => from > to));
    const reaches = (start, target) => {
      const seen = new Set(), stack = [start];
      while (stack.length) {
        const next = stack.pop();
        if (next === target) return true;
        if (seen.has(next)) continue; seen.add(next);
        stack.push(...parents[next].keys());
      }
      return false;
    };
    records.forEach(({row}, i) => {
      if (parents[i].size) return;
      const declared = Object.hasOwn(row, "parent_decision_id") && row.parent_decision_id != null && row.parent_decision_id !== "";
      const previous = i - 1;
      if (!modern && !declared && previous >= 0 && (!hasBackward || !reaches(previous, i))) parents[i].set(previous, "sequence");
    });
    for (const outgoing of children) outgoing.length = 0;
    parents.forEach((incoming, to) => { for (const from of incoming.keys()) children[from].push(to); });
    const degree = parents.map(incoming => incoming.size), depth = records.map(() => 1);
    const queue = records.map((_, i) => i).filter(i => !degree[i]), topology = [];
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const from = queue[cursor]; topology.push(from);
      for (const to of children[from]) {
        depth[to] = Math.max(depth[to], depth[from] + 1);
        if (--degree[to] === 0) queue.push(to);
      }
    }
    const nodes = [{ key: "root", index: -1, row: null, x: X, y: Y, type: "root" },
      ...records.map(({key, index, row}, i) => ({key, index, row, x: X + depth[i] * COLUMN, y: Y, type: "decision"}))];
    const layers = [];
    for (const i of topology) (layers[depth[i]] ||= []).push(i);
    const average = values => values.reduce((sum, value) => sum + value, 0) / values.length;
    const place = (layer, desired) => {
      const ordered = layer.map(i => ({i, y:desired(i)})).sort((a,b) => a.y-b.y || a.i-b.i);
      let last = -Infinity;
      const packed = ordered.map(item => (last = Math.max(item.y, last + LANE)));
      const shift = Math.min(packed[0] - Y, average(packed.map((y, i) => y - ordered[i].y)));
      ordered.forEach(({i}, rank) => { nodes[i+1].y = packed[rank] - shift; });
    };
    // Barycentric sweeps center forks and joins; packing keeps each layer legible.
    for (let pass = 0; pass < 3; pass++) {
      for (const layer of layers) if (layer) place(layer, i => parents[i].size ? average([...parents[i].keys()].map(p => nodes[p+1].y)) : Y);
      for (let level = layers.length - 1; level > 0; level--) if (layers[level]) {
        place(layers[level], i => children[i].length ? average(children[i].map(c => nodes[c+1].y)) : nodes[i+1].y);
      }
    }
    const edges = [];
    records.forEach((record, to) => {
      if (!parents[to].size) edges.push({from:"root", to:record.key, kind:"sequence"});
      for (const [from, kind] of parents[to]) edges.push({from:records[from].key, to:record.key, kind});
      const sources = record.row.source_decision_ids;
      for (const id of new Set(Array.isArray(sources) ? sources : [])) {
        const from = resolve(id);
        if (from >= 0 && from !== to && !parents[to].has(from)) edges.push({from:records[from].key, to:record.key, kind:"source"});
      }
    });
    separateReferences(nodes,edges);
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
    const structural = [],incoming = new Map(model.nodes.map(node => [node,[]])),outgoing = new Map(model.nodes.map(node => [node,[]]));
    for (const edge of model.edges) {
      const from = byKey.get(edge.from), to = byKey.get(edge.to);
      const attributes = { class: `decision-edge decision-edge-${edge.kind}`, "data-kind": edge.kind,
        stroke: edge.kind === "adoption" ? "var(--orange, #b45b35)" : "var(--muted, #8a7568)",
        "stroke-width": edge.kind === "adoption" ? 1.5 : 1, "vector-effect": "non-scaling-stroke",
        opacity: edge.kind === "adoption" ? 0.85 : edge.kind === "dependency" ? 0.6 : edge.kind === "sequence" ? 0.25 : 0.45 };
      if (edge.kind === "source") attributes["stroke-dasharray"] = "4 4";
      const line = element("line", attributes);
      const defaultTitle = {sequence:"Sequence (not causality)",dependency:"Declared dependency",
        adoption:"Confirmed adoption",source:"Source reference (not dependency or adoption)"}[edge.kind];
      line.append(element("title", {}, typeof edgeLabel === "function" ? edgeLabel(edge.kind) ?? defaultTitle : defaultTitle));
      plane.append(line); lines.push({ line, from, to });
      if (edge.kind !== "source") {
        const relation={from,to,dx:to.x-from.x,dy:to.y-from.y};structural.push(relation);
        incoming.get(to).push(relation);outgoing.get(from).push(relation);
      }
    }
    // A newly reported parent can invalidate cached x positions. Project forward
    // in topological order, retaining manual y placement and the camera.
    separateReferences(model.nodes,model.edges,pinned);
    const physicsOrder=model.nodes.slice().sort((a,b)=>a.home.x-b.home.x);
    for(const node of physicsOrder){
      for(const {from} of incoming.get(node))node.x=Math.max(node.x,from.x+60);
      node.anchor.x=node.x;
      node.anchor.y=node.y;
    }
    for(const edge of structural){edge.dx=edge.to.x-edge.from.x;edge.dy=edge.to.y-edge.from.y;}
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
      for (const edge of incoming.get(node)) left = Math.max(left, edge.from.x + 60);
      for (const edge of outgoing.get(node)) right = Math.min(right, edge.to.x - 60);
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
      // Preserve natural glyph proportions; wrap words instead of squeezing text.
      const measure = value => Array.from(value).reduce((sum, char) => sum +
        (char.codePointAt(0) > 255 ? 16 : /[MW@#]/.test(char) ? 14 : /[il.,:;!| ]/.test(char) ? 5 : 9), 0);
      const remaining = Array.from(fullLabel.replace(/\s+/g, " ").trim()), labelLines = [];
      while(remaining.length && labelLines.length < 2){
        let count=0;
        while(count<remaining.length && measure(remaining.slice(0,count+1).join(""))<=LABEL_WIDTH-8)count++;
        count=Math.max(1,count);
        if(count<remaining.length && labelLines.length===0){
          const boundary=remaining.slice(0,count+1).lastIndexOf(" ");
          if(boundary>Math.floor(count/2))count=boundary;
        }
        let line=remaining.splice(0,count).join("").trim();
        while(remaining[0]===" ")remaining.shift();
        if(labelLines.length===1 && remaining.length){
          while(measure(line+"…")>LABEL_WIDTH-8)line=Array.from(line).slice(0,-1).join("");
          line+="…";
        }
        labelLines.push(line);
      }
      if(!labelLines.length)labelLines.push("");
      const text = element("text", { class: "decision-label", x: 0, y: 30,
        "text-anchor": "middle", "font-size": 16, "font-family": "var(--sans, sans-serif)" });
      if(labelLines.length===1)text.textContent=labelLines[0];
      else labelLines.forEach((line,index)=>text.append(element("tspan",{x:0,y:30+index*22},line)));
      const labelWidth=Math.max(...labelLines.map(measure));
      group.append(element("rect", { class: "decision-label-backplate", x: -labelWidth / 2 - 7,
        y: 13, width: labelWidth + 14, height: labelLines.length*22+4, rx: 3,
        fill: "#f8f7f2", "pointer-events": "none" }));
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
        top = Math.min(top, n.y - 24); bottom = Math.max(bottom, n.y + 64);
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
