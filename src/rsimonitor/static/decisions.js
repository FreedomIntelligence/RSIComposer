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
 * Invalid raw self references and structural cycles are filtered with issues;
 * valid outside edges survive. References never affect topology.
 * attempts(rows) contracts ONLY recorded research_attempt_id identities; each
 * node retains every member and each edge retains its original decision edges.
 * Titles are literal metadata, never interpreted as outcomes. Missing IDs remain
 * in experiment logs, never overview nodes. Contraction cycles retain Attempt
 * nodes and recorded edges with diagnostics; full evidence stays in drilldown.
 * Root stays fixed. This pure layout is independent of the interactive renderer.
 *
 * render({rows, model, selected, onSelect, label, startLabel, edgeLabel, view, onIdle, compact})
 * returns a responsive SVG (height defaults to 420px) with dispose(), fit(),
 * zoomBy(factor), reset(). Supplied projections are never mutated by rendering.
 * view is caller-owned per run: positions Map<key,{x,y}>, camera {x,y,scale},
 * pinned Set<key>, interacting/animating booleans. Reuse it across rerenders.
 * compact uses tighter vertical spacing for a summarized view. Dragged nodes
 * stay pinned until reset. Only layout edges are ever drawn.
 * onIdle() follows interaction/settling; dispose() NEVER calls it.
 * No idle RAF/timer: a drag starts at most 24 frames, release at most 18 frames.
 * selected is a node key; label(row, originalIndex) returns its full label.
 * onSelect(key, row, originalIndex, keyboard) fires for click, Enter, or Space.
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
    const attemptMetadata = records.some(({row}) => Object.hasOwn(row, "research_attempt_id") || Object.hasOwn(row, "research_attempt_title"));
    const modern = attemptMetadata || records.some(({row}) => Object.hasOwn(row, "depends_on_decision_ids") || Object.hasOwn(row, "adopted_decision_ids"));
    const issues = [];
    const parents = records.map(() => new Map());
    records.forEach(({row}, to) => {
      const add = (id, kind) => {
        const from = resolve(id);
        if (from < 0) { if (identifier(id)) issues.push({type:"unresolved", id, to:records[to].key}); return; }
        if (from === to) { issues.push({type:"self-reference",id,to:records[to].key}); return; }
        if (!parents[to].has(from) || kind === "adoption") parents[to].set(from, kind);
      };
      add(row.parent_decision_id, "dependency");
      for (const id of Array.isArray(row.depends_on_decision_ids) ? row.depends_on_decision_ids : []) add(id, "dependency");
      for (const id of Array.isArray(row.adopted_decision_ids) ? row.adopted_decision_ids : []) add(id, "adoption");
    });
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
      if (!modern && !declared && i > 0 && !reaches(i - 1, i)) parents[i].set(i - 1, "sequence");
    });
    const edges = [];
    records.forEach((record, to) => {
      for (const [from, kind] of parents[to]) edges.push({from:records[from].key, to:record.key, kind});
      for (const id of new Set(Array.isArray(record.row.source_decision_ids) ? record.row.source_decision_ids : [])) {
        const from = resolve(id);
        if (from >= 0 && from !== to) edges.push({from:records[from].key, to:record.key, kind:"source"});
        else if (from < 0 && identifier(id)) issues.push({type:"unresolved", id, to:record.key});
      }
    });
    return arrange(records, edges, {issues, rootKind:attemptMetadata ? "root" : "sequence",discardCycles:true});
  }

  // Use strongly connected components for geometry, retaining every real edge.
  // This also handles cycles introduced by contracting interleaved attempts.
  function arrange(records, edges, {issues = [], rootKind = "root",discardCycles=false} = {}) {
    const rank = new Map(records.map((r,i)=>[r.key,i]));
    const parents = records.map(()=>new Set()), children = records.map(()=>new Set());
    for (const edge of edges) if (edge.kind !== "source") {
      const from=rank.get(edge.from),to=rank.get(edge.to);
      parents[to].add(from);children[from].add(to);
    }
    const visited = new Set(), order = [];
    for (let start = 0; start < records.length; start++) {
      if (visited.has(start)) continue;
      visited.add(start);
      const stack = [[start, 0]];
      while (stack.length) {
        const top = stack[stack.length - 1], next = [...children[top[0]]][top[1]++];
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
    const members=Array.from({length:componentID},()=>[]),nextComponents=members.map(()=>new Set()),degree=members.map(()=>0),componentDepth=members.map(()=>1);
    records.forEach((_,i)=>members[component.get(i)].push(i));
    for (const edge of edges) if (edge.kind !== "source") {
      const from=component.get(rank.get(edge.from)),to=component.get(rank.get(edge.to));
      if (from===to) { edge.cyclic=true; issues.push({type:discardCycles?"raw-cycle":"cycle",from:edge.from,to:edge.to}); }
      else if(!nextComponents[from].has(to)){nextComponents[from].add(to);degree[to]++;}
    }
    if(discardCycles&&edges.some(e=>e.cyclic))return arrange(records,edges.filter(e=>!e.cyclic),{issues,rootKind});
    const roots=new Set(degree.map((n,i)=>n===0?i:-1).filter(i=>i>=0));
    const queue=[...roots],topology=[];
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const from = queue[cursor]; topology.push(...members[from]);
      for (const to of nextComponents[from]) {
        componentDepth[to] = Math.max(componentDepth[to], componentDepth[from] + 1);
        if (--degree[to] === 0) queue.push(to);
      }
    }
    const depth=records.map((_,i)=>componentDepth[component.get(i)]);
    parents.forEach((incoming,to)=>{for(const from of incoming)if(component.get(from)===component.get(to)){incoming.delete(from);children[from].delete(to);}});
    const nodes = [{ key: "root", index: -1, row: null, x: X, y: Y, type: "root" },
      ...records.map((record, i) => ({...record, x: X + depth[i] * COLUMN, y: Y, type:record.type||"decision"}))];
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
        place(layers[level], i => children[i].size ? average([...children[i]].map(c => nodes[c+1].y)) : nodes[i+1].y);
      }
    }
    // Root connectors are navigation anchors, never inferred dependencies.
    const structuralPairs=new Set(edges.filter(e=>e.kind!=="source").map(e=>JSON.stringify([e.from,e.to]))),rooted=[];
    records.forEach((record,i)=>{
      if(roots.has(component.get(i)))rooted.push({from:"root",to:record.key,kind:rootKind});
      rooted.push(...edges.filter(edge=>edge.to===record.key&&(edge.kind!=="source"||!structuralPairs.has(JSON.stringify([edge.from,edge.to])))));
    });
    separateReferences(nodes,rooted);
    let width = X + LABEL_WIDTH + 24, height = 240;
    for (const node of nodes) {
      width = Math.max(width, node.x + LABEL_WIDTH + 24);
      height = Math.max(height, node.y + 56);
    }
    return { nodes, edges:rooted, width, height, issues };
  }

  function attempts(rows = [], interpretations = null) {
    const interpreted=interpretations?.schema_version===1 && interpretations.authority==="model_interpretation";
    const items=interpreted && Array.isArray(interpretations.items)?interpretations.items:[];
    const definitions=new Map(),membership=new Map(),conflicts=new Set();
    for(const item of items){
      if(!identifier(item?.attempt_id)||definitions.has(item.attempt_id))continue;
      definitions.set(item.attempt_id,item);
      for(const id of Array.isArray(item.member_decision_ids)?item.member_decision_ids:[]){
        if(!identifier(id))continue;
        if(membership.has(id)&&membership.get(id)!==item.attempt_id)conflicts.add(id);
        membership.set(id,item.attempt_id);
      }
    }
    // View-only membership: the original decision ledger remains untouched.
    const projected=interpreted?rows.map(row=>{
      const copy={...row};delete copy.research_attempt_id;delete copy.research_attempt_title;
      const id=conflicts.has(row.decision_id)?null:membership.get(row.decision_id),item=definitions.get(id);
      if(item){copy.research_attempt_id=id;copy.research_attempt_title=item.title;}
      return copy;
    }):rows;
    const full=layout(projected),records=full.nodes.filter(n=>n.row),groups=[],byID=new Map(),owner=new Map(),issues=[...full.issues];
    const used=new Set(full.nodes.map(n=>n.key));
    for(const node of records){
      const row=node.row,id=row.research_attempt_id;
      let group;
      if(identifier(id)){
        group=byID.get(id);
        if(!group){
          let key="attempt:"+JSON.stringify(id);while(used.has(key))key="attempt:"+key;used.add(key);
          group={key,index:groups.length,row,rows:[],members:[],type:"attempt",attemptID:id,titles:[]};
          byID.set(id,group);groups.push(group);
        }
        if(identifier(row.research_attempt_title)&&!group.titles.includes(row.research_attempt_title))group.titles.push(row.research_attempt_title);
        if(!identifier(row.research_attempt_title))issues.push({type:"missing-title",key:group.key});
      }else{
        group={key:node.key,index:groups.length,row,rows:[],members:[],type:"ungrouped"};groups.push(group);
        if(Object.hasOwn(row,"research_attempt_id")||Object.hasOwn(row,"research_attempt_title"))issues.push({type:"invalid-attempt",key:group.key});
      }
      group.rows.push(row);group.members.push(node);owner.set(node.key,group.key);
    }
    // Keep interpreted attempts even when all members predate the bounded snapshot.
    for(const [id,item] of definitions){
      let group=byID.get(id);
      if(!group){
        let key="attempt:"+JSON.stringify(id);while(used.has(key))key="attempt:"+key;used.add(key);
        group={key,index:groups.length,row:{},rows:[],members:[],type:"attempt",attemptID:id,titles:identifier(item.title)?[item.title]:[]};
        byID.set(id,group);groups.push(group);
      }
      group.interpretation=item;
      const visible=new Set(group.rows.map(row=>row.decision_id));
      group.missingDecisionIDs=[...new Set(Array.isArray(item.member_decision_ids)?item.member_decision_ids:[])].filter(id=>!visible.has(id));
    }
    for(const group of groups)if(group.type==="attempt"){
      group.title=group.titles[0]||"";
      if(group.titles.length>1)issues.push({type:"conflicting-title",key:group.key});
    }
    const contracted=new Map(),attemptKeys=new Set(groups.filter(g=>g.type==="attempt").map(g=>g.key));
    const rawByKey=new Map(records.map(n=>[n.key,n]));
    for(const edge of full.edges){
      if(edge.from==="root" || (interpreted&&edge.kind==="sequence"))continue;
      const from=owner.get(edge.from),to=owner.get(edge.to);
      if(from===to||!attemptKeys.has(from)||!attemptKeys.has(to))continue;
      const key=JSON.stringify([from,to]);
      if(!contracted.has(key))contracted.set(key,{from,to,kind:edge.kind,decisions:[],types:[]});
      const link=contracted.get(key),priority={sequence:0,source:1,dependency:2,adoption:3};
      if(priority[edge.kind]>priority[link.kind])link.kind=edge.kind;
      const target=rawByKey.get(edge.to).row,id=rawByKey.get(edge.from).row.decision_id;
      const includes=(field)=>Array.isArray(target[field])&&target[field].includes(id);
      const types=edge.kind==="sequence"?["sequence"]:edge.kind==="source"?["source"]:[
        ...(target.parent_decision_id===id||includes("depends_on_decision_ids")?["dependency"]:[]),
        ...(includes("adopted_decision_ids")?["adoption"]:[]),
        ...(includes("source_decision_ids")?["source"]:[])];
      for(const kind of types){if(!link.types.includes(kind))link.types.push(kind);link.decisions.push({from:edge.from,to:edge.to,kind});}
    }
    // Never route through unassigned records or infer a transitive Attempt edge.
    const ungrouped=groups.filter(g=>g.type==="ungrouped");
    const model=arrange(groups.filter(g=>g.type==="attempt"),[...contracted.values()],{issues,rootKind:"root"});
    if(model.issues.some(i=>i.type==="cycle"))model.issues.push({type:"contraction-cycle"});
    return {...model,full,attemptCount:byID.size,ungroupedCount:ungrouped.length,ungrouped};
  }

  function subpath(projection, key) {
    const group=projection.nodes.find(n=>n.key===key),members=group?.members||[];
    const keys=new Set(members.map(n=>n.key));
    return arrange(members.map(({key,index,row})=>({key,index,row})),
      projection.full.edges.filter(e=>keys.has(e.from)&&keys.has(e.to)).map(e=>({...e})));
  }

  function render({ rows = [], model: suppliedModel, selected = null, onSelect, label, startLabel = "Start", edgeLabel,
    view = {}, onIdle, compact = false, height = 420 } = {}) {
    const original=suppliedModel||layout(rows),model={...original,nodes:original.nodes.map(n=>({...n})),edges:original.edges.map(e=>({...e}))}, HEIGHT = height, document = root.document;
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
        : Y + (node.y - Y) * (compact ? 1 : 1.8) + (compact ? 0 : depth % 2 ? -64 : 64) + hash(node.key) * 20 - 10 };
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
        opacity: edge.kind === "adoption" ? 0.85 : edge.kind === "dependency" ? 0.6 : ["sequence","root"].includes(edge.kind) ? 0.25 : 0.45 };
      if (edge.kind === "source") attributes["stroke-dasharray"] = "4 4";
      const line = element("line", attributes);
      const defaultTitle = {sequence:"Sequence (not causality)",dependency:"Declared dependency",
        root:"Layout anchor (not dependency)",adoption:"Confirmed adoption",source:"Source reference (not dependency or adoption)"}[edge.kind];
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
      const failed = !isRoot && node.type!=="attempt" && ["failed","failure","error","cancelled","canceled"].includes(node.row.feedback?.status||node.row.status);
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
        const select = (keyboard=false) => { if (!disposed && typeof onSelect === "function") onSelect(node.key, node.row, node.index, keyboard); };
        listen(group, "click", event => {
          if (suppressed === node.key && event?.detail !== 0) { suppressed = null; return; }
          select();
        });
        listen(group, "keydown", event => {
          if (["Enter", " ", "Spacebar"].includes(event.key)) {
            event.preventDefault(); if (!event.repeat) { suppressed = null; select(true); }
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
        fill: active ? "var(--orange, #b45b35)" : failed ? "var(--red, #9a4d42)" : "var(--muted, #8a7568)" }));
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

  const api = Object.freeze({ layout, attempts, subpath, render });
  root.RSIDecisions = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
