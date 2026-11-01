"use strict";

/**
 * Pure run lineage: RSILineage.build(nodes, edges = []) -> { nodes, edges }.
 * Also exported through CommonJS for standalone tests. No DOM or dependencies.
 *
 * Input nodes have a string key, type "project" or "run", and optional
 * layoutOrder. Run identity is node.runId (falling back to row.run_id).
 * node.row, when present, is the COMPLETE authoritative record; only when it is
 * absent is node.snapshot used. Neither old edges nor status/score establish
 * parentage. The caller should put its authoritative record in node.row.
 *
 * Output preserves node order and all original fields (shallow copies), adding:
 *   depth: project=0, first-level run=1, otherwise tree parent depth + 1;
 *   iterationDepth: initial/invalid-parent run=0, valid parent hops thereafter;
 *   treeRow: zero-based leaf rows, parents centered on their subtree leaf span;
 *   treeParentKey: layout parent key, or null for projects/unassigned runs;
 *   parentRunId: declared parent identifier, or null;
 *   ancestryStatus: "root" | "initial" | "known" | "unknown";
 *   ancestryReason: null, or the diagnostic reason for unknown ancestry;
 *   missing / ambiguous: unresolved parent AND source identifiers;
 *   sourceReferences: [{runId, status, sourceKey}], independent of ancestry.
 * These depths count DECLARED continuation links, never accepted training runs
 * or improvements. Acceptance requires separate evidence and UI explanation.
 *
 * All cycle members lose their cyclic parent link and attach to the project
 * root. Valid descendants retain their direct parent link, with unknown ancestry
 * propagated and depth counted only along the remaining acyclic links.
 *
 * Siblings sort by numeric layoutOrder, then key, then run ID; exact duplicates
 * retain input order. Each project occupies a separate span of leaf rows.
 * Identity resolution is project-local: explicit node.projectKey, projectId,
 * row.project_id / row.project.project_id, or input membership edges establish
 * ownership. A lone project is the fallback ONLY without explicit ownership.
 * Conflicting/unknown ownership is never guessed; such runs remain unattached.
 * Duplicate graph keys cannot safely be edge endpoints and are marked unknown.
 *
 * Edges are rebuilt as membership (root -> first level, including unknown),
 * run-parent (valid parent -> child), and run-source (source -> child reference).
 * Matching input edges supply decoration only; obsolete edges are dropped.
 * Sources, including one also named as parent, remain separate reference edges.
 * All traversals are iterative and linear; deterministic sibling/run/source
 * sorting adds O((N + E) log(N + E)) worst-case time. Space is O(N + E).
 */
(function (root) {
  const identifier = value => typeof value === "string" && value.trim().length > 0;
  const record = node => node.row ?? node.snapshot ?? {};
  const compareText = (a, b) => a < b ? -1 : a > b ? 1 : 0;
  const edgeId = (from, to, kind) => JSON.stringify([from, to, kind]);

  function build(inputNodes = [], inputEdges = []) {
    const nodes = inputNodes.map(node => ({ ...node }));
    const count = nodes.length;
    const records = nodes.map(record);
    const runIds = nodes.map((node, i) => node.runId ?? records[i].run_id);
    const keys = new Map(), projectsById = new Map(), projects = [], runs = [];
    const parent = new Int32Array(count).fill(-1);
    const project = new Int32Array(count).fill(-1);
    const children = Array.from({ length: count }, () => []);
    const reasons = new Array(count).fill(null);
    const memberships = new Map(), decorations = new Map();

    function add(index, key, value) {
      if (!index.has(key)) index.set(key, []);
      index.get(key).push(value);
    }
    function uniqueKey(key) {
      const matches = keys.get(key);
      return matches?.length === 1 ? matches[0] : -1;
    }
    function compare(a, b) {
      const ao = Number.isFinite(nodes[a].layoutOrder) ? nodes[a].layoutOrder : 0;
      const bo = Number.isFinite(nodes[b].layoutOrder) ? nodes[b].layoutOrder : 0;
      return ao - bo || compareText(nodes[a].key, nodes[b].key)
        || compareText(runIds[a] || "", runIds[b] || "");
    }

    nodes.forEach((node, i) => {
      if (!identifier(node.key) || !["project", "run"].includes(node.type)) {
        throw new TypeError('Lineage nodes require a string key and type "project" or "run"');
      }
      add(keys, node.key, i);
      Object.assign(node, { depth: node.type === "project" ? 0 : 1,
        iterationDepth: 0, treeRow: 0, treeParentKey: null, parentRunId: null,
        ancestryStatus: node.type === "project" ? "root" : "initial",
        ancestryReason: null, missing: [], ambiguous: [], sourceReferences: [] });
      if (node.type === "project") {
        projects.push(i);
        const id = node.projectId ?? records[i].project_id;
        if (identifier(id)) add(projectsById, id, i);
      } else runs.push(i);
    });

    for (const edge of inputEdges) {
      if (!edge || !identifier(edge.from) || !identifier(edge.to)) continue;
      const id = edgeId(edge.from, edge.to, edge.kind);
      if (!decorations.has(id)) decorations.set(id, edge);
      if (edge.kind !== "membership") continue;
      const from = uniqueKey(edge.from), to = uniqueKey(edge.to);
      if (from < 0 || to < 0 || nodes[from].type !== "project" || nodes[to].type !== "run") continue;
      if (!memberships.has(to)) memberships.set(to, new Set());
      memberships.get(to).add(from);
    }

    const idsByProject = new Map();
    for (const i of runs) {
      const node = nodes[i], row = records[i];
      const projectId = node.projectId ?? row.project_id ?? row.project?.project_id;
      let candidates;
      if (node.projectKey != null) {
        const p = uniqueKey(node.projectKey);
        candidates = p >= 0 && nodes[p].type === "project" ? [p] : [];
      }
      if (projectId != null) {
        const matches = projectsById.get(projectId) || [];
        candidates = candidates ? candidates.filter(p => matches.includes(p)) : matches;
      }
      const owners = memberships.get(i);
      if (owners) candidates = candidates ? candidates.filter(p => owners.has(p)) : [...owners];
      if (!candidates) candidates = projects.length === 1 ? projects : [];
      if (candidates.length === 1 && keys.get(nodes[candidates[0]].key).length === 1) {
        project[i] = candidates[0];
        if (!idsByProject.has(project[i])) idsByProject.set(project[i], new Map());
        if (identifier(runIds[i])) add(idsByProject.get(project[i]), runIds[i], i);
      } else reasons[i] = candidates.length > 1 ? "ambiguous-project" : "unknown-project";
      if (keys.get(node.key).length !== 1) reasons[i] = "duplicate-key";
      else if (!identifier(runIds[i])) reasons[i] = "missing-run-id";
    }

    function resolve(i, id) {
      const matches = idsByProject.get(project[i])?.get(id) || [];
      if (!matches.length) return { status: "missing", index: -1 };
      if (matches.length !== 1 || keys.get(nodes[matches[0]].key).length !== 1) {
        return { status: "ambiguous", index: -1 };
      }
      return { status: matches[0] === i ? "self" : "resolved", index: matches[0] };
    }
    function diagnose(node, id, status) {
      if (status === "missing") node.missing.push(id);
      if (status === "ambiguous") node.ambiguous.push(id);
    }

    for (const i of runs) {
      const node = nodes[i], id = records[i].parent_run_id;
      node.parentRunId = identifier(id) ? id : null;
      if (id == null || id === "") continue;
      if (!identifier(id)) { reasons[i] ||= "invalid-parent"; continue; }
      const match = resolve(i, id);
      diagnose(node, id, match.status);
      if (reasons[i]) continue;
      if (match.status === "resolved") parent[i] = match.index;
      else reasons[i] = { missing: "missing-parent", ambiguous: "ambiguous-parent", self: "self-parent" }[match.status];
    }

    // In a functional graph each node has at most one parent. Walk each node
    // once, cutting EVERY cycle member rather than an order-dependent last edge.
    const state = new Uint8Array(count), position = new Int32Array(count).fill(-1);
    for (const start of runs) {
      if (state[start]) continue;
      const path = [];
      let current = start;
      while (current >= 0 && state[current] === 0) {
        state[current] = 1; position[current] = path.length;
        path.push(current); current = parent[current];
      }
      if (current >= 0 && state[current] === 1) {
        for (let k = position[current]; k < path.length; k++) {
          const i = path[k]; parent[i] = -1; reasons[i] = "cycle";
        }
      }
      for (const i of path) state[i] = 2;
    }

    const forest = [...projects];
    for (const i of runs) {
      const p = parent[i] >= 0 ? parent[i] : project[i];
      if (p >= 0) { children[p].push(i); nodes[i].treeParentKey = nodes[p].key; }
      else forest.push(i);
    }
    forest.sort(compare);
    for (const siblings of children) siblings.sort(compare);

    // Preorder establishes depths/ancestry; postorder centers internal nodes.
    // Explicit stacks keep even very long chains independent of the JS stack.
    let nextRow = 0;
    const first = new Float64Array(count), last = new Float64Array(count);
    const stack = forest.slice().reverse().map(index => ({ index, exit: false }));
    while (stack.length) {
      const { index: i, exit } = stack.pop(), node = nodes[i];
      if (exit) {
        const descendants = children[i];
        first[i] = first[descendants[0]];
        last[i] = last[descendants[descendants.length - 1]];
        node.treeRow = (first[i] + last[i]) / 2;
        continue;
      }
      if (node.type === "run") {
        const p = parent[i];
        if (p >= 0) {
          node.depth = nodes[p].depth + 1;
          node.iterationDepth = nodes[p].iterationDepth + 1;
          if (!reasons[i] && nodes[p].ancestryStatus === "unknown") reasons[i] = "ancestor-unknown";
        }
        node.ancestryStatus = reasons[i] ? "unknown" : p >= 0 ? "known" : "initial";
        node.ancestryReason = reasons[i];
      }
      if (!children[i].length) {
        first[i] = last[i] = node.treeRow = nextRow++;
      } else {
        stack.push({ index: i, exit: true });
        for (let k = children[i].length - 1; k >= 0; k--) stack.push({ index: children[i][k], exit: false });
      }
    }

    const edges = [], emitted = new Set();
    function emit(from, to, kind) {
      if (keys.get(nodes[from].key).length !== 1 || keys.get(nodes[to].key).length !== 1) return;
      const id = edgeId(nodes[from].key, nodes[to].key, kind);
      if (emitted.has(id)) return;
      emitted.add(id);
      edges.push({ ...decorations.get(id), from: nodes[from].key, to: nodes[to].key, kind });
    }
    // Stable edge order is independent of node and input edge ordering.
    for (const i of runs.slice().sort(compare)) {
      if (parent[i] >= 0) emit(parent[i], i, "run-parent");
      else if (project[i] >= 0) emit(project[i], i, "membership");
      const sources = records[i].source_run_ids;
      for (const id of [...new Set(Array.isArray(sources) ? sources.filter(identifier) : [])].sort(compareText)) {
        const match = resolve(i, id);
        diagnose(nodes[i], id, match.status);
        nodes[i].sourceReferences.push({ runId: id, status: match.status,
          sourceKey: match.status === "resolved" ? nodes[match.index].key : null });
        if (match.status === "resolved") emit(match.index, i, "run-source");
      }
      nodes[i].missing = [...new Set(nodes[i].missing)];
      nodes[i].ambiguous = [...new Set(nodes[i].ambiguous)];
    }
    return { nodes, edges };
  }

  const api = Object.freeze({ build });
  root.RSILineage = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
