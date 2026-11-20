"""LLM analyst/proposer/critic components for the RRSI controller.

All roles share the configured frozen policy model. Only evolve-set traces are
visible. This adapter emits complete Python files rather than git tool edits.
"""

import difflib
import json
import subprocess
from pathlib import Path

from rsisys.algorithms.rrsi.controller import COMPONENTS, STRUCTURAL
from rsisys.api import ArtifactRef, file_hash as digest, sandbox_step, validate_source

INTERFACE = """The harness is Python defining step(problem: str, history: list[dict]) -> dict.
The driver calls it in a fresh isolated process before every policy request.
Return {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}
to call the fixed policy, or {"answer": str} to finish. Each history entry contains
content, tokens, prompt_tokens, completion_tokens, endpoint and finish_reason.
The sandbox has stdlib, no network, filesystem datasets, grader, model access or
persistent state. All state must be recoverable from the current trial history.
At most 3 model calls are allowed. Always terminate. Do not change model,
max_tokens, sampling parameters, driver, grader or these limits.
You may implement bounded verification/revision, branching from trial history,
context organization, output normalization and generic procedural skills.
No benchmark names, question literals, cached answers or task-specific constants.
Final answers must be LaTeX boxed mathematical expressions."""

ROLE_SCHEMAS = {
    "proposer": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {"type": "string"},
            "summary": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "component": {"type": "string", "enum": sorted(COMPONENTS)},
                        "hypothesis": {"type": "string"},
                    },
                    "required": ["component", "hypothesis"],
                },
            },
        },
        "required": ["source", "summary", "edits"],
    },
    "critic": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "reject"]},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "reasons"],
    },
}


def parse_object(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


class HarnessResearcher:
    def __init__(self, context, config, *, pool=None, executor=sandbox_step):
        allowed = {
            "endpoints",
            "model",
            "max_tokens",
            "timeout",
            "repairs",
            "structured_outputs",
        }
        if set(config) - allowed:
            raise ValueError("Unknown researcher option")
        self.root = Path(context.data_dir) / "rrsi-research"
        # A resumed run rebuilds the researcher over its own existing directory.
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, "restoring", False)))
        self.pool = pool or context.models.pool(
            {**config, "max_tokens": config.get("max_tokens", 6144),
             "temperature": 0.5, "timeout": config.get("timeout", 240)}, purpose="researcher")
        self.executor, self.config = executor, config
        self.artifacts = context.artifacts
        self.sequence = 0

    # File names under root derive from sequence, round, candidate and repair. After a stop
    # inside propose(), the restored sequence reuses those names: files the interrupted
    # attempt wrote are overwritten or stay unreferenced by the restored controller.
    def state(self):
        return {"sequence": self.sequence}

    def restore(self, state):
        self.sequence = state["sequence"]

    def _ask(self, role, prompt):
        schema = (
            ROLE_SCHEMAS.get(role)
            if self.config.get("structured_outputs", True)
            else None
        )
        reply = self.pool.complete(
            [
                {
                    "role": "system",
                    "content": "You are the "
                    + role
                    + " in RRSI harness-only recursive improvement. "
                    + INTERFACE,
                },
                {"role": "user", "content": prompt},
            ],
            seed=7300 + self.sequence,
            max_tokens=2048
            if role == "analyst"
            else 1024
            if role == "critic"
            else self.config.get("max_tokens", 6144),
            response_schema=schema,
        )
        (self.root / f"model-{self.sequence:03d}-{role}.json").write_text(
            json.dumps(
                {"prompt": prompt, "reply": reply, "response_schema": schema},
                ensure_ascii=False,
                indent=2,
            )
        )
        self.sequence += 1
        return reply["content"]

    def propose(
        self,
        harness,
        measurement,
        *,
        round_number,
        budget,
        count,
        history,
        counts,
        stalled,
        prune_window,
    ):
        if digest(harness) != measurement["harness_sha256"]:
            raise ValueError("Incumbent harness changed")
        traces = json.loads(self.artifacts.read(ArtifactRef(**measurement["traces"])))["records"]
        failures = [r for r in traces if not r["correct"]][:4]
        successes = [r for r in traces if r["correct"]][:2]
        evidence = [
            {
                **{
                    k: r[k]
                    for k in ("id", "question", "correct", "calls", "tokens", "error")
                },
                "completion": (r["completion"] or "")[-2500:],
            }
            for r in failures + successes
        ]
        source = Path(harness).read_text()
        recent = history[-prune_window:]
        # Report and history contain no heldout measurements or gold solutions.
        report = self._ask(
            "analyst",
            "Analyze general failure modes and wasted policy tokens. Diagnose failure AND successful behavior.\n"
            + json.dumps(
                {
                    "score": measurement["score"],
                    "cost": measurement["cost"],
                    "examples": evidence,
                    "recent_history": recent,
                },
                ensure_ascii=False,
            )
            + "\nCurrent source:\n"
            + source
            + "\nSuggest attributable general mechanisms, identify low-yield machinery to remove. Do not propose question-specific fixes.",
        )
        results = []
        for index in range(count):
            unused = sorted(STRUCTURAL - set(counts))
            explore = bool(stalled and unused and index == count - 1)
            prompt = (
                "Propose one independent candidate derived from the incumbent. Return STRICT JSON only with keys "
                '"source" (full Python source), "edits" (list of {"component": tag, "hypothesis": text}), "summary". '
                f"Use 1..{budget} independent edits, tags from {sorted(COMPONENTS)}. "
                "Keep parent mechanisms unless evidence supports pruning. Declare every independent change. "
                "Do not add general commentary outside JSON.\n"
                + (
                    "This exploration slot must implement one genuinely unused structural component: "
                    + str(unused)
                    + ".\n"
                    if explore
                    else ""
                )
                + f"Variant {index}; round {round_number}.\nAnalysis:\n"
                + report
                + "\nParent source:\n"
                + source
            )
            objections = ""
            for attempt in range(self.config.get("repairs", 2) + 1):
                proposal = None
                review = None
                try:
                    proposal = parse_object(self._ask("proposer", prompt + objections))
                    candidate = proposal["source"]
                    validate_source(candidate)
                    edits = proposal["edits"]
                    if (
                        candidate == source
                        or not isinstance(edits, list)
                        or not 1 <= len(edits) <= budget
                    ):
                        raise ValueError("Empty proposal or edit-budget violation")
                    if any(
                        not isinstance(e, dict)
                        or e.get("component") not in COMPONENTS
                        or not isinstance(e.get("hypothesis"), str)
                        or not e["hypothesis"].strip()
                        for e in edits
                    ):
                        raise ValueError("Invalid edit declarations")
                    diff = "".join(
                        difflib.unified_diff(
                            source.splitlines(True),
                            candidate.splitlines(True),
                            fromfile="parent.py",
                            tofile="candidate.py",
                        )
                    )
                    review = parse_object(
                        self._ask(
                            "critic",
                            "Independently screen this diff. Reject leakage, task memorization, no-op/dead mechanisms, "
                            "undeclared bundled edits, cross-task runtime memory, grader access, unbounded work, or fictitious structural labels. "
                            "Prompt-only changes must be labeled prompt. The critic must assess whether declared changes actually exist. "
                            'Return STRICT JSON {"verdict":"accept"|"reject","reasons":[str]}.\n'
                            + json.dumps(
                                {
                                    "summary": proposal.get("summary"),
                                    "edits": edits,
                                    "reserved_structural_exploration": explore,
                                    "unused": unused,
                                }
                            )
                            + "\n"
                            + diff,
                        )
                    )
                    if review.get("verdict") != "accept":
                        raise ValueError(
                            "Critic rejected: " + str(review.get("reasons"))
                        )
                    if explore and not any(e["component"] in unused for e in edits):
                        raise ValueError("Missing unused structural component")
                    directory = (
                        self.root
                        / f"round-{round_number}"
                        / f"candidate-{index}-repair-{attempt}"
                    )
                    # Exists only when a resumed run repeats an interrupted proposal round.
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "harness.py"
                    path.write_text(candidate)
                    # Constructor/control-flow smoke; no real benchmark answers are exposed.
                    for trial_history in (
                        [],
                        [
                            {
                                "content": r"Checked derivation. \boxed{17}",
                                "tokens": 5,
                                "prompt_tokens": 2,
                                "completion_tokens": 3,
                                "endpoint": "local",
                                "finish_reason": "stop",
                            }
                        ],
                    ):
                        self.executor(
                            path,
                            "Solve a new mathematical problem using general reasoning.",
                            trial_history,
                        )
                    record = {
                        "harness": str(path),
                        "harness_sha256": digest(path),
                        "components": [e["component"] for e in edits],
                        "edits": edits,
                        "summary": proposal.get("summary", ""),
                        "critic": review,
                    }
                    (directory / "proposal.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )
                    results.append(record)
                    break
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    SyntaxError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    objections = (
                        "\nYour last candidate was rejected. Repair it; preserve the intended independent mechanisms. Reason: "
                        + str(exc)
                    )
                    (
                        self.root
                        / f"rejection-r{round_number}-v{index}-a{attempt}.json"
                    ).write_text(
                        json.dumps(
                            {"error": str(exc), "proposal": proposal, "review": review},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
        return results

    def record(self, record):
        (self.root / f"round-{record['round']}-selection.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
