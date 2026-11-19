"""RRSI harness-only search (arXiv:2609.24972), with fixed model weights.

The controller owns analysis/proposal/critique and Algorithm 2 selection. A Lab
measures immutable harnesses; it never chooses the incumbent. This small-scale
port uses one task domain, within-task bootstrap calibration and bounded search.
"""

import math
import random
import re
import statistics
from copy import deepcopy
from pathlib import Path

from rsisys.api import Decision, ExecutionStatus, OperationCall, file_hash as digest

STRUCTURAL = {"client_tool", "skill", "memory", "subagent"}
COMPONENTS = {
    "prompt",
    "control_flow",
    "config",
    "output_plumbing",
    "context_mgmt",
    *STRUCTURAL,
}


def edit_budget(t, horizon, minimum=1, maximum=4):
    return math.ceil(
        round(
            minimum + (maximum - minimum) * (1 + math.cos(math.pi * t / horizon)) / 2, 9
        )
    )


def calibrated_delta(rewards, z=2.0, reps=2000):
    """Official single-evaluation bootstrap, uniform trial/task weights."""
    rng = random.Random(7)
    estimates = []
    for _ in range(reps):
        estimates.append(
            statistics.mean(rng.choice(group) for group in rewards for _ in group)
        )
    return round(z * math.sqrt(2) * statistics.pstdev(estimates), 6)


def adjudicate(
    candidate,
    incumbent,
    best_score,
    delta,
    counts,
    *,
    beta0=0.10,
    beta1=40.0,
    w_s=100.0,
    w_c=15.0,
    w_n=0.5,
):
    """Algorithm 2: stability floor, relative token cost, structural novelty."""
    score, cost = candidate["score"], candidate["cost"]
    gain = score - incumbent["score"]
    relative = (
        (cost - incumbent["cost"]) / incumbent["cost"]
        if cost and incumbent["cost"]
        else 0.0
    )
    novelty = sum(
        c in STRUCTURAL and counts.get(c, 0) == 0 for c in set(candidate["components"])
    )
    rule = (
        relative <= beta0 + beta1 * gain
        if gain > delta
        else w_s * gain - w_c * relative + w_n * novelty > 0
    )
    floor = score >= best_score - delta
    valid = candidate.get("harness_errors", 0) <= 0.15 * candidate.get("samples", 1)
    return {
        "admissible": bool(floor and rule and valid),
        "score": score,
        "cost": cost,
        "delta_score": gain,
        "delta_cost": relative,
        "novelty": novelty,
        "reason": (
            "invalid_trials"
            if not valid
            else "admissible"
            if floor and rule
            else ("stability_floor" if not floor else "cost_rule")
        ),
    }


class RRSIController:
    def __init__(
        self,
        *,
        initial_harness,
        search_data,
        heldout_data,
        researcher,
        metric,
        lab="harness",
        iterations=3,
        candidates=2,
        trials=2,
        b_min=1,
        b_max=4,
        stall_window=3,
        prune_window=4,
        delta=None,
    ):
        options = dict(
            initial_harness=initial_harness,
            search_data=search_data,
            heldout_data=heldout_data,
            lab=lab,
            iterations=iterations,
            candidates=candidates,
            trials=trials,
            b_min=b_min,
            b_max=b_max,
            stall_window=stall_window,
            prune_window=prune_window,
            delta=delta,
        )
        self.validate_config(options)
        self.options, self.researcher, self.metric = options, researcher, metric
        if metric.direction != 'maximize':
            raise ValueError('RRSI bootstrap calibration requires a maximizing success-rate metric')
        self.harness = initial_harness
        self.incumbent = None
        self.best_score = 0.0
        self.delta = delta
        self.round = 0
        self.pending = None
        self.stage = "baseline"
        self.counts = {}
        self.history = []
        self.round_candidates = []
        self.proposals = []
        self.index = 0
        self.registered = {}
        self.deferred = None
        self.stop_reason = None

    _FIELDS = ("harness", "incumbent", "best_score", "delta", "round", "pending", "stage", "counts",
               "history", "round_candidates", "proposals", "index", "registered", "deferred", "stop_reason")

    def state(self):
        return {**{name: deepcopy(getattr(self, name)) for name in self._FIELDS},
                "researcher": self.researcher.state()}

    def restore(self, state):
        if set(state) != {*self._FIELDS, "researcher"}:
            raise ValueError("Unexpected RRSI controller state")
        # Proposed harness files are run-owned; a changed file would be measured as something else.
        for proposal in state["proposals"]:
            if digest(proposal["harness"]) != proposal["harness_sha256"]:
                raise ValueError(f"RRSI candidate harness changed while the run was stopped: {proposal['harness']}")
        for name in self._FIELDS:
            setattr(self, name, deepcopy(state[name]))
        self.researcher.restore(state["researcher"])

    @staticmethod
    def validate_config(config):
        allowed = {
            "initial_harness",
            "search_data",
            "heldout_data",
            "lab",
            "iterations",
            "candidates",
            "trials",
            "b_min",
            "b_max",
            "stall_window",
            "prune_window",
            "delta",
            "researcher",
        }
        if set(config) - allowed:
            raise ValueError("Unknown rrsi option")
        for key in ("initial_harness", "search_data", "heldout_data"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"{key} must be a nonempty path")
        if (
            Path(config["search_data"]).resolve()
            == Path(config["heldout_data"]).resolve()
        ):
            raise ValueError("RRSI requires independent heldout data")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", config.get("lab", "harness")):
            raise ValueError("Invalid Lab alias")
        for key, default in (
            ("iterations", 3),
            ("candidates", 2),
            ("trials", 2),
            ("b_min", 1),
            ("b_max", 4),
            ("stall_window", 3),
            ("prune_window", 4),
        ):
            if (
                type(config.get(key, default)) is not int
                or config.get(key, default) < 1
            ):
                raise ValueError(f"{key} must be a positive integer")
        if config.get("b_min", 1) > config.get("b_max", 4):
            raise ValueError("b_min exceeds b_max")
        delta = config.get("delta")
        if delta is not None and (
            type(delta) not in (int, float)
            or not math.isfinite(delta)
            or not 0 <= delta <= 1
        ):
            raise ValueError("delta must be a finite fraction")

    def required_profiles(self):
        return {self.options['lab']: 'artifact-search/v1'}

    def _request(self, harness, dataset, split):
        if harness not in self.registered:
            self.deferred = {'harness': harness, 'dataset': dataset, 'split': split}
            self.pending = OperationCall(self.options['lab'] + '.register', {'candidate': {'harness': harness}})
        else:
            self.pending = OperationCall(self.options['lab'] + '.evaluate', {
                'candidate_id': self.registered[harness], 'split': split, 'indices': None,
                'trials': self.options['trials'], 'round': self.round, 'baseline': self.stage == 'baseline'})
        return Decision.invoke(deepcopy(self.pending))

    def _consume(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        if (feedback is None or feedback.call_id != self.pending.call_id):
            raise RuntimeError("RRSI requires the matching terminal receipt")
        if feedback.data.get("request") != self.pending.arguments:
            raise ValueError("RRSI receipt request mismatch")
        if feedback.status != ExecutionStatus.COMPLETED:
            self.stop_reason = f"RRSI stopped after failed measurement: {feedback.error or feedback.status.value}"
            return
        data = deepcopy(feedback.data)
        if self.pending.name.endswith('.register'):
            harness = self.deferred['harness']
            if data.get('candidate_id') != digest(harness):
                raise ValueError('RRSI registration identity mismatch')
            self.registered[harness] = data['candidate_id']
            self.pending = None
            return
        if self.pending.name.endswith('.select'):
            self.stage, self.pending = 'done', None
            return
        data['score'] = self.metric.value(data)
        if self.registered.get(data.get('harness')) != self.pending.arguments['candidate_id']:
            raise ValueError("RRSI measurement describes a different harness")
        for name in ("score", "cost"):
            if type(data.get(name)) not in (int, float) or not math.isfinite(
                data[name]
            ):
                raise ValueError("Non-finite or missing RRSI measurement")
        if not 0 <= data["score"] <= 1 or data["cost"] < 0:
            raise ValueError("Invalid score or token cost")
        if self.stage == "baseline":
            self.incumbent = data
            self.best_score = data["score"]
            if self.delta is None:
                self.delta = calibrated_delta(data["rewards"])
            self.stage = "search"
        elif self.stage == "candidate":
            proposal = self.proposals[self.index]
            self.round_candidates.append({**data, **proposal})
            self.index += 1
            self.stage = "search"
        self.pending = None

    def _select(self):
        decisions = []
        for candidate in self.round_candidates:
            verdict = adjudicate(
                candidate, self.incumbent, self.best_score, self.delta, self.counts
            )
            decisions.append({"harness": candidate["harness"], **verdict})
        accepted = [
            c for c, d in zip(self.round_candidates, decisions) if d["admissible"]
        ]
        winner = max(accepted, key=lambda c: c["score"]) if accepted else None
        before = self.harness
        if winner:
            self.harness, self.incumbent = winner["harness"], winner
            self.best_score = max(self.best_score, winner["score"])
            for component in set(winner["components"]):
                self.counts[component] = self.counts.get(component, 0) + 1
        self.history.append(
            {
                "round": self.round,
                "parent": before,
                "winner": self.harness,
                "best_score": self.best_score,
                "delta": self.delta,
                "decisions": decisions,
                "screened": deepcopy(self.proposals),
            }
        )
        self.researcher.record(self.history[-1])
        self.round += 1
        self.proposals, self.round_candidates, self.index = [], [], 0

    def decide(self, window):
        if self.stop_reason:
            return Decision("finish", reason=self.stop_reason)
        if self.pending is not None:
            self._consume(window)
        if self.stop_reason:
            return Decision("finish", reason=self.stop_reason)
        if self.deferred is not None:
            arguments, self.deferred = self.deferred, None
            return self._request(**arguments)
        if self.stage == "baseline":
            return self._request(self.harness, self.options["search_data"], "search")
        if self.stage == "search":
            while self.round < self.options["iterations"]:
                if not self.proposals:
                    recent = self.history[-self.options["stall_window"] :]
                    stalled = len(recent) >= self.options["stall_window"] and all(
                        r["parent"] == r["winner"] for r in recent
                    )
                    self.proposals = self.researcher.propose(
                        self.harness,
                        self.incumbent,
                        round_number=self.round,
                        budget=edit_budget(
                            self.round,
                            self.options["iterations"],
                            self.options["b_min"],
                            self.options["b_max"],
                        ),
                        count=self.options["candidates"],
                        history=self.history,
                        counts=self.counts,
                        stalled=stalled,
                        prune_window=self.options["prune_window"],
                    )
                if self.index < len(self.proposals):
                    self.stage = "candidate"
                    return self._request(
                        self.proposals[self.index]["harness"],
                        self.options["search_data"],
                        "search",
                    )
                self._select()
            self.stage = 'select'
        if self.stage == 'select':
            self.pending = OperationCall(self.options['lab'] + '.select', {
                'candidate_id': self.registered[self.harness]})
            return Decision.invoke(deepcopy(self.pending))
        return Decision(
            "finish",
            reason="RRSI harness-only rounds completed; host sealed evaluation follows",
        )


def build_rrsi(context, config):
    from rsisys.algorithms.rrsi.researcher import HarnessResearcher

    options = dict(config)
    researcher = HarnessResearcher(context, options.pop("researcher"))
    return RRSIController(researcher=researcher,
        metric=context.lab_catalog.primary_metric(options.get('lab', 'harness')), **options)
