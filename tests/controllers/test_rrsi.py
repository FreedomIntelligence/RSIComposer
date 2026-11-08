"""Behavioral regression for fixed-weight RRSI search, not API definitions."""

import unittest
from copy import deepcopy

from rsisys.api import DecisionInput, ExecutionStatus, OperationResult, RRSIController
from rsisys.controllers.rrsi import adjudicate, calibrated_delta, edit_budget


class Researcher:
    def __init__(self):
        self.inputs = []
        self.records = []

    def propose(self, harness, measurement, **options):
        self.inputs.append((harness, measurement, options))
        return [
            {
                "harness": f"candidate-{options['round_number']}",
                "components": ["control_flow"],
            }
        ]

    def record(self, record):
        self.records.append(record)


class RRSITests(unittest.TestCase):
    def test_noisy_gain_cannot_bypass_cost_rule_and_best_score_floor(self):
        parent = {"score": 0.5, "cost": 100}

        def judge(score, cost, components=()):
            return adjudicate(
                {"score": score, "cost": cost, "components": components},
                parent,
                0.6,
                0.05,
                {},
            )

        self.assertEqual(judge(0.54, 1)["reason"], "stability_floor")
        self.assertFalse(judge(0.56, 1000)["admissible"])
        self.assertTrue(judge(0.6, 200)["admissible"])
        self.assertFalse(
            adjudicate(
                {"score": 0.5, "cost": 100, "components": ["prompt"]},
                parent,
                0.5,
                0.05,
                {},
            )["admissible"]
        )
        self.assertTrue(
            adjudicate(
                {"score": 0.5, "cost": 100, "components": ["memory"]},
                parent,
                0.5,
                0.05,
                {},
            )["admissible"]
        )
        self.assertFalse(
            adjudicate(
                {"score": 0.5, "cost": 100, "components": ["memory"]},
                parent,
                0.5,
                0.05,
                {"memory": 1},
            )["admissible"]
        )

    def test_annealing_and_bootstrap_use_unchanged_trials(self):
        self.assertEqual([edit_budget(t, 4) for t in range(4)], [4, 4, 3, 2])
        self.assertEqual(calibrated_delta([[1, 1], [0, 0]]), 0)
        self.assertGreater(calibrated_delta([[1, 0], [0, 1]]), 0)
        self.assertEqual(calibrated_delta([[1, 0]]), calibrated_delta([[1, 0]]))

    def test_selected_harness_is_parent_next_round_and_heldout_is_final_only(self):
        researcher = Researcher()
        controller = RRSIController(
            initial_harness="seed",
            search_data="search",
            heldout_data="heldout",
            researcher=researcher,
            iterations=2,
            delta=0,
        )
        action = controller.decide(DecisionInput("test"))
        calls = []
        while action.kind != "finish":
            call = action.operation
            calls.append(call)
            score = 0.5 if call.arguments["harness"] == "seed" else 0.75
            action = controller.decide(
                DecisionInput(
                    "test",
                    last_operation=call,
                    feedback=OperationResult(
                        "measured",
                        (),
                        False,
                        data={
                            "request": deepcopy(call.arguments),
                            "harness": call.arguments["harness"],
                            "score": score,
                            "cost": 100,
                            "rewards": [[1, 0], [1, 0]],
                        },
                    ),
                )
            )
        self.assertEqual(
            [c.arguments["split"] for c in calls], ["search"] * 3 + ["heldout"] * 2
        )
        self.assertEqual([r[0] for r in researcher.inputs], ["seed", "candidate-0"])
        self.assertEqual(controller.harness, "candidate-0")
        self.assertEqual(len(controller.history), 2)
        self.assertEqual(len(controller.heldout_results), 2)
        self.assertNotIn("heldout", repr(researcher.inputs))

    def test_missing_stale_and_failed_receipts_do_not_advance_search(self):
        controller = RRSIController(
            initial_harness="seed",
            search_data="search",
            heldout_data="heldout",
            researcher=Researcher(),
        )
        call = controller.decide(DecisionInput("test")).operation
        with self.assertRaises(RuntimeError):
            controller.decide(DecisionInput("test"))
        with self.assertRaises(ValueError):
            controller.decide(
                DecisionInput(
                    "test",
                    last_operation=call,
                    feedback=OperationResult("bad", (), False, data={"request": {}}),
                )
            )
        result = controller.decide(
            DecisionInput(
                "test",
                last_operation=call,
                feedback=OperationResult(
                    "failed",
                    (),
                    False,
                    ExecutionStatus.FAILED,
                    "offline",
                    data={"request": call.arguments},
                ),
            )
        )
        self.assertEqual(result.kind, "finish")
        self.assertIsNone(controller.incumbent)

    def test_bad_config_fails_before_resources_are_created(self):
        base = {
            "initial_harness": "seed",
            "search_data": "search",
            "heldout_data": "heldout",
        }
        for bad in (
            {"trials": True},
            {"b_min": 5},
            {"heldout_data": "search"},
            {"delta": float("nan")},
            {"unknown": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RRSIController.validate_config({**base, **bad})
