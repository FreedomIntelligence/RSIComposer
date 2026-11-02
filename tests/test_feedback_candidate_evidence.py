"""Fixed call48 attribution regression; fixture inference, no training/services.

The exact receipt, supplied evaluations and response are pinned below. History
and grouping scaffolding are reduced to the identities cited by that response.
Tests never require the ignored live run directory.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.feedback.evidence import candidate_evidence
from rsisys.loops.decision_making.feedback.learning import (
    FeedbackLearning, SerializedLearningBackend, _evidence_packet, validate_proposals,
)
from rsisys.loops.decision_making.feedback.service import FeedbackLearningService
from rsisys.runtime.json_tools import result_id
from tests.test_feedback_learning_integration import ImmediateExecutor


CALL48 = json.loads(r'''
{
  "raw": {
    "receipt": {
      "directive": {
        "allowed_actions": [
          "echo"
        ],
        "directive_id": "0e15089b-f5dc-4bbc-9480-b5443cb4f44f",
        "input_refs": [
          {
            "artifact_id": "070a7076d7854c879f4375b2ff5f2aec",
            "digest": "d695dae747e3d9171927e3f2e18b5b3c15f647fbcf05c50303b68b561d839a62",
            "kind": "training_data",
            "uri": "artifact://070a7076d7854c879f4375b2ff5f2aec",
            "version": "1"
          }
        ],
        "max_steps": 4,
        "mode": "train",
        "objective": "Autonomously research and improve this model to increase both MedXpertQA and MedQA accuracy through evidence-grounded research. MedXpertQA has ten answer labels A-J; MedQA has four A-D. Grading requires exactly one legal label, with no period, prefix or explanation. Prompt changes apply to both benchmarks. The initial baseline has ALREADY been measured; its results and evidence IDs are below. Read its evidence with read_artifact if useful. The initial system is not a candidate. evaluate accepts only a candidate_id listed in context.candidates; a sole candidate and the frozen evaluation protocol default automatically, but multiple candidates require an explicit ID. Only evaluate and apply_candidate require an existing candidate. Creating a candidate does NOT require an existing candidate: configure_prompt proposes a new prompt candidate and train produces a new SFT candidate from the fixed initializer and selected training data. When context.candidates is empty, you can still use either creation directive, inspect or prepare data, define recipes, and read or update research notes, according to the available capabilities. An empty candidate list does not mean that research is blocked. Use returned candidate IDs after creation; do not invent IDs or repeat baseline measurement. You choose the research steps: change training data/recipe or prompt, evaluate and apply an accepted candidate. Choose data construction, SFT recipes, inference prompts or combinations according to your hypotheses. No particular method or training submission is required. Compare candidates using frozen development evidence. A few negative results do not exhaust the search; seed changes alone are not mechanism improvements. Application follows frozen selection_rule=pareto. The default pareto rule requires at least one accuracy improvement over the current incumbent and no decreases. Tradeoff candidates remain available for comparison. Only method=sft and restart from the fixed initializer are installed; no real RL. Use define_recipe for controlled steps/lr/seed/weight_decay/warmup_ratio, preview_training_data (at most 8 rows), choose_training_data for a subset, or prepare_data for validated domain JSONL. read_artifact returns summaries for training data. Never use development or protected examples for training. For train explicitly select input_ids of training_data only, apply_result=false, and omit review_after_seconds: training owns the allocated GPUs until the processes exit. configure_prompt/evaluate/apply_candidate require input_ids=[]. Default recipe_ref=medical-native:3a72b5bff67bf011dec4f0a3456a3ad29507b086155fa8409c51e4df5f547643; default steps=32, lr=0.0001. Frozen train artifact_id=070a7076d7854c879f4375b2ff5f2aec. Evaluation protocol_ref=medical-exact-v1:7108f2fbcdc4a8a620d7122a861bf75b67a891a3551573637e7a9012714e464b. Fixed initial anchor: {\"candidate_id\": \"initial\", \"protocol_ref\": \"medical-exact-v1:7108f2fbcdc4a8a620d7122a861bf75b67a891a3551573637e7a9012714e464b\", \"metrics\": {\"medxpertqa.accuracy\": 0.1953125, \"medxpertqa.invalid_format\": 0.0, \"medxpertqa.macro_f1\": 0.18686937161528813, \"score\": 0.1953125, \"medqa.accuracy\": 0.71875, \"medqa.invalid_format\": 0.0, \"medqa.macro_f1\": 0.7107976268951879}, \"evidence_refs\": [\"84713912d8b84e9c868b7d8f5b886442\"], \"simulated\": false} Continuous research: the host owns budgets and stopping. Choose the next useful hypothesis and executable action using current evidence and capabilities. A completed or rejected attempt is not the end of research. Do not save an action for a future cycle; there is no model-controlled cycle or resource reset. A few negative results do not establish a local optimum or prove overfitting. If no useful next step is supported, wait with a concrete blocker or unresolved question; do not repeat completed queries without new information. No particular experiment or positive outcome is required.\nReturn one JSON object with unique keys in every object, including nested objects. Canonical outer envelopes are {\"kind\":\"invoke\",\"operation\":\"<name from context.operations>\",\"arguments\":{}} or {\"kind\":\"explore\",\"mode\":\"<mode from context.directive_modes>\",\"strategy\":\"<your strategy>\",\"input_ids\":[],\"arguments\":{}}. Fill arguments according to the chosen operation or directive input_schema. Put operation parameters (including view, kind, limit) inside arguments, never in the top-level kind; top-level kind selects the decision envelope. train requires nonempty top-level input_ids selecting available training_data artifacts. Omit arguments.origin: initialization is host-owned and origin is not an accepted model argument. The executor binds the current experimental parent and fixed initial model automatically, using restart without a checkpoint. Training data belongs only in input_ids. Choose your own available action and research steps according to the control contract; these envelopes are syntax templates, not a required workflow.",
        "payload": {
          "apply_result": false,
          "base": null,
          "method": "sft",
          "origin": {
            "checkpoint": null,
            "initial_model": {
              "backend_id": "medical-local",
              "model_id": "/home/work/user-job-dir/app/slai/user-xiexinyuan/xinyuan/models/Qwen3.5-9B-c2022362",
              "revision": "local-sha256:b0e29619703862d191fa1b6853f0a242acb2f9831374146f895ef47374463786"
            },
            "parent_system": "7fbe3af34b1cd71b8b488f89e504e9a74418382a2e3fe4b65a311338b96abb10",
            "resume_mode": "restart"
          },
          "recipe_ref": "medical-native:2c747a0287556c275bc33fdf59f0f192fb603bcba2fe6e2e4bcf279139c8a2b3"
        },
        "review_after_seconds": null,
        "simulated": false,
        "strategy": "Train a new SFT candidate from the fixed initializer on the full frozen MedQA artifact using the newly defined 64-step recipe (medical-native:2c747a02, lr=1e-4, no warmup, no weight decay). The current incumbent (7fbe3af3, 32 steps) has MedQA 0.7344 and MedXpertQA 0.2109. The default recipe ran only 32 steps (epoch 0.0625), so the model is far from converged; more training may raise MedXpertQA (the harder 10-option benchmark) without dropping MedQA, satisfying the pareto rule. This is a distinct recipe_ref, not an exact duplicate of prior runs."
      },
      "feedback": {
        "adopted_candidate_ids": [],
        "artifact_refs": [
          {
            "artifact_id": "f9906b36e97b4bd6b93eb4acbe6b5d7b",
            "digest": "2536cb7d364ba2c381c764f165abad258197d1b3021e9f58b83c23a6a93f81c4",
            "kind": "medical_training_evidence",
            "uri": "artifact://f9906b36e97b4bd6b93eb4acbe6b5d7b",
            "version": "1"
          },
          {
            "artifact_id": "f9a509e2b91b4573872040af83fb33c1",
            "digest": "f811bfa6da63c2260e96f6255ed7a9928e29fe89d2db6497860673eb7b09d1fd",
            "kind": "native_model_manifest",
            "uri": "artifact://f9a509e2b91b4573872040af83fb33c1",
            "version": "1"
          },
          {
            "artifact_id": "a0801d8f4eaa4fae8f1586b6cff76cd7",
            "digest": "726dbf1191a623a31b2ac244486c2924e273766897257c9fbe6b5949f5f78eff",
            "kind": "training_evidence",
            "uri": "artifact://a0801d8f4eaa4fae8f1586b6cff76cd7",
            "version": "1"
          }
        ],
        "candidates": [
          {
            "affected_roles": [
              "actor"
            ],
            "base_version": "local-sha256:b0e29619703862d191fa1b6853f0a242acb2f9831374146f895ef47374463786",
            "candidate_id": "f6ce2f04f1d54c4696fa85ac3b3f65ae86a3f1486b17bf73bda6acbeaf93501a",
            "changes": {
              "medical_training": {
                "artifact_id": "f9906b36e97b4bd6b93eb4acbe6b5d7b",
                "digest": "2536cb7d364ba2c381c764f165abad258197d1b3021e9f58b83c23a6a93f81c4",
                "kind": "medical_training_evidence",
                "uri": "artifact://f9906b36e97b4bd6b93eb4acbe6b5d7b",
                "version": "1"
              },
              "model": {
                "artifact_id": "f9a509e2b91b4573872040af83fb33c1",
                "digest": "f811bfa6da63c2260e96f6255ed7a9928e29fe89d2db6497860673eb7b09d1fd",
                "kind": "native_model_manifest",
                "uri": "artifact://f9a509e2b91b4573872040af83fb33c1",
                "version": "1"
              },
              "training_evidence": {
                "artifact_id": "a0801d8f4eaa4fae8f1586b6cff76cd7",
                "digest": "726dbf1191a623a31b2ac244486c2924e273766897257c9fbe6b5949f5f78eff",
                "kind": "training_evidence",
                "uri": "artifact://a0801d8f4eaa4fae8f1586b6cff76cd7",
                "version": "1"
              }
            },
            "parent_system": "7fbe3af34b1cd71b8b488f89e504e9a74418382a2e3fe4b65a311338b96abb10"
          }
        ],
        "directive_id": "0e15089b-f5dc-4bbc-9480-b5443cb4f44f",
        "error": null,
        "evidence_refs": [
          "9f923222-a4c3-46fb-86e7-f05eaad65c01"
        ],
        "execution_metrics": {},
        "jobs": [
          {
            "backend": "medical:1c84ee8f94dde996ac1e7fabbad45b89b169d56fcb7a39d97c18c755f76e4021",
            "job_id": "f6ce2f04f1d54c4696fa85ac3b3f65ae86a3f1486b17bf73bda6acbeaf93501a"
          }
        ],
        "quality_metrics": {},
        "simulated": false,
        "status": "completed",
        "summary": "train completed",
        "training_metrics": {
          "epoch": 0.125,
          "total_flos": 1585273771307616.0,
          "train_loss": 0.23192430805738695,
          "train_runtime": 242.9853,
          "train_samples_per_second": 0.527,
          "train_steps_per_second": 0.263
        }
      }
    },
    "evaluations": [
      {
        "candidate_id": "d113fb9b278d3f4844c419f20d1eb2d9323f10671238f75cc64a3c31fae796ab",
        "directive_id": "cc33fdba-7e90-41b1-991a-9131d6f0f126",
        "evidence_refs": [
          "1e935df0-7eb4-46c6-8ebd-d17482ba0314"
        ],
        "input_refs": [],
        "protocol_ref": "medical-exact-v1:7108f2fbcdc4a8a620d7122a861bf75b67a891a3551573637e7a9012714e464b",
        "quality_metrics": {
          "medqa.accuracy": 0.703125,
          "medqa.invalid_format": 0.0,
          "medqa.macro_f1": 0.6970218295218296,
          "medxpertqa.accuracy": 0.1796875,
          "medxpertqa.invalid_format": 0.0,
          "medxpertqa.macro_f1": 0.1745491537253551,
          "score": 0.1796875
        },
        "simulated": false,
        "status": "completed"
      },
      {
        "candidate_id": "prompt-73a22b79-c7cd-4a08-b8ca-97df617ef46b",
        "directive_id": "ce093845-ab02-4d28-975a-72ccee2835ac",
        "evidence_refs": [
          "dc37c71b-016b-4303-93af-2461cd20a19e"
        ],
        "input_refs": [],
        "protocol_ref": "medical-exact-v1:7108f2fbcdc4a8a620d7122a861bf75b67a891a3551573637e7a9012714e464b",
        "quality_metrics": {
          "medqa.accuracy": 0.703125,
          "medqa.invalid_format": 0.0,
          "medqa.macro_f1": 0.6951219512195121,
          "medxpertqa.accuracy": 0.234375,
          "medxpertqa.invalid_format": 0.0,
          "medxpertqa.macro_f1": 0.23137135278731202,
          "score": 0.234375
        },
        "simulated": false,
        "status": "completed"
      }
    ],
    "history": [
      {
        "record_id": "68838a90-672d-44d6-8acd-c9db3c0716a0",
        "kind": "feedback",
        "task_id": "0e15089b-f5dc-4bbc-9480-b5443cb4f44f",
        "summary": "train completed",
        "simulated": false,
        "source_refs": [],
        "details": {
          "training_metrics": {
            "epoch": 0.125,
            "total_flos": 1585273771307616.0,
            "train_loss": 0.23192430805738695,
            "train_runtime": 242.9853,
            "train_samples_per_second": 0.527,
            "train_steps_per_second": 0.263
          }
        }
      }
    ],
    "research_context": {},
    "grouping_context": {
      "decisions": [
        {
          "decision_id": "f39b807e-2806-4ffb-bd9f-dfbc7536a796",
          "status": "completed"
        },
        {
          "decision_id": "1952c2b8-56f6-44a6-a886-d61e0e5f8b22",
          "status": "completed"
        }
      ],
      "attempts": []
    }
  },
  "response": "{\n  \"proposals\": [\n    {\n      \"kind\": \"memory\",\n      \"content\": \"The SFT candidate (f6ce2f04) trained on MedQA with 64 steps (lr=1e-4) achieved a MedQA accuracy of 0.703125 and a MedXpertQA accuracy of 0.1796875. This represents a decrease in MedQA performance compared to the incumbent (0.7344) and a low performance on MedXpertQA, indicating that the current training recipe and data mix are insufficient for the dual-objective goal.\",\n      \"applicability\": \"Future training runs should consider adjusting the learning rate, increasing training steps, or incorporating MedXpertQA data to improve performance on both benchmarks.\",\n      \"source_refs\": [\n        \"feedback:ed2af009c39fa74e61f151aff510f99e02583cc831523e4b3b9707538acbcd60\",\n        \"history:68838a90-672d-44d6-8acd-c9db3c0716a0\"\n      ],\n      \"validation_plan\": \"Compare future candidate metrics against the incumbent baseline (MedQA 0.7344) and the current candidate's metrics to verify improvement.\"\n    },\n    {\n      \"kind\": \"memory\",\n      \"content\": \"The minimal exact-label prompt (prompt-73a22b79) achieved a MedQA accuracy of 0.703125 and a MedXpertQA accuracy of 0.234375. While the MedQA accuracy remains below the incumbent, the MedXpertQA accuracy is higher than the SFT candidate (0.1796875), suggesting that prompt engineering may be a more effective lever for improving MedXpertQA performance in the short term.\",\n      \"applicability\": \"Prompt optimization should be prioritized for MedXpertQA, while SFT strategies need refinement for MedQA.\",\n      \"source_refs\": [\n        \"feedback:ed2af009c39fa74e61f151aff510f99e02583cc831523e4b3b9707538acbcd60\",\n        \"decision:1952c2b8-56f6-44a6-a886-d61e0e5f8b22\"\n      ],\n      \"validation_plan\": \"Test variations of the exact-label prompt to see if MedXpertQA accuracy can be further increased without degrading MedQA performance.\"\n    }\n  ],\n  \"attempt_updates\": [\n    {\n      \"attempt_id\": null,\n      \"base_revision\": 0,\n      \"title\": \"SFT and Prompt Optimization for Medical QA\",\n      \"summary\": \"This attempt explores two parallel strategies to improve medical QA performance: SFT training on MedQA data and prompt engineering for exact-label responses. The SFT candidate showed a drop in MedQA accuracy compared to the incumbent, while the prompt candidate showed a modest improvement in MedXpertQA accuracy. The next steps should focus on refining the SFT recipe to recover MedQA performance and further optimizing the prompt to boost MedXpertQA accuracy.\",\n      \"member_decision_ids\": [\n        \"f39b807e-2806-4ffb-bd9f-dfbc7536a796\",\n        \"1952c2b8-56f6-44a6-a886-d61e0e5f8b22\"\n      ],\n      \"source_refs\": [\n        \"decision:f39b807e-2806-4ffb-bd9f-dfbc7536a796\",\n        \"decision:1952c2b8-56f6-44a6-a886-d61e0e5f8b22\",\n        \"feedback:ed2af009c39fa74e61f151aff510f99e02583cc831523e4b3b9707538acbcd60\"\n      ]\n    }\n  ]\n}"
}
''')


class Backend:
    supports_response_schema = True

    def __init__(self, output):
        self.output, self.requests = output, []

    def count_tokens(self, request):
        return 100  # Transport fixture, not a claim about a real tokenizer.

    def generate(self, request):
        self.requests.append(request)
        value = self.output(json.loads(request.prompt)) if callable(self.output) else self.output
        text = value if isinstance(value, str) else json.dumps(value)
        return GenerationResponse(text, request.model, simulated=False)


def valid_output(packet):
    source = packet['candidate_evidence']['evaluations'][0]
    claim = {key: deepcopy(source[key]) for key in ('candidate_id', 'protocol_ref', 'simulated')}
    claim.update(evaluation_ref=source['source_ref'], metrics={
        'medqa.accuracy': 0.703125, 'medxpertqa.accuracy': 0.1796875})
    return {'proposals': [{'kind': 'memory', 'content': 'Observed metrics for the cited candidate only.',
        'applicability': 'This candidate and protocol', 'validation_plan': 'Read the original evaluation',
        'source_refs': [source['source_ref']], 'evaluation_claims': [claim]}], 'attempt_updates': []}


class CandidateEvidenceTests(unittest.TestCase):
    def make(self, output=valid_output, *, path=None, schema=True):
        backend = Backend(output)
        backend.supports_response_schema = schema
        learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor(), path=path)
        self.addCleanup(lambda: None if learning._closed else learning.close())
        raw = deepcopy(CALL48['raw'])
        row = {'job_id': 'call48', 'capture_index': 0, 'receipt_digest': result_id(raw['receipt']),
            'model': asdict(ModelRef('fixture', 'call48', 'v1')), 'raw_input': raw,
            'status': 'queued', 'proposals': [], 'error': None, 'simulated': False, 'response': None}
        self.assertTrue(learning._bound_input(row))
        learning._commit({**learning.state, 'jobs': {row['job_id']: row}})
        published = []
        service = FeedbackLearningService(learning, SimpleNamespace(publish=published.append))
        return backend, learning, service, published

    def complete(self, service):
        service.refresh()
        service.refresh()
        return service.learning.state['jobs']['call48']

    def test_exact_call48_response_passed_old_whitelist_but_is_now_rejected_and_not_published(self):
        backend, learning, service, published = self.make(CALL48['response'])
        original = json.loads(CALL48['response'])
        raw = deepcopy(learning.state['jobs']['call48']['raw_input'])
        self.assertEqual(result_id(raw['receipt']),
            'ed2af009c39fa74e61f151aff510f99e02583cc831523e4b3b9707538acbcd60')
        # Establish that the original citations themselves were all admissible.
        refs = learning.state['jobs']['call48']['input']['source_refs']
        self.assertEqual(validate_proposals(json.dumps({'proposals': original['proposals']}), refs),
                         original['proposals'])
        row = self.complete(service)
        self.assertEqual(row['status'], 'failed')
        self.assertIn('Invalid learning proposal fields', row['error'])
        self.assertEqual(row['response']['text'], CALL48['response'])
        self.assertEqual(row['proposals'], [])
        self.assertEqual(learning.attempts()['items'], [])
        self.assertEqual(published, [])
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(row['raw_input'], raw)

    def test_input_separates_trained_candidate_from_prior_evaluation_and_preserves_numbers(self):
        _, learning, _, _ = self.make()
        row = learning.state['jobs']['call48']
        evidence = row['input']['candidate_evidence']
        current = row['raw_input']['receipt']['feedback']['candidates'][0]['candidate_id']
        self.assertTrue(current.startswith('f6ce2f04'))
        self.assertEqual(evidence['current_candidate_ids'], [current])
        self.assertEqual(evidence['candidates'][0], {'candidate_id': current, 'evaluation_refs': [],
            'evaluation_coverage': 'not_observed_in_snapshot'})
        source = evidence['evaluations'][0]
        self.assertTrue(source['candidate_id'].startswith('d113fb9b'))
        self.assertEqual(source['quality_metrics']['medqa.accuracy'], 0.703125)
        self.assertEqual(source['quality_metrics']['medxpertqa.accuracy'], 0.1796875)
        self.assertEqual(source['source_ref'], 'evaluation:' + result_id(row['raw_input']['evaluations'][0]))
        page = learning.read('call48', evidence_path=source['raw_ref'], limit=16)
        self.assertEqual(page['sha256'], result_id(row['raw_input']['evaluations'][0]))
        self.assertEqual(json.loads(page['text']), row['raw_input']['evaluations'][0])

    def test_correct_structured_claim_publishes_with_explicit_prose_limit(self):
        backend, _, service, published = self.make()
        row = self.complete(service)
        self.assertEqual(row['status'], 'completed')
        self.assertTrue(row['published'])
        self.assertEqual(len(published), 1)
        content = json.loads(published[0].content)
        self.assertEqual(content['evaluation_validation'],
                         'structured_attribution_only; prose_remains_unverified')
        self.assertEqual(published[0].status, 'unverified')
        self.assertEqual(content['interpretations'][0]['evaluation_claims'], row['proposals'][0]['evaluation_claims'])
        schema = backend.requests[0].response_schema['properties']
        for branch in schema['proposals']['items']['oneOf']:
            self.assertIn('evaluation_claims', branch['required'])
        self.assertIn('evaluation_claims', schema['attempt_updates']['items']['required'])

    def test_wrong_candidate_protocol_metric_and_simulation_rejected_with_or_without_schema(self):
        changes = [('candidate_id', CALL48['raw']['receipt']['feedback']['candidates'][0]['candidate_id']),
                   ('candidate_id', 'd113fb9b'), ('protocol_ref', 'different-protocol'),
                   ('simulated', True), ('metrics', {'medqa.accuracy': 0.7031}),
                   ('metrics', {'medqa.invalid_format': False}), ('metrics', {'invented.metric': 0.703125}),
                   ('metrics', {'medqa.accuracy': float('nan')}), ('metrics', {})]
        for schema in (True, False):
            for key, value in changes:
                with self.subTest(schema=schema, key=key, value=value):
                    def output(packet):
                        result = valid_output(packet)
                        result['proposals'][0]['evaluation_claims'][0][key] = value
                        return result
                    _, _, service, published = self.make(output, schema=schema)
                    self.assertEqual(self.complete(service)['status'], 'failed')
                    self.assertEqual(published, [])

    def test_training_feedback_and_uncited_or_other_evaluation_cannot_support_metrics(self):
        for change in ('training', 'uncited', 'other-candidate', 'duplicate'):
            with self.subTest(change=change):
                def output(packet):
                    result = valid_output(packet)
                    item = result['proposals'][0]
                    claim = item['evaluation_claims'][0]
                    if change == 'training':
                        claim['evaluation_ref'] = packet['source_refs'][0]
                        item['source_refs'] = [claim['evaluation_ref']]
                    elif change == 'uncited':
                        item['source_refs'] = [packet['source_refs'][0]]
                    elif change == 'other-candidate':
                        claim['evaluation_ref'] = packet['candidate_evidence']['evaluations'][1]['source_ref']
                        item['source_refs'] = [claim['evaluation_ref']]
                    else:
                        item['evaluation_claims'].append(deepcopy(claim))
                    return result
                _, _, service, published = self.make(output)
                self.assertEqual(self.complete(service)['status'], 'failed')
                self.assertEqual(published, [])

    def test_invalid_attempt_claim_isolated_while_valid_memory_publishes(self):
        def output(packet):
            result = valid_output(packet)
            original = deepcopy(json.loads(CALL48['response'])['attempt_updates'][0])
            original['evaluation_claims'] = deepcopy(result['proposals'][0]['evaluation_claims'])
            original['evaluation_claims'][0]['candidate_id'] = packet['candidate_evidence']['current_candidate_ids'][0]
            original['source_refs'].append(original['evaluation_claims'][0]['evaluation_ref'])
            result['attempt_updates'] = [original]
            return result
        _, learning, service, published = self.make(output)
        row = self.complete(service)
        self.assertEqual(row['status'], 'completed')
        self.assertIn('mismatch', row['grouping_error'])
        self.assertEqual(learning.attempts()['items'], [])
        self.assertEqual(len(published), 1)
        self.assertEqual(json.loads(published[0].content)['attempt_interpretations'], [])

    def test_valid_attempt_claim_survives_revision_and_memory(self):
        def output(packet):
            result = valid_output(packet)
            original = deepcopy(json.loads(CALL48['response'])['attempt_updates'][0])
            original['summary'] = 'Metrics belong only to the explicitly cited evaluated candidate.'
            original['evaluation_claims'] = deepcopy(result['proposals'][0]['evaluation_claims'])
            original['source_refs'].append(original['evaluation_claims'][0]['evaluation_ref'])
            result['attempt_updates'] = [original]
            return result
        _, learning, service, published = self.make(output)
        row = self.complete(service)
        self.assertIsNone(row['grouping_error'])
        revision = learning.attempts()['items'][0]
        self.assertEqual(revision['evaluation_claims'], row['attempt_updates'][0]['evaluation_claims'])
        self.assertEqual(json.loads(published[0].content)['attempt_interpretations'][0], revision)

    def test_unpublished_legacy_call48_is_blocked_after_restore_without_model_replay(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            _, learning, _, _ = self.make(path=path)
            state = deepcopy(learning.state)
            row = state['jobs']['call48']
            row['input'].pop('candidate_evidence')
            payload = json.loads(CALL48['response'])
            row.update(status='completed', proposals=payload['proposals'],
                       attempt_updates=payload['attempt_updates'], response={'text': CALL48['response']})
            learning._commit(state)
            learning.close()
            backend = Backend(CALL48['response'])
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path, executor=ImmediateExecutor())
            self.addCleanup(restored.close)
            published = []
            service = FeedbackLearningService(restored, SimpleNamespace(publish=published.append))
            service.refresh()
            row = restored.state['jobs']['call48']
            self.assertEqual(row['status'], 'failed')
            self.assertTrue(restored.read('call48')['publication_error'])
            self.assertEqual(row['response']['text'], CALL48['response'])
            self.assertEqual(row['proposals'], [])
            self.assertEqual(restored.attempts()['items'], [])
            self.assertEqual(published, [])
            self.assertEqual(backend.requests, [])

    def test_publication_rechecks_claims_and_canonical_evidence(self):
        for change in ('claim', 'catalog', 'indices', 'evaluation', 'missing-claims'):
            with self.subTest(change=change):
                _, learning, service, published = self.make()
                learning.poll()
                learning.poll()
                row = learning.state['jobs']['call48']
                if change == 'claim':
                    row['proposals'][0]['evaluation_claims'][0]['candidate_id'] = 'f6ce2f04'
                elif change == 'catalog':
                    row['input']['candidate_evidence']['evaluations'][0]['candidate_id'] = 'f6ce2f04'
                elif change == 'indices':
                    row['input']['evidence_packaging']['streams']['/evaluations']['supplied_indices'] = [-1]
                elif change == 'evaluation':
                    row['input']['evaluations'][0]['quality_metrics']['medqa.accuracy'] = 1.0
                else:
                    row['proposals'][0].pop('evaluation_claims')
                service.refresh()
                self.assertEqual(learning.state['jobs']['call48']['status'], 'failed')
                self.assertEqual(published, [])

    def test_valid_completed_job_publishes_after_restore_without_inference(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            _, learning, _, _ = self.make(path=path)
            learning.poll()
            learning.poll()
            learning.close()
            backend = Backend('must not replay')
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path, executor=ImmediateExecutor())
            self.addCleanup(restored.close)
            published = []
            service = FeedbackLearningService(restored, SimpleNamespace(publish=published.append))
            service.refresh()
            service.refresh()
            self.assertEqual(restored.state['jobs']['call48']['status'], 'completed')
            self.assertEqual(len(published), 1)
            self.assertEqual(backend.requests, [])

    def test_omitted_evaluation_is_distinct_from_snapshot_absence_and_not_citable(self):
        _, learning, _, _ = self.make()
        row = learning.state['jobs']['call48']
        current = row['raw_input']['receipt']['feedback']['candidates'][0]['candidate_id']
        original = deepcopy(row['raw_input']['evaluations'][0])
        original['candidate_id'] = current
        row['raw_input']['evaluations'][0] = original
        packet, _ = _evidence_packet(row, 0, 64)
        evidence = packet['candidate_evidence']
        self.assertEqual(evidence['candidates'][0]['evaluation_coverage'], 'not_supplied')
        self.assertEqual(evidence['evaluations'], [])
        self.assertNotIn('evaluation:' + result_id(original), packet['source_refs'])

    def test_evaluate_receipt_only_adapter_binds_explicit_target_not_returned_candidate(self):
        raw = deepcopy(CALL48['raw'])
        directive, feedback = raw['receipt']['directive'], raw['receipt']['feedback']
        directive.update(mode='evaluate', payload={'candidate_id': 'explicit-target', 'protocol_ref': 'p1'})
        feedback['quality_metrics'] = {'accuracy': 0.0}
        raw['evaluations'] = []
        evidence = candidate_evidence(raw, [])
        source = evidence['evaluations'][0]
        self.assertEqual(source['candidate_id'], 'explicit-target')
        self.assertEqual(source['quality_metrics'], {'accuracy': 0.0})
        self.assertEqual(source['raw_ref'], '/receipt')
        self.assertEqual(evidence['candidates'][0]['evaluation_coverage'], 'not_observed_in_snapshot')

    def test_failed_evaluation_cannot_support_success_metrics(self):
        _, learning, service, published = self.make()
        row = learning.state['jobs']['call48']
        row['raw_input']['evaluations'][0]['status'] = 'failed'
        self.assertTrue(learning._bound_input(row))
        self.assertEqual(self.complete(service)['status'], 'failed')
        self.assertEqual(published, [])

    def test_empty_claims_do_not_certify_prose_semantics_or_completeness(self):
        # Document the deliberate limit. A model can still fail to enumerate a
        # prose claim, or describe an old open question as unresolved. There is
        # no fragile text regex or claim that those semantic errors are solved.
        def output(packet):
            item = deepcopy(json.loads(CALL48['response'])['proposals'][0])
            item['evaluation_claims'] = []
            return {'proposals': [item], 'attempt_updates': []}
        _, _, service, published = self.make(output)
        self.assertEqual(self.complete(service)['status'], 'completed')
        content = json.loads(published[0].content)
        self.assertIn('prose_remains_unverified', content['evaluation_validation'])
        self.assertEqual(content['interpretations'][0]['evaluation_claims'], [])
        self.assertEqual(content['authority'], 'unverified_interpretations')


if __name__ == '__main__':
    unittest.main()
