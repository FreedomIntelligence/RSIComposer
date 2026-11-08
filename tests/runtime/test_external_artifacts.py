"""External references travel as data; built-in stores retain strict ownership checks."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import EvaluationSpec, SFTSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest, EvaluationResult
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import FileArtifactStore, InMemoryArtifactStore
from rsisys.runtime.jobs import MockJobQueue
from rsisys.snapshot.codec import decode, encode


class RecordingBackend:
    """Record requests and return fixture results without any artifact storage or I/O."""

    def __init__(self, result):
        self.requests = []
        self.output = result
        self.jobs = MockJobQueue('external-fixture', pending_polls=0)

    def submit(self, request, *, idempotency_key):
        self.requests.append(deepcopy(request))
        return self.jobs.submit(request, lambda: self.output, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)


class ExternalArtifactContractTests(unittest.TestCase):
    def test_defaults_and_legacy_five_positional_arguments(self):
        ref = ArtifactRef('dataset', 'training_data', 'datasets/latest')
        self.assertIsNone(ref.version)
        self.assertIsNone(ref.digest)
        self.assertEqual(ArtifactRef('dataset', 'training_data', 'datasets/latest', 'v2', 'etag'),
                         replace(ref, version='v2', digest='etag'))

    def test_required_fields_are_nonempty_text(self):
        fields = dict(artifact_id='dataset', kind='data', uri='backend handle')
        for name in fields:
            for value in (None, '', ' \t\n', 0, False, [], {}):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    ArtifactRef(**dict(fields, **{name: value}))

    def test_optional_fields_are_none_or_nonempty_text(self):
        for name in ('version', 'digest'):
            for value in ('', ' \t\n', 0, False, [], {}):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    ArtifactRef('dataset', 'data', 'opaque', **{name: value})

    def test_uri_and_optional_metadata_are_preserved_without_interpretation(self):
        for uri in ('s3://bucket/key', 'https://example.invalid/data', '/missing/data',
                    '../relative/data', 'opaque backend handle', 'custom+scheme:value'):
            for version, digest in ((None, None), ('release', None), (None, 'etag'), ('v1', 'opaque digest')):
                with self.subTest(uri=uri, version=version, digest=digest):
                    ref = ArtifactRef('dataset', 'data', uri, version, digest)
                    self.assertEqual(ref.uri, uri)
                    self.assertEqual((ref.version, ref.digest), (version, digest))
                    encoded = json.loads(json.dumps(encode(ref)))
                    self.assertEqual(decode(encoded), ref)
                    self.assertEqual(encoded['fields']['version'], version)
                    self.assertEqual(encoded['fields']['digest'], digest)


class ExternalArtifactFlowTests(unittest.TestCase):
    def test_unregistered_refs_survive_directives_training_evaluation_and_records(self):
        data = ArtifactRef('external-data', 'dataset', 'dataset handle')
        model = ArtifactRef('external-model', 'model', 's3://models/checkpoint')
        base = ModelRef('external', 'actor', 'v1')
        candidate = Candidate('candidate', base.revision, {'model': model}, ('actor',))
        result = TrainingResult(base, replace(base, revision='v2'), candidate, simulated=True)
        learner = RecordingBackend(result)
        evaluator = RecordingBackend(EvaluationResult(candidate.candidate_id, 'protocol', {}, (), True))
        records = InMemoryExperienceStore()
        policy, environment = Mock(), Mock()
        loop = ExplorationLoop(DefaultExplorationHarness(
            policy, environment, BasicExperienceSystem(records), learner=learner, evaluator=evaluator,
        ), EchoActionSpace())

        def finish(directive):
            # The real JSON codec preserves external input and candidate references.
            restored = decode(json.loads(json.dumps(encode(directive))))
            self.assertEqual(restored, directive)
            progress = loop.accept(restored)
            for _ in range(5):
                if progress.terminal:
                    break
                progress = loop.advance(directive.directive_id)
            self.assertTrue(progress.terminal)
            self.assertEqual(progress.feedback.status, ExecutionStatus.COMPLETED)
            loop.acknowledge(directive.directive_id)
            return progress.feedback

        training = finish(Directive('train', 'train', 'sft', mode='sft', input_refs=(data,),
                                    payload=SFTSpec('recipe', base)))
        self.assertEqual(learner.requests, [ActorTrainingRequest(base, (data,), 'recipe')])
        self.assertEqual(training.artifact_refs, (model,))
        self.assertEqual(training.candidates, (candidate,))
        self.assertEqual(decode(json.loads(json.dumps(encode(learner.requests[0])))), learner.requests[0])

        evaluation = finish(Directive('evaluate', 'evaluate', 'validate', mode='evaluate',
                                      input_refs=(data,), payload=EvaluationSpec(candidate, 'protocol')))
        self.assertEqual(evaluator.requests, [EvaluationRequest(candidate, 'protocol', (data,))])
        training_record, evaluation_record = records.records()
        self.assertEqual((training_record.kind, evaluation_record.kind), ('training_run', 'evaluation'))
        for record in (training_record, evaluation_record):
            self.assertEqual(record.details['request']['data_refs'], (asdict(data),))
        self.assertEqual(evaluation_record.details['request']['candidate']['changes']['model'], asdict(model))
        self.assertEqual(training.evidence_refs, (training_record.record_id,))
        self.assertEqual(evaluation.evidence_refs, (evaluation_record.record_id,))
        self.assertEqual(policy.mock_calls, [])
        self.assertEqual(environment.mock_calls, [])

        # Decision training accepts the same unregistered data without experience IDs.
        decision_request = DecisionTrainingRequest(base, (), 'recipe', (data,))
        backend = RecordingBackend(replace(result, candidate=replace(candidate, affected_roles=('decision_maker',))))
        job = backend.submit(decision_request, idempotency_key='decision')
        self.assertTrue(backend.poll(job).terminal)
        self.assertEqual(backend.requests, [decision_request])


class ManagedArtifactStoreTests(unittest.TestCase):
    def test_stores_reject_external_refs_and_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            stores = (InMemoryArtifactStore(), FileArtifactStore(directory))
            for store in stores:
                ref = store.put(b'original', kind='data')
                self.assertEqual(ref.version, '1')
                self.assertEqual(ref.digest, sha256(b'original').hexdigest())
                self.assertEqual(store.get(ref), b'original')
                readers = (store.get, store.resolve) if isinstance(store, FileArtifactStore) else (store.get,)
                invalid_refs = (
                    ArtifactRef('external', 'data', 'https://example.invalid/data'),
                    ArtifactRef('f' * 32, 'data', 'opaque', 'v1', 'etag'),
                    replace(ref, uri='external handle'),
                    replace(ref, kind='other'),
                    replace(ref, version=None),
                    replace(ref, version='2'),
                    replace(ref, digest=None),
                    replace(ref, digest='wrong'),
                )
                for read in readers:
                    for invalid in invalid_refs:
                        with self.subTest(store=type(store).__name__, read=read.__name__, ref=invalid):
                            with self.assertRaises((KeyError, ValueError)):
                                read(invalid)
                self.assertEqual(store.get(ref), b'original')
            self.assertEqual(stores[1].refs(), (ref,))

    def test_stores_reject_refs_owned_by_another_store(self):
        with tempfile.TemporaryDirectory() as directory:
            for owner, other in (
                (InMemoryArtifactStore(), InMemoryArtifactStore()),
                (FileArtifactStore(Path(directory) / 'a'), FileArtifactStore(Path(directory) / 'b')),
            ):
                ref = owner.put(b'content', kind='data')
                with self.subTest(store=type(owner).__name__), self.assertRaises(KeyError):
                    other.get(ref)

    def test_in_memory_content_hash_is_still_checked(self):
        store = InMemoryArtifactStore()
        ref = store.put(b'original', kind='data')
        store._items[ref.artifact_id] = (ref, b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            store.get(ref)
        missing_digest = replace(ref, digest=None)
        store._items[ref.artifact_id] = (missing_digest, b'changed')
        with self.assertRaises(ValueError):
            store.get(missing_digest)

    def test_file_content_hash_is_still_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileArtifactStore(directory)
            ref = store.put(b'original', kind='data')
            store.resolve(ref).write_bytes(b'changed')
            for read in (lambda: store.get(ref), lambda: store.resolve(ref), store.refs):
                with self.assertRaisesRegex(ValueError, 'digest'):
                    read()

    def test_file_metadata_cannot_drop_digest_or_claim_external_ownership(self):
        for change in ({'digest': None}, {'version': None}, {'uri': 'external handle'}):
            for omit_none in (False, True):
                with self.subTest(change=change, omit_none=omit_none), tempfile.TemporaryDirectory() as directory:
                    store = FileArtifactStore(directory)
                    ref = store.put(b'original', kind='data')
                    tampered = replace(ref, **change)
                    metadata = asdict(tampered)
                    if omit_none:
                        metadata = {key: value for key, value in metadata.items() if value is not None}
                    (Path(directory) / ref.artifact_id / 'ref.json').write_text(json.dumps(metadata))
                    for supplied in (ref, tampered):
                        with self.assertRaises(ValueError):
                            store.get(supplied)
                        with self.assertRaises(ValueError):
                            store.resolve(supplied)
                    with self.assertRaises(ValueError):
                        store.refs()


if __name__ == '__main__':
    unittest.main()
