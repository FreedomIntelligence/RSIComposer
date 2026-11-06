"""Deployment boundaries: timed decisions, exact admission, allocated child devices."""
import json
import hashlib
from collections import UserDict
from pathlib import Path
import os
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.open_lab_experiment import (
    AllocatedProcessBackend, HostAdmissionStop, RecordedBackend, TimedOpenPolicy,
    AbsoluteHostPolicy,
    allocated_resources,
    sglang_template_tools,
)
from examples.medical import open_lab_experiment
from rsisys.loops.contracts.model_service import ModelRef, GenerationRequest, GenerationResponse
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest
from rsisys.runtime.run_policy import RunContext
from rsisys.runtime.artifacts import InMemoryArtifactStore


MODEL = ModelRef('meta', 'fixture', 'revision')


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        self.seen = messages, options
        return list(range(100 + len(options.get('tools', []))))


class Transport:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail
    def _generation_payload(self, request):
        return {'model': request.model.model_id, 'messages': request.chat_messages(), **request.chat_tools()}
    def generate_with_response_observer(self, request, observer):
        self.calls += 1
        observer({'body': '{"provider":"fixture"}'})
        if self.fail:
            raise ValueError('Rejected model identity')
        return GenerationResponse('{"kind":"finish"}', request.model, True)


class MedicalOpenLabTests(unittest.TestCase):
    def test_first_prompt_data_paths_resolve_in_the_seeded_workspace(self):
        class FixtureStop(Exception):
            pass

        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            inputs.mkdir()
            task = {'train': 'train.jsonl', 'development': [
                {'path': 'nested/dev.jsonl', 'benchmark_id': 'fixture'}],
                'closed': {'path': 'DO_NOT_READ_PRIVATE.jsonl'}}
            (inputs / 'nested').mkdir()
            (inputs / 'task.json').write_text(json.dumps(task))
            for name in ('train.jsonl', 'nested/dev.jsonl'):
                (inputs / name).write_text('{"id":"fixture"}\n')
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            inference_runtime = root / 'installed-inference-runtime'
            inference_runtime.mkdir()
            (inference_runtime / 'driver-metadata.txt').write_text('runtime-fixture-v1')
            research_runtime = root / 'research-only-runtime'
            research_runtime.mkdir()
            (research_runtime / 'helper.txt').write_text('not-formal-inference')
            backends = []
            def recorded_backend(*args, **kwargs):
                value = RecordedBackend(*args, **kwargs)
                backends.append(value)
                return value

            def inspect_application(config, *, modules, **options):
                cleanups = []
                context = SimpleNamespace(data_dir=config.data_dir, artifacts=InMemoryArtifactStore(),
                                          add_owned_cleanup=cleanups.append)
                lab = modules.lab_factory(context)
                try:
                    native = lab.lab if hasattr(lab, 'lab') else lab
                    recorded = backends[-1]
                    self.assertFalse(native.wait_interrupted())
                    for flag in ('pause_requested', 'stop_requested'):
                        setattr(recorded, flag, True)
                        self.assertTrue(native.wait_interrupted())
                        setattr(recorded, flag, False)
                    with patch.object(open_lab_experiment.time, 'monotonic', return_value=100):
                        recorded.deadline = 100
                        self.assertTrue(native.wait_interrupted())
                        recorded.deadline = 101
                        self.assertFalse(native.wait_interrupted())
                    recorded.deadline = None
                    project = json.loads((lab.workspace / 'lab_config.json').read_text())
                    policy = TimedOpenPolicy(Transport(), MODEL, project_context=project)
                    prompt = json.loads(policy._decision_request(
                        DecisionContext('fixture', (), (), ())).prompt)
                    advertised = prompt['project_context']
                    formal = advertised['fixed_evaluation']
                    self.assertEqual(formal['evaluation_protocol']['closed_interval_seconds'], 1800)
                    self.assertFalse(formal['closed']['enabled'])
                    self.assertEqual(formal['base_model_paths'], [str(model.resolve())])
                    specs = {spec.name: spec for spec in native.describe()}
                    self.assertNotIn('medical_diagnose', specs)
                    self.assertNotIn('argv', specs['medical_evaluate'].input_schema['properties'])
                    self.assertIn('submission_path', specs['medical_evaluate'].input_schema['properties'])
                    self.assertEqual(native.formal_configuration['evaluation_protocol'], formal['evaluation_protocol'])
                    self.assertIn(str(inference_runtime.resolve()), native.candidate_execution.read_only)
                    self.assertNotIn(str(research_runtime.resolve()), native.candidate_execution.read_only)
                    self.assertEqual(native.installed_runtime['files'][str((inference_runtime / 'driver-metadata.txt').resolve())],
                        hashlib.sha256(b'runtime-fixture-v1').hexdigest())
                    if mode == 'resource_pool':
                        self.assertEqual(advertised['research_execution']['program_slots'], 2)
                        self.assertEqual(advertised['initial_programs'], [])
                        self.assertEqual(advertised['workspace_mode'], 'resource_pool')
                        self.assertFalse((lab.workspace/'candidate.py').exists())
                        self.assertTrue((lab.workspace/'README.md').is_file())
                        for name in ('model_path', 'model_loading', 'inference', 'candidate_reference',
                                     'endpoint', 'served_model', 'weight_identity'):
                            self.assertNotIn(name, advertised)
                        self.assertEqual(advertised['available_models'][0]['model_path'], str(model.resolve()))
                        self.assertIsNone(json.loads((config.data_dir.parent/'model-inventory.json').read_text())['actor'])
                    elif mode == 'managed':
                        self.assertEqual(advertised['inference']['program_slots'], 2)
                        self.assertEqual(advertised['inference']['owner'], 'meta_agent')
                        self.assertNotIn('endpoint', advertised)
                        self.assertNotIn('served_model', advertised)
                        self.assertEqual((lab.workspace/'candidate.py').read_text(), open_lab_experiment.CANDIDATE)
                    else:
                        self.assertEqual(advertised['inference']['owner'], 'external_read_only')
                        self.assertEqual(advertised['endpoint'], 'http://explicit.invalid/v1')
                        self.assertEqual(advertised['served_model'], 'explicit-candidate')
                        self.assertEqual((lab.workspace/'candidate.py').read_text(), open_lab_experiment.EXTERNAL_CANDIDATE)
                    for entry in advertised['available_data']:
                        path = lab.workspace / entry['path']
                        self.assertTrue(path.is_file(), f'Advertised data path missing: {entry["path"]}')
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])
                    seeded_task = json.loads((lab.workspace / advertised['task']).read_text())
                    self.assertEqual(seeded_task, {k: task[k] for k in ('train', 'development')})
                    self.assertNotIn('DO_NOT_READ_PRIVATE', json.dumps(advertised))
                    self.assertNotIn('closed', seeded_task)
                    self.assertEqual((lab.workspace / advertised['data_root'] / task['train']).read_bytes(),
                                     (inputs / task['train']).read_bytes())
                    self.assertEqual(advertised['path_base'], 'lab_workspace')
                    resources = advertised['resources']
                    self.assertEqual(resources['device_count'], len(physical))
                    self.assertEqual(resources['physical_devices'], physical)
                    self.assertEqual(resources['logical_to_physical'], dict(zip(map(str, range(len(physical))), physical)))
                    self.assertEqual(resources, json.loads((config.data_dir.parent / 'resource-allocation.json').read_text()))
                    self.assertEqual(resources['memory_availability']['capacity_mib_per_device'], 65536)
                    ident = lab.execute(open_lab_experiment.OperationCall('lab_start', {'key': 'mapping',
                        'argv': [sys.executable, '-c', 'import os;print(os.environ["ASCEND_RT_VISIBLE_DEVICES"])']})).data['job_id']
                    native = lab.lab if hasattr(lab, 'lab') else lab
                    native.backend.wait(native._jobs[ident][0])
                    receipt = lab.execute(open_lab_experiment.OperationCall('lab_result', {'job_id': ident}))
                    self.assertEqual(receipt.data['stdout'].strip(), resources['visibility_environment']['ASCEND_RT_VISIBLE_DEVICES'])
                finally:
                    lab.close()
                    for cleanup in reversed(cleanups):
                        cleanup()
                raise FixtureStop()

            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()))
            for mode in ('managed', 'external', 'resource_pool'):
                physical = list(map(str, range(8, 16))) if mode == 'resource_pool' else ['8','9','10','11','13','14','15']
                with self.subTest(mode=mode), patch.dict(sys.modules, {'transformers': transformers}), \
                     patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': ','.join(physical),
                                            'RSISYS_LAB_ENDPOINT': 'http://implicit.invalid/v1'}), \
                     patch.object(open_lab_experiment.ResearchAssembly, 'langmem',
                                  return_value=open_lab_experiment.ResearchAssembly()), \
                     patch.object(open_lab_experiment, 'RecordedBackend', side_effect=recorded_backend), \
                     patch.object(open_lab_experiment, 'Application', side_effect=inspect_application), \
                     self.assertRaises(FixtureStop):
                    extra = (['--lab-endpoint', 'http://explicit.invalid/v1', '--lab-model-id', 'explicit-candidate']
                        if mode == 'external' else ['--workspace-mode', 'candidate_scaffold'] if mode == 'managed' else [])
                    open_lab_experiment.main(['--root', str(root / ('run-'+mode)), '--task', str(inputs / 'task.json'),
                        '--execution-policy', json.dumps({'read_only': list(__import__(
                            'rsisys.runtime.isolation', fromlist=['system_runtime_paths']).system_runtime_paths()) + [str(research_runtime)]}),
                        '--inference-runtime-path', str(inference_runtime),
                        '--model-path', str(model), '--python', sys.executable,
                        '--endpoint', 'http://127.0.0.1:1/v1', '--run-id', 'fixture', '--device-type', 'npu',
                        '--resource-allocation', json.dumps({'physical_devices': list(map(int, physical)),
                            'memory_availability': {'capacity_mib_per_device': 65536}}), *extra])

    def test_owned_inference_service_and_fixed_evaluation_run_together_and_clean_up(self):
        from examples.medical.fixed_evaluation import FixedMedicalLab
        from rsisys.loops.contracts.decision import OperationCall
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'fixture.jsonl'
            data.write_text(json.dumps({'id':'one', 'question':'fixture', 'options':{'A':'a','B':'b'}, 'gold':'A'})+'\n')
            backend = open_lab_experiment.RetainedAllocatedProcessBackend(root/'executor', {'CUDA_VISIBLE_DEVICES':'0',
                'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src')})
            lab = FixedMedicalLab(backend, InMemoryArtifactStore(), max_parallel=2, timeout_seconds=10,
                python=sys.executable, evaluation_root=root/'evaluations', evaluation_groups=[{
                    'benchmark_id':'fixture','path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}])
            try:
                lab.execute(OperationCall('lab_write_file', {'path':'server.py', 'content': '''import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        b=json.dumps({'predictions':[{'benchmark_id':i['benchmark_id'],'id':i['id'],'text':'A'} for i in p['items']]}).encode()
        self.send_response(200);self.end_headers();self.wfile.write(b)
    def log_message(self,*args): pass
server=HTTPServer(('127.0.0.1',0),Handler)
Path('port.txt').write_text(str(server.server_port))
server.serve_forever()
'''}))
                service = lab.execute(OperationCall('lab_start', {'key':'owned-service','argv':[sys.executable,'server.py']})).data['job_id']
                until = time.monotonic()+5
                while not (lab.workspace/'port.txt').exists() and time.monotonic()<until:
                    time.sleep(.01)
                self.assertTrue((lab.workspace/'port.txt').is_file())
                lab.execute(OperationCall('lab_write_file', {'path':'client.py', 'content': '''import sys
from pathlib import Path
from urllib.request import Request,urlopen
port=Path('port.txt').read_text()
with urlopen(Request('http://127.0.0.1:'+port,data=sys.stdin.buffer.read(),headers={'Content-Type':'application/json'}),timeout=3) as r:
    sys.stdout.buffer.write(r.read())
'''}))
                ident = lab.execute(OperationCall('medical_evaluate', {'key':'evaluation','candidate_id':'owned-fixture',
                    'argv':[sys.executable,'client.py']})).data['job_id']
                completed = backend.wait(lab._jobs[ident][0])
                self.assertEqual(completed.exit_code, 0, completed.stderr)
                receipt = lab.execute(OperationCall('lab_result', {'job_id':ident}))
                self.assertEqual(receipt.data['fixed_evaluation']['metrics']['fixture'], {'accuracy':1.0,'n':1})
                self.assertEqual(lab.execute(OperationCall('lab_poll', {'job_id':service})).data['state'], 'running')
                lab.execute(OperationCall('lab_cancel', {'job_id':service}))
                backend.wait(lab._jobs[service][0])
            finally:
                lab.close()
                backend.close_all()
            self.assertEqual(backend.active_count, 0)

    def test_conflicting_resource_declarations_fail_before_lab_execution(self):
        environment = {'ASCEND_RT_VISIBLE_DEVICES': '8,9'}
        for declaration in ({'physical_devices': [0, 1]}, {'physical_devices': [9, 8]},
                            {'logical_devices': [8, 9]}):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, 'mapping'):
                allocated_resources(environment, 'npu', declaration)
        for mapping in ('', '8,8', '8, 9'):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                allocated_resources({'ASCEND_RT_VISIBLE_DEVICES': mapping}, 'npu')

    def test_no_idle_envelopes_and_no_host_budget_in_context(self):
        policy = TimedOpenPolicy(Transport(), MODEL, directives=DirectiveDefinitions(()))
        context = DecisionContext('objective', (), (), (), directive_modes=(),
            run_control={'remaining_seconds': 10, 'research_round': 4})
        self.assertEqual(policy._context_inputs(context)['run_control'], {'stopping_owner': 'host'})
        schema = policy.decision_schema(context)
        self.assertFalse(any(b['properties']['kind']['const'] in ('wait', 'finish') for b in schema['oneOf']))
        with self.assertRaises(ValueError):
            policy._parse_decision(GenerationResponse('{"kind":"finish"}', MODEL, True), context)

    def test_actual_native_messages_and_tools_are_counted(self):
        with TemporaryDirectory() as directory:
            tokenizer = Tokenizer()
            backend = RecordedBackend(Transport(), tokenizer, directory, max_calls=1)
            request = GenerationRequest('decision_maker', MODEL, '', messages=(
                {'role': 'user', 'content': 'observation'},), tools=(
                {'type': 'function', 'function': {'name': 'record', 'parameters': {'type': 'object'}}},))
            self.assertEqual(backend.count_tokens(request), 101)
            self.assertEqual(tokenizer.seen[0], request.chat_messages())
            self.assertEqual(tokenizer.seen[1]['tools'], list(request.tools))
            self.assertFalse(tokenizer.seen[1]['enable_thinking'])

    def test_transformers_batch_encoding_mapping_counts_tokens_not_fields(self):
        class BatchTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return UserDict(input_ids=list(range(54075)), attention_mask=[1] * 54075)
        with TemporaryDirectory() as directory:
            backend = RecordedBackend(Transport(), BatchTokenizer(), directory, max_calls=1)
            self.assertEqual(backend.count_tokens(GenerationRequest('decision_maker', MODEL, 'long')), 54075)

    def test_sglang_tool_template_uses_provider_defaults_without_mutating_wire_tools(self):
        tools = ({'type': 'function', 'function': {'name': 'record', 'parameters': {'type': 'object'}}},
                 {'type': 'function', 'function': {'name': 'edit', 'strict': True,
                    'parameters': {'type': 'object'}}, 'defer_loading': True})
        request = GenerationRequest('feedback', MODEL, 'observe', tools=tools)
        with TemporaryDirectory() as directory:
            tokenizer = Tokenizer()
            backend = RecordedBackend(Transport(), tokenizer, directory, max_calls=1,
                                      template_tool_normalizer=sglang_template_tools)
            self.assertEqual(backend.count_tokens(request), 102)
            rendered = tokenizer.seen[1]['tools']
            self.assertEqual(rendered[0], {'type': 'function', 'defer_loading': None, 'function': {
                'name': 'record', 'description': None, 'parameters': {'type': 'object'}, 'strict': False}})
            self.assertTrue(rendered[1]['function']['strict'])
            self.assertTrue(rendered[1]['function']['defer_loading'])
            self.assertNotIn('strict', request.tools[0]['function'])
            self.assertNotIn('defer_loading', request.tools[0])
            for choice in ('none', {'type': 'function', 'function': {'name': 'edit'}}):
                selected = GenerationRequest('feedback', MODEL, 'observe', tools=tools, tool_choice=choice)
                backend.count_tokens(selected)
                self.assertEqual([t['function']['name'] for t in tokenizer.seen[1].get('tools', [])],
                                 [] if choice == 'none' else ['edit'])

    def test_host_signal_closes_admission_before_another_request(self):
        with TemporaryDirectory() as directory:
            transport = Transport()
            backend = RecordedBackend(transport, Tokenizer(), directory, max_calls=10)
            backend.stop_requested = True
            with self.assertRaises(HostAdmissionStop):
                backend.generate(GenerationRequest('decision_maker', MODEL, 'choose'))
            self.assertEqual(transport.calls, 0)

    def test_absolute_deadline_cannot_be_extended_by_a_new_run(self):
        backend = type('Backend', (), {'stop_requested': False})()
        policy = AbsoluteHostPolicy(backend, 36000, deadline_epoch=1)
        outcome = policy.evaluate(RunContext('objective', 0, 1, False))
        self.assertEqual((outcome.action, outcome.shutdown), ('stop', 'cancel'))

    def test_rejected_response_is_accounted_and_not_replayed_after_limit(self):
        with TemporaryDirectory() as directory:
            transport = Transport(fail=True)
            backend = RecordedBackend(transport, Tokenizer(), directory, max_calls=1)
            request = GenerationRequest('decision_maker', MODEL, 'choose')
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.generate(request)
            with self.assertRaises(HostAdmissionStop):
                backend.generate(request)
            self.assertEqual(transport.calls, 1)
            errors = (Path(directory) / 'model_errors.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(errors[0])['call'], 1)
            self.assertEqual(len((Path(directory) / 'model_wire.jsonl').read_text().splitlines()), 1)

    def test_real_child_inherits_only_explicit_host_environment(self):
        with TemporaryDirectory() as directory:
            backend = AllocatedProcessBackend(directory, {'CUDA_VISIBLE_DEVICES': '3,4',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HUB_OFFLINE': '1'})
            sandbox = backend.create(SandboxSpec('local', 10))
            self.addCleanup(backend.close_all)
            ref = backend.exec(sandbox, ExecRequest((sys.executable, '-c',
                'import os,json;print(json.dumps({k:os.environ.get(k) for k in '
                '["CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","PRIVATE_CREDENTIAL"]}))'), 10))
            result = backend.wait(ref)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(json.loads(result.stdout), {'CUDA_VISIBLE_DEVICES': '3,4',
                'HF_HUB_OFFLINE': '1', 'PRIVATE_CREDENTIAL': None})


if __name__ == '__main__':
    unittest.main()
