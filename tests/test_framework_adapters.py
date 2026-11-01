"""Real local subprocess/HTTP transport tests; fixtures do not train or run vLLM.

Optional real Transformers+PEFT training is validated separately by deployment.
"""
from dataclasses import asdict, replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.adapters.training import (ProcessTrainingBackend, TrainingCommand,
    transformers_peft_command, llamafactory_command, _identity, _native_inputs,
    _transformers_peft, _worker_ranks, _worker_group)
from rsisys.adapters.serving import VLLMBackend, VLLMDeploymentBackend, VLLMLoRAActivator
from rsisys.loops.contracts.backend import BackendCapabilities, DeploymentStatus
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobStatus, JobRef
from rsisys.loops.contracts.model_service import ModelRef, BackendBinding, GenerationRequest
from rsisys.loops.contracts.training import TrainingResult, TrainingOrigin, ModelActivationPending, ResumeMode
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.runtime.model import ModelResponseError

BASE = ModelRef('vllm', 'base', 'r0')
TRAINED = replace(BASE, revision='r1')
FIXTURE = r'''
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ['RSISYS_JOB_DIR'])
payload = json.loads(pathlib.Path(os.environ['RSISYS_REQUEST_FILE']).read_text())
request = payload['request']
with (root / 'starts').open('a') as stream: stream.write('start\n')
(root / 'environment.json').write_text(json.dumps(dict(os.environ)))
(root / 'argv.json').write_text(json.dumps(sys.argv[1:]))
time.sleep(float(os.environ.get('FIXTURE_DELAY', '0')))
base = request['base']
model = dict(base, revision='r1')
candidate = dict(candidate_id='fixture', base_version=base['revision'], changes={}, affected_roles=['actor'],
                 parent_system=(request.get('origin') or {}).get('parent_system'))
result = dict(base=base, model=model, candidate=candidate, metrics={'loss': 0.5}, simulated=False)
if os.environ.get('FIXTURE_BAD_BASE'): result['base']['revision'] = 'wrong'
if os.environ.get('FIXTURE_MALFORMED'): result = {'unexpected': True}
if os.environ.get('FIXTURE_WRITE', '1') == '1':
    pathlib.Path(os.environ['RSISYS_RESULT_FILE']).write_text(json.dumps(result))
sys.exit(int(os.environ.get('FIXTURE_EXIT', '0')))
'''


class ProcessAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / 'fixture.py'
        self.script.write_text(FIXTURE)
        self.request = ActorTrainingRequest(BASE, (), 'native.json')

    def backend(self, **environment):
        return ProcessTrainingBackend(self.root / 'jobs', lambda request, directory:
            TrainingCommand((sys.executable, str(self.script), 'literal;$(touch nope)'),
                            str(directory), environment, timeout_seconds=5))

    def finish(self, backend, job, *, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            outcome = backend.poll(job)
            if outcome.terminal or outcome.status == JobStatus.UNKNOWN:
                return outcome
            time.sleep(.025)
        self.fail('Fixture did not reach a terminal state')

    def started(self, job):
        directory = self.root / 'jobs' / job.job_id
        deadline = time.monotonic() + 5
        while not (directory / 'starts').exists() and time.monotonic() < deadline:
            time.sleep(.025)
        self.assertTrue((directory / 'starts').exists())
        return directory

    def test_real_subprocess_and_new_backend_reattach_exactly_once(self):
        backend = self.backend(FIXTURE_DELAY='.2')
        job = backend.submit(self.request, idempotency_key='same')
        other = self.backend(FIXTURE_DELAY='.2')
        self.assertEqual(other.submit(self.request, idempotency_key='same'), job)
        outcome = self.finish(other, job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        self.assertFalse(outcome.simulated)
        self.assertEqual(other.result(job).model, TRAINED)
        directory = self.started(job)
        self.assertEqual((directory / 'starts').read_text(), 'start\n')
        self.assertEqual(json.loads((directory / 'argv.json').read_text()), ['literal;$(touch nope)'])
        self.assertFalse((directory / 'nope').exists())
        (directory / 'result.json').write_text('{}')
        self.assertEqual(self.backend().reattach(job).result, outcome.result)
        with self.assertRaisesRegex(ValueError, 'different request'):
            other.submit(replace(self.request, recipe_ref='changed'), idempotency_key='same')
        with self.assertRaises(ValueError):
            other.poll(JobRef('../bad', job.backend))

    def test_controller_process_exits_and_receipt_survives(self):
        controller = self.root / 'controller.py'
        controller.write_text('''import os, sys\nfrom pathlib import Path\nfrom rsisys.adapters.training import ProcessTrainingBackend,TrainingCommand\nfrom rsisys.loops.contracts.model_service import ModelRef\nfrom rsisys.loops.exploration.harness.learner import ActorTrainingRequest\nb=ProcessTrainingBackend(sys.argv[1],lambda r,d:TrainingCommand((sys.executable,sys.argv[2]),str(d),{'FIXTURE_DELAY':'.3'}))\nj=b.submit(ActorTrainingRequest(ModelRef('vllm','base','r0'),(),'native.json'),idempotency_key='survivor')\nPath(sys.argv[3]).write_text(j.job_id)\nos._exit(0)\n''')
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        completed = subprocess.run([sys.executable, str(controller), str(self.root / 'jobs'),
            str(self.script), str(self.root / 'job')], env=env, capture_output=True, timeout=8)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        backend = self.backend()
        job = JobRef((self.root / 'job').read_text(), backend.backend)
        self.assertEqual(self.finish(backend, job).status, JobStatus.SUCCEEDED)

    def test_concurrent_submit_launches_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = list(pool.map(lambda _: self.backend(FIXTURE_DELAY='.1').submit(
                self.request, idempotency_key='concurrent'), range(6)))
        self.assertEqual(len(set(jobs)), 1)
        self.assertEqual(self.finish(self.backend(), jobs[0]).status, JobStatus.SUCCEEDED)
        self.assertEqual((self.started(jobs[0]) / 'starts').read_text(), 'start\n')

    def test_cancel_after_reattach_and_timeout(self):
        backend = self.backend(FIXTURE_DELAY='20')
        job = backend.submit(self.request, idempotency_key='cancel')
        directory = self.started(job)
        self.backend().cancel(job)
        self.assertEqual(self.finish(self.backend(), job).status, JobStatus.CANCELLED)
        child = json.loads((directory / 'state.json').read_text())['child_pid']
        self.assertIsNone(_identity(child))
        timed = ProcessTrainingBackend(self.root / 'timeout', lambda r,d:
            TrainingCommand((sys.executable, str(self.script)), str(d), {'FIXTURE_DELAY':'20'}, .1))
        job = timed.submit(self.request, idempotency_key='timeout')
        outcome = self.finish(timed, job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertIn('timed out', outcome.error)

    def test_lost_supervisor_and_reused_pid_are_unknown_never_restarted(self):
        backend = self.backend(FIXTURE_DELAY='20')
        job = backend.submit(self.request, idempotency_key='lost')
        directory = self.started(job)
        receipt = json.loads((directory / 'state.json').read_text())
        os.kill(receipt['pid'], signal.SIGKILL)
        try:
            deadline = time.monotonic() + 3
            while _identity(receipt['pid']) is not None and time.monotonic() < deadline:
                time.sleep(.025)
            self.assertEqual(backend.reattach(job).status, JobStatus.UNKNOWN)
            self.assertEqual(backend.submit(self.request, idempotency_key='lost'), job)
            receipt.update(pid=os.getpid(), identity='different-start')
            (directory / 'state.json').write_text(json.dumps(receipt))
            backend.cancel(job)  # A marker only; cannot signal the recycled PID.
            self.assertEqual(backend.poll(job).status, JobStatus.UNKNOWN)
            self.assertEqual((directory / 'starts').read_text(), 'start\n')
        finally:
            os.killpg(receipt['child_pid'], signal.SIGKILL)

    def test_exit_nonzero_missing_and_malformed_results_fail(self):
        for index, env in enumerate(({'FIXTURE_EXIT':'7'}, {'FIXTURE_WRITE':'0'},
                                    {'FIXTURE_MALFORMED':'1'}, {'FIXTURE_BAD_BASE':'1'})):
            with self.subTest(env=env):
                backend = self.backend(**env)
                job = backend.submit(self.request, idempotency_key=str(index))
                self.assertEqual(self.finish(backend, job).status, JobStatus.FAILED)
                with self.assertRaises(RuntimeError):
                    backend.result(job)

    def test_terminal_receipt_written_during_identity_check_is_not_unknown(self):
        backend = self.backend()
        job = JobRef('a' * 64, backend.backend)
        directory = self.root / 'jobs' / job.job_id
        directory.mkdir()
        (directory / 'receipt.json').write_text('{}')
        state = directory / 'state.json'
        state.write_text(json.dumps({'status': 'running', 'pid': 123, 'identity': 'fixture'}))
        def exited(pid):
            state.write_text(json.dumps({'status': 'failed', 'error': 'fixture failure'}))
            return None
        with patch('rsisys.adapters.training._identity', side_effect=exited):
            outcome = backend.poll(job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertEqual(outcome.error, 'fixture failure')

    def test_scheduler_visibility_is_preserved_credentials_are_not(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'3', 'NVIDIA_VISIBLE_DEVICES':'GPU-3',
            'ASCEND_RT_VISIBLE_DEVICES':'2', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
            'NCCL_SOCKET_IFNAME':'eth0', 'NCCL_IB_HCA':'mlx5_0', 'NCCL_DEBUG':'WARN',
            'LD_LIBRARY_PATH':'/pixi/cuda-compat:/pixi/lib:/pixi/targets/x86_64-linux/lib',
            'CUDA_HOME':'/pixi', 'NCCL_IB_TIMEOUT':'22', 'NCCL_CUMEM_ENABLE':'0',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1', 'NCCL_PRIVATE_TOKEN':'secret',
            'PRIVATE_API_TOKEN':'must-not-be-inherited'}):
            backend = self.backend()
            job = backend.submit(self.request, idempotency_key='env')
            self.assertEqual(self.finish(backend, job).status, JobStatus.SUCCEEDED)
        env = json.loads((self.started(job) / 'environment.json').read_text())
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
        self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'], 'GPU-3')
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '2')
        self.assertEqual(env['MKL_NUM_THREADS'], '1')
        self.assertEqual(env['NCCL_SOCKET_IFNAME'], 'eth0')
        self.assertEqual(env['NCCL_IB_HCA'], 'mlx5_0')
        self.assertEqual(env['NCCL_DEBUG'], 'WARN')
        self.assertEqual(env['LD_LIBRARY_PATH'],
                         '/pixi/cuda-compat:/pixi/lib:/pixi/targets/x86_64-linux/lib')
        self.assertEqual(env['CUDA_HOME'], '/pixi')
        self.assertEqual(env['NCCL_IB_TIMEOUT'], '22')
        self.assertEqual(env['NCCL_CUMEM_ENABLE'], '0')
        self.assertEqual(env['TORCH_NCCL_ASYNC_ERROR_HANDLING'], '1')
        self.assertNotIn('NCCL_PRIVATE_TOKEN', env)
        self.assertNotIn('PRIVATE_API_TOKEN', env)

    def test_capabilities_reject_unsupported_before_launch(self):
        backend = self.backend()
        with self.assertRaises(NotImplementedError):
            backend.submit(replace(self.request, method='dpo'), idempotency_key='bad-method')
        origin = TrainingOrigin('parent', BASE, ResumeMode.FULL_STATE,
                                ArtifactRef('checkpoint','trainer_checkpoint','file:///checkpoint'))
        with self.assertRaises(NotImplementedError):
            backend.submit(replace(self.request, origin=origin), idempotency_key='bad-resume')
        self.assertEqual(list((self.root / 'jobs').glob('*/receipt.json')), [])
        with self.assertRaisesRegex(ValueError, 'live updates'):
            ProcessTrainingBackend(self.root / 'false-capabilities', capabilities=BackendCapabilities(
                methods=('sft',), update_fields=('learning_rate',)))

    def test_invalid_terminal_result_is_stable_and_result_does_not_poll(self):
        backend = self.backend(FIXTURE_MALFORMED='1')
        job = backend.submit(self.request, idempotency_key='invalid')
        with self.assertRaises(RuntimeError):
            backend.result(job)
        outcome = self.finish(backend,job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        result = TrainingResult(BASE,TRAINED,Candidate('corrected','r0',{},('actor',)))
        (self.started(job)/'result.json').write_text(json.dumps(asdict(result)))
        self.assertEqual(self.backend().reattach(job), outcome)

    def test_native_worker_rejects_existing_output_before_framework_import(self):
        from rsisys.adapters.training import _transformers_peft
        data = self.root / 'rows.jsonl'
        data.write_text('{"input_ids":[1,2],"labels":[1,2]}\n')
        config = self.root / 'recipe.json'
        output = self.root / 'existing'
        output.mkdir()
        config.write_text(json.dumps({'initial_model':asdict(BASE),'train_dataset':str(data),
            'training_args':{'output_dir':str(output)}}))
        (self.root/'request.json').write_text(json.dumps({'request_type':'ActorTrainingRequest',
                                                        'request':asdict(self.request)}))
        with patch.dict(os.environ, {'RSISYS_JOB_DIR':str(self.root),
            'RSISYS_REQUEST_FILE':str(self.root/'request.json')}), self.assertRaises(FileExistsError):
            _transformers_peft(config)

    def test_native_commands_and_explicit_result_reader(self):
        command = llamafactory_command('recipe with spaces.yaml', cwd=self.root)
        self.assertEqual(command.argv, ('llamafactory-cli','train','recipe with spaces.yaml'))
        self.assertIn('transformers-peft', transformers_peft_command('recipe.json',cwd=self.root).argv)
        backend = ProcessTrainingBackend(self.root / 'native-cli', lambda r,d:
            TrainingCommand((sys.executable, '-c', 'print("native CLI fixture")'), str(d)),
            result_reader=lambda r,d: TrainingResult(r.base, TRAINED,
                Candidate('native',r.base.revision,{},('actor',))))
        job = backend.submit(self.request, idempotency_key='native')
        self.assertEqual(self.finish(backend, job).result.model, TRAINED)

    def test_native_input_identity_digest_and_no_implicit_conversion(self):
        data = self.root / 'rows.jsonl'
        data.write_text('{"input_ids":[1,2],"labels":[1,2]}\n')
        config = self.root / 'recipe.json'
        native = {'initial_model':asdict(BASE),'model_config':{'model_type':'gpt2'},
                  'lora_config':{'r':2},'train_dataset':'rows.jsonl'}
        config.write_text(json.dumps(native))
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        request = replace(self.request, recipe_ref=str(config), data_refs=(
            ArtifactRef('rows','tokenized_rows', data.as_uri(), digest=digest),))
        _, rows, evidence = _native_inputs(asdict(request), config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(evidence['data_sha256'], digest)
        self.assertEqual(evidence['initialization'], 'random_from_config')
        self.assertFalse(evidence['weight_identity_verified'])
        with self.assertRaisesRegex(ValueError, 'data_refs differ'):
            _native_inputs(asdict(replace(request, data_refs=(ArtifactRef('source','medical',
                (self.root / 'raw-medical.jsonl').as_uri()),))), config)
        native['initial_model'] = asdict(TRAINED)
        config.write_text(json.dumps(native))
        with self.assertRaisesRegex(ValueError, 'initial_model'):
            ProcessTrainingBackend(self.root / 'default').submit(request, idempotency_key='bad')


class NativeDistributedWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = self.root / 'recipe with spaces.json'
        self.data = self.root / 'rows.jsonl'
        self.data.write_text('{"input_ids":[1,2,3],"labels":[-100,2,3]}\n')
        self.config = {'initial_model': asdict(BASE), 'train_dataset': str(self.data),
            'model_name_or_path': 'official', 'lora_config': {'r': 2},
            'model_kwargs': {'device_map': 'auto', 'local_files_only': True},
            'training_args': {'max_steps': 1, 'gradient_accumulation_steps': 2}}
        self.request = ActorTrainingRequest(BASE, (), str(self.recipe))
        self.request_file = self.root / 'request.json'
        self.request_file.write_text(json.dumps({'request_type': 'ActorTrainingRequest',
                                                'request': asdict(self.request)}))

    def write_recipe(self):
        self.recipe.write_text(json.dumps(self.config))

    def test_launch_argv_and_real_subprocess_without_optional_packages(self):
        single = transformers_peft_command(self.recipe, cwd=self.root)
        self.assertEqual(single.argv, (sys.executable, '-m', 'rsisys.adapters.training',
                                      'transformers-peft', str(self.recipe)))
        command = transformers_peft_command(self.recipe, cwd=self.root, num_processes=8)
        self.assertEqual(command.argv, (sys.executable, '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', '--nproc-per-node=8', '-m',
            'rsisys.adapters.training', 'transformers-peft', str(self.recipe)))
        self.assertEqual(transformers_peft_command(self.recipe, cwd=self.root,
            python='/custom/python', num_processes=8).argv[0], '/custom/python')
        # Execute the exact command against a recording launcher, without torch/GPU.
        package = self.root / 'torch' / 'distributed'
        package.mkdir(parents=True)
        (package.parent / '__init__.py').write_text('')
        (package / '__init__.py').write_text('')
        (package / 'run.py').write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        completed = subprocess.run(command.argv, cwd=command.cwd,
            env=dict(os.environ, PYTHONPATH=str(self.root)), capture_output=True, text=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), list(command.argv[3:]))
        for value in (True, False, 0, -1, 1.5, '8', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'num_processes'):
                transformers_peft_command(self.recipe, cwd=self.root, num_processes=value)

    def test_default_backend_selects_recipe_count_and_validates_before_launch(self):
        for count in (None, 8):
            with self.subTest(count=count):
                if count is not None:
                    self.config['num_processes'] = count
                self.write_recipe()
                backend = ProcessTrainingBackend(self.root / f'jobs-{count}')
                with patch('rsisys.adapters.training.subprocess.Popen') as popen:
                    job = backend.submit(self.request, idempotency_key='native')
                popen.assert_called_once()
                receipt = json.loads((backend.root / job.job_id / 'receipt.json').read_text())
                expected = transformers_peft_command(self.recipe.resolve(), cwd=backend.root,
                                                      num_processes=count or 1).argv
                self.assertEqual(receipt['command']['argv'], list(expected))
                fingerprint = json.loads((backend.root / job.job_id / 'native-inputs.json').read_text())
                self.assertEqual(fingerprint['data_sha256'], hashlib.sha256(self.data.read_bytes()).hexdigest())
        for count in (True, '8', 0, -1):
            self.config['num_processes'] = count
            self.write_recipe()
            with self.subTest(count=count), patch('rsisys.adapters.training.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'num_processes'):
                    ProcessTrainingBackend(self.root / 'invalid').submit(self.request,
                        idempotency_key=f'invalid-{count}')
                popen.assert_not_called()

    def test_rank_validation_requires_one_node_and_matching_recipe(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_worker_ranks({}), (0, 0, 1))
        environment = {'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8', 'RANK': '7', 'LOCAL_RANK': '7'}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(_worker_ranks({'num_processes': 8}), (7, 7, 8))
            with self.assertRaisesRegex(ValueError, 'WORLD_SIZE'):
                _worker_ranks({'num_processes': 1})
        for updates in ({'LOCAL_WORLD_SIZE': '4'}, {'RANK': '8'}, {'LOCAL_RANK': '0'},
                        {'WORLD_SIZE': '0'}, {'LOCAL_RANK': '-1'}):
            with self.subTest(updates=updates), patch.dict(os.environ, dict(environment, **updates), clear=True):
                with self.assertRaises(ValueError):
                    _worker_ranks({})
        with patch.dict(os.environ, {'WORLD_SIZE': '8'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'single-node'):
                _worker_ranks({})

    def fake_frameworks(self, rank, world_size, *, fail_train=False):
        events = []
        counts = [{'presented_rows': 4, 'presented_input_tokens': 8},
                  {'presented_rows': 2, 'presented_input_tokens': 10}]
        model = SimpleNamespace(training=True, config=SimpleNamespace(use_cache=True))
        outer = self
        def gather(target, local):
            events.append('gather')
            outer.assertEqual(local, counts[rank])
            target[:] = counts
        distributed = SimpleNamespace(
            init_process_group=lambda **kw: events.append(('init', kw)),
            destroy_process_group=lambda: events.append('destroy'),
            barrier=lambda: events.append('barrier'), get_world_size=lambda: world_size,
            all_gather_object=gather)
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: world_size,
            set_device=lambda device: events.append(('device', device)))
        torch.distributed = distributed
        torch.manual_seed = lambda seed: events.append(('seed', seed))
        class AutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                events.append(('load', path, kwargs))
                if world_size > 1:
                    outer.assertEqual(kwargs['device_map'], {'': rank})
                    outer.assertLess(events.index(('device', rank)), len(events) - 1)
                    outer.assertIn('barrier', events)
                return model
            @staticmethod
            def from_config(config):
                events.append(('from_config', config))
                return model
        class Trainer:
            def __init__(self, **kwargs):
                self.args = SimpleNamespace(**kwargs['args'])
                self.state = SimpleNamespace(global_step=1, max_steps=1)
                self.callbacks = kwargs['callbacks']
                events.append(('trainer_args', kwargs['args']))
            def compute_loss(self, model, inputs, *args, **kwargs):
                return .5
            def train(self):
                events.append('train')
                if fail_train:
                    raise RuntimeError('synthetic training failure')
                batch = SimpleNamespace(shape=(2, 3), numel=lambda: 6) if rank == 0 else SimpleNamespace(
                    shape=(1, 5), numel=lambda: 5)
                inputs = {'input_ids': batch}
                if rank == 0:
                    inputs['attention_mask'] = SimpleNamespace(sum=lambda: 4)
                for _ in range(2):  # Two actual training microbatches, not dataset size * epochs.
                    self.compute_loss(model, inputs)
                model.training = False
                self.compute_loss(model, inputs)  # Evaluation must not inflate exposure.
                for callback in self.callbacks:
                    callback.on_log(self.args, self.state, None, {'loss': .5, 'bad': float('nan')})
                return SimpleNamespace(metrics={'loss': .5})
            def save_model(self, path):
                events.append('save')
                if rank == 0:
                    (Path(path) / 'adapter_config.json').write_text('{}')
        transformers = ModuleType('transformers')
        transformers.AutoConfig = SimpleNamespace(for_model=lambda kind, **kw: dict(model_type=kind, **kw))
        transformers.AutoModelForCausalLM = AutoModel
        transformers.Trainer = Trainer
        transformers.TrainerCallback = object
        transformers.TrainingArguments = lambda **kwargs: kwargs
        transformers.default_data_collator = object()
        peft = ModuleType('peft')
        peft.LoraConfig = lambda **kwargs: kwargs
        peft.get_peft_model = lambda model, config: model
        peft.PeftModel = SimpleNamespace(from_pretrained=lambda model, uri, **kwargs:
            (events.append(('resume', uri, kwargs)) or model))
        return {'torch': torch, 'transformers': transformers, 'peft': peft}, events, counts

    def run_worker(self, rank, world_size, *, fail_train=False):
        modules, events, counts = self.fake_frameworks(rank, world_size, fail_train=fail_train)
        environment = {'RSISYS_JOB_DIR': str(self.root), 'RSISYS_REQUEST_FILE': str(self.request_file),
            'RSISYS_RESULT_FILE': str(self.root / 'result.json')}
        if world_size > 1:
            environment.update(WORLD_SIZE=str(world_size), LOCAL_WORLD_SIZE=str(world_size),
                               RANK=str(rank), LOCAL_RANK=str(rank))
        self.write_recipe()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True):
            _transformers_peft(self.recipe)
        return events, counts

    def test_rank_zero_writes_global_evidence_after_synchronized_save(self):
        self.config['num_processes'] = 2
        self.config['training_args']['gradient_checkpointing'] = True
        events, counts = self.run_worker(0, 2)
        self.assertEqual(events[:3], [('device', 0), ('init', {'backend': 'nccl'}), 'barrier'])
        self.assertEqual(events[-6:], ['train', 'barrier', 'save', 'barrier', 'gather', 'destroy'])
        args = next(event[1] for event in events if isinstance(event, tuple) and event[0] == 'trainer_args')
        self.assertEqual(args['local_rank'], 0)
        self.assertEqual(args['ddp_backend'], 'nccl')
        self.assertFalse(args['save_on_each_node'])
        self.assertEqual(args['gradient_accumulation_steps'], 2)
        self.assertFalse(args['ddp_find_unused_parameters'])
        progress = json.loads((self.root / 'progress.json').read_text())
        self.assertEqual(progress['metrics'], {'loss': .5})
        evidence_path = self.root / 'training-evidence.json'
        evidence = json.loads(evidence_path.read_text())
        self.assertEqual(evidence['presented_rows'], 6)
        self.assertEqual(evidence['presented_input_tokens'], 18)
        self.assertEqual(evidence['per_rank_exposure'], [dict(rank=i, **c) for i, c in enumerate(counts)])
        self.assertEqual(evidence['world_size'], 2)
        self.assertEqual(evidence['exposure_scope'], 'all_worker_ranks')
        self.assertFalse(evidence['weight_identity_verified'])
        self.assertEqual(evidence['initialization'], 'pretrained')
        self.assertEqual(evidence['model_kwargs']['device_map'], 'auto')  # Original recipe fingerprint.
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['candidate']['changes']['training_evidence']['digest'],
                         hashlib.sha256(evidence_path.read_bytes()).hexdigest())

    def test_nonzero_rank_does_not_create_output_or_publish_shared_files(self):
        self.config['num_processes'] = 2
        self.config['training_args'].update(gradient_checkpointing=True, ddp_find_unused_parameters=True)
        # Existing rank-zero directory must neither raise nor be recreated by rank one.
        output = self.root / 'adapter'
        output.mkdir()
        events, _ = self.run_worker(1, 2)
        self.assertEqual(events[0], ('device', 1))
        self.assertIn('save', events)
        self.assertIn('gather', events)
        args = next(event[1] for event in events if isinstance(event, tuple) and event[0] == 'trainer_args')
        self.assertTrue(args['ddp_find_unused_parameters'])
        self.assertEqual(list(output.iterdir()), [])
        for name in ('progress.json', 'training-evidence.json', 'result.json'):
            self.assertFalse((self.root / name).exists(), name)

    def test_single_process_keeps_model_kwargs_and_local_exposure(self):
        events, _ = self.run_worker(0, 1)
        self.assertNotIn('barrier', events)
        self.assertNotIn('gather', events)
        self.assertNotIn('destroy', events)
        self.assertIn(('load', 'official', self.config['model_kwargs']), events)
        evidence = json.loads((self.root / 'training-evidence.json').read_text())
        self.assertEqual((evidence['presented_rows'], evidence['presented_input_tokens']), (4, 8))
        self.assertEqual(evidence['exposure_scope'], 'this_worker_rank')

    def test_worker_rechecks_fingerprints_before_distributed_setup(self):
        self.config['num_processes'] = 2
        self.write_recipe()
        _, _, evidence = _native_inputs(asdict(self.request), self.recipe)
        (self.root / 'native-inputs.json').write_text(json.dumps(evidence))
        self.data.write_text('{"input_ids":[4],"labels":[4]}\n')
        with patch('rsisys.adapters.training._worker_group') as group:
            with self.assertRaisesRegex(ValueError, 'changed after submission'):
                self.run_worker(0, 2)
            group.assert_not_called()

    def test_group_cleanup_on_failure_and_missing_cuda(self):
        modules, events, _ = self.fake_frameworks(1, 2)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                with _worker_group(1, 2):
                    raise RuntimeError('failure')
            self.assertEqual(events[-1], 'destroy')
            modules['torch'].cuda.is_available = lambda: False
            with self.assertRaisesRegex(ValueError, 'visible CUDA GPU'):
                with _worker_group(1, 2):
                    self.fail('Must reject CPU distributed training')

    def test_training_failure_cleans_up_without_publishing_result(self):
        self.config['num_processes'] = 2
        modules, events, _ = self.fake_frameworks(0, 2, fail_train=True)
        with patch.object(self, 'fake_frameworks', return_value=(modules, events, [])):
            with self.assertRaisesRegex(RuntimeError, 'synthetic training failure'):
                self.run_worker(0, 2, fail_train=True)
        self.assertEqual(events[-1], 'destroy')
        self.assertNotIn('save', events)
        self.assertFalse((self.root / 'result.json').exists())

    def test_distributed_random_initialization_and_weights_only_origin(self):
        self.config.update(num_processes=2, model_config={'model_type': 'gpt2', 'n_layer': 1})
        origin = TrainingOrigin('parent', BASE, ResumeMode.WEIGHTS_ONLY,
            ArtifactRef('checkpoint', 'peft_adapter', (self.root / 'prior-adapter').as_uri()))
        self.request_file.write_text(json.dumps({'request_type': 'ActorTrainingRequest',
            'request': asdict(replace(self.request, origin=origin))}))
        events, _ = self.run_worker(0, 2)
        self.assertIn(('from_config', {'model_type': 'gpt2', 'n_layer': 1}), events)
        self.assertTrue(any(isinstance(event, tuple) and event[0] == 'resume' and
            event[2] == {'is_trainable': True} for event in events))
        evidence = json.loads((self.root / 'training-evidence.json').read_text())
        self.assertEqual(evidence['initialization'], 'random_from_config')
        self.assertEqual(evidence['initial_model_binding'], asdict(BASE))
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['candidate']['parent_system'], 'parent')

    def test_distributed_worker_rejects_other_parallel_strategies_before_group_init(self):
        environment = {'RSISYS_JOB_DIR': str(self.root), 'RSISYS_REQUEST_FILE': str(self.request_file),
            'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'}
        for key, value in (('deepspeed', 'zero.json'), ('fsdp', 'full_shard'),
                           ('parallelism_config', {'tp_size': 2}), ('use_cpu', True)):
            self.config['training_args'] = {key: value}
            self.write_recipe()
            with self.subTest(key=key), patch.dict(os.environ, environment, clear=True):
                with patch('rsisys.adapters.training._worker_group') as group:
                    with self.assertRaisesRegex(ValueError, 'native CUDA Trainer DDP only'):
                        _transformers_peft(self.recipe)
                    group.assert_not_called()

    def test_import_remains_framework_free(self):
        code = ('import sys; import rsisys.adapters.training; '
                'assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))')
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)), timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class HTTPFixture:
    def __init__(self):
        self.rows = {'base': {'id':'base','root':'base','owned_by':'vllm'}}
        self.requests = []
        self.probe_error = False
        self.mismatch = False
        self.drop_load_response = False
        self.hide = False
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def respond(self, data, status=200, plain=False):
                self.send_response(status)
                self.end_headers()
                self.wfile.write(data.encode() if plain else json.dumps(data).encode())
            def do_GET(self):
                outer.requests.append(('GET',self.path,None))
                self.respond({'data':list(outer.rows.values()) if not outer.hide else []})
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.requests.append(('POST',self.path,data))
                if self.path.endswith('/load_lora_adapter'):
                    alias = data['lora_name']
                    outer.rows[alias] = {'id':alias, 'root':data['lora_path'], 'parent':'base'}
                    self.respond('uncertain' if outer.drop_load_response else 'Success',
                                 503 if outer.drop_load_response else 200, plain=True)
                elif self.path.endswith('/unload_lora_adapter'):
                    outer.rows.pop(data['lora_name'], None)
                    self.respond('Success',plain=True)
                else:
                    if outer.probe_error:
                        self.respond({'error':'not ready'},503)
                    else:
                        self.respond({'model':'wrong' if outer.mismatch else data['model'],
                            'choices':[{'message':{'content':'OK'},'finish_reason':'length'}],
                            'usage':{'prompt_tokens':3,'completion_tokens':1}})
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ServingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = HTTPFixture()
        self.addCleanup(self.fixture.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.binding = BackendBinding('vllm','http',f'http://127.0.0.1:{self.fixture.server.server_port}/v1',
                                      json_mode=False, timeout_seconds=2)
        self.backend = VLLMBackend(self.binding, generation_options={
            'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':16})

    def deployments(self, name='deployments', **kwargs):
        return VLLMDeploymentBackend(self.backend, Path(self.temp.name)/name, **kwargs)

    def test_read_only_native_generation_preserves_unverified_revision(self):
        response = self.backend.generate(GenerationRequest('actor',BASE,'hello'))
        self.assertEqual(response.model,BASE)
        self.assertFalse(response.simulated)
        self.assertFalse(response.revision_verified)
        self.assertEqual(response.usage.output_tokens,1)
        sent = self.fixture.requests[-1][2]
        self.assertEqual(sent['chat_template_kwargs'], {'enable_thinking':False})
        self.assertEqual(sent['temperature'],0)
        self.assertEqual(sent['max_tokens'],16)
        deployments = self.deployments()
        state = deployments.prepare(BASE,idempotency_key='static')
        self.assertTrue(state.ready)
        self.assertTrue(deployments.reattach(state.deployment).ready)
        with self.assertRaises(NotImplementedError):
            deployments.release(state.deployment,idempotency_key='release')
        self.assertFalse(any('lora_adapter' in path for _,path,_ in self.fixture.requests))

    def test_native_token_limit_is_capped_and_probe_is_one_token(self):
        self.backend.generate(GenerationRequest('actor',BASE,'hello',max_output_tokens=2))
        self.assertEqual(self.fixture.requests[-1][2]['max_tokens'],2)
        self.backend.probe(BASE)
        self.assertEqual(self.fixture.requests[-1][2]['max_tokens'],1)
        for options in ({'max_tokens':True}, {'max_tokens':0}, {'max_tokens':float('nan')},
                        {'max_completion_tokens':100}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VLLMBackend(self.binding,generation_options=options)

    def test_owned_load_probe_query_release_and_idempotency(self):
        deployments = self.deployments(adapter_paths={TRAINED:'/owned/adapter'}, allow_runtime_lora=True)
        result = TrainingResult(BASE,TRAINED,Candidate('candidate','r0',{},('actor',)))
        activator = VLLMLoRAActivator(deployments)
        self.assertEqual(activator.activate(result,idempotency_key='load'), TRAINED)
        state = deployments.prepare(TRAINED,idempotency_key='load')
        self.assertTrue(state.ready)
        alias = self.backend.models[TRAINED]
        self.assertTrue(alias.startswith('rsisys-'))
        self.assertEqual(sum(path.endswith('/load_lora_adapter') for _,path,_ in self.fixture.requests),1)
        new = VLLMDeploymentBackend(VLLMBackend(self.binding),Path(self.temp.name)/'deployments',allow_runtime_lora=True)
        before = len(self.fixture.requests)
        self.assertTrue(new.reattach(state.deployment).ready)
        self.assertFalse(any('lora_adapter' in path for _,path,_ in self.fixture.requests[before:]))
        with self.assertRaises(ValueError):
            deployments.prepare(BASE,idempotency_key='load')
        self.assertEqual(new.release(state.deployment,idempotency_key='release').status,DeploymentStatus.RELEASED)
        self.assertEqual(new.release(state.deployment,idempotency_key='release').status,DeploymentStatus.RELEASED)
        self.assertIn('base',self.fixture.rows)
        self.assertNotIn(alias,self.fixture.rows)

    def test_probe_failure_is_pending_and_queries_never_reload(self):
        deployments = self.deployments(adapter_paths={TRAINED:'/owned/adapter'},allow_runtime_lora=True)
        self.fixture.probe_error = True
        result = TrainingResult(BASE,TRAINED,Candidate('c','r0',{},('actor',)))
        with self.assertRaises(ModelActivationPending):
            VLLMLoRAActivator(deployments).activate(result,idempotency_key='load')
        self.fixture.probe_error = False
        self.assertEqual(VLLMLoRAActivator(deployments).activate(result,idempotency_key='load'),TRAINED)
        self.assertEqual(sum(path.endswith('/load_lora_adapter') for _,path,_ in self.fixture.requests),1)

    def test_ambiguous_load_recovers_by_query_without_second_post(self):
        deployments = self.deployments(adapter_paths={TRAINED:'/owned/adapter'},allow_runtime_lora=True)
        self.fixture.drop_load_response = True
        state = deployments.prepare(TRAINED,idempotency_key='lost-response')
        self.assertEqual(state.status,DeploymentStatus.UNKNOWN)
        self.assertTrue(deployments.reattach(state.deployment).ready)
        self.assertTrue(deployments.prepare(TRAINED,idempotency_key='lost-response').ready)
        self.assertEqual(sum(path.endswith('/load_lora_adapter') for _,path,_ in self.fixture.requests),1)

    def test_path_mismatch_prevents_readiness_or_unloading(self):
        deployments = self.deployments(adapter_paths={TRAINED:'/owned/adapter'},allow_runtime_lora=True)
        state = deployments.prepare(TRAINED,idempotency_key='load')
        alias = self.backend.models[TRAINED]
        self.fixture.rows[alias]['root'] = '/someone-else/adapter'
        self.assertEqual(deployments.query(state.deployment).status,DeploymentStatus.UNKNOWN)
        self.assertEqual(deployments.release(state.deployment,idempotency_key='release').status,DeploymentStatus.UNKNOWN)
        self.assertFalse(any(path.endswith('/unload_lora_adapter') for _,path,_ in self.fixture.requests))

    def test_runtime_mutation_opt_in_identity_and_server_scope(self):
        with self.assertRaises(NotImplementedError):
            self.deployments(adapter_paths={TRAINED:'/adapter'}).prepare(TRAINED,idempotency_key='no')
        with self.assertRaises(ValueError):
            VLLMBackend(self.binding,generation_options={'model':'override'})
        self.backend.bind(BASE,'base')
        with self.assertRaises(ValueError):
            self.backend.bind(TRAINED,'base')
        self.fixture.mismatch=True
        with self.assertRaises(ModelResponseError):
            self.backend.generate(GenerationRequest('actor',BASE,'hello'))
        self.assertEqual(self.deployments('other').prepare(BASE,idempotency_key='bad-probe').status,DeploymentStatus.UNKNOWN)
        with self.assertRaises(ValueError):
            VLLMDeploymentBackend(VLLMBackend(replace(self.binding,endpoint='http://127.0.0.1:1/v1')),
                                  Path(self.temp.name)/'other')


if __name__ == '__main__':
    unittest.main()
