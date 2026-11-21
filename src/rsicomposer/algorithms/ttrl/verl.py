"""Registered TTRL Lab delegating training to the pinned official verl fork.

The separately installed TTRL checkout is MIT at its root. This adapter
uses the bundled verl backend, which is licensed under Apache-2.0:
https://github.com/PRIME-RL/TTRL/blob/5806e119789132fcb43552beaf2532d7d0447213/verl/LICENSE
"""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from rsicomposer.algorithms.ttrl.lab import TTRLLab
from rsicomposer.algorithms.ttrl.lab import validate_config as validate_reference
from rsicomposer.api import (ArtifactRef, ExecutionStatus, JobOutcome, JobProgress, JobRef, JobStatus,
                        OperationProgress, OperationResult, WeightEngine, WorkerSpec, file_hash)
from .native_jobs import atomic_json, control_checkpoint, latest_progress, process_identity, read_events

from .upstream import TTRL_REVISION, TTRLReward, checkout

DEFAULTS = dict(total_epochs=10, train_batch_size=32, samples_per_prompt=32,
    eval_samples=16, evaluation_temperature=0.6, evaluation_top_p=0.95,
    temperature=0.6, top_p=1.0, max_tokens=3072, max_length=4096,
    apply_chat_template=True, save_freq=2000000)
NATIVE_OPTIONS = {'upstream_root', 'total_epochs', 'train_batch_size', 'samples_per_prompt', 'native_overrides', 'save_freq'}


def validate_config(config):
    merged = {**DEFAULTS, **config}
    validate_reference({k: v for k, v in merged.items() if k not in NATIVE_OPTIONS})
    if not isinstance(merged.get('upstream_root'), str) or not merged['upstream_root'].strip():
        raise ValueError('ttrl_verl requires upstream_root pointing to the pinned TTRL checkout')
    for key in ('total_epochs', 'train_batch_size', 'samples_per_prompt', 'save_freq'):
        if type(merged[key]) is not int or merged[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if merged['samples_per_prompt'] < 2:
        raise ValueError('TTRL training requires at least two responses')
    overrides = merged.get('native_overrides', {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k for k in overrides):
        raise ValueError('native_overrides must be a mapping of Hydra config keys to JSON values')
    reserved = ('data.train_files', 'data.val_files', 'data.train_batch_size', 'custom_reward_function', 'ttrl',
                'trainer.default_local_dir', 'trainer.resume', 'trainer.total',
                'trainer.save_freq', 'trainer.test_freq', 'trainer.val_before_train',
                'actor_rollout_ref.model.path', 'actor_rollout_ref.model.lora_rank', 'trainer.n_gpus_per_node', 'trainer.nnodes')
    if any(any(k == r or k.startswith(r + '.') or k.startswith(r + '_') or r.startswith(k + '.')
               for r in reserved) for k in overrides):
        raise ValueError('native_overrides cannot replace owned data, reward, model, schedule or publication paths')


class VerlTTRLEngine(WeightEngine):
    name = 'ttrl-official-verl-' + TTRL_REVISION
    worker_module = 'rsicomposer.algorithms.ttrl.worker'

    def bind_execution(self, context):
        resources = getattr(context, 'resources', None)
        owner = getattr(resources, 'owner', resources)
        if getattr(owner, 'config', {}).get('executor', 'local') == 'ray':
            raise ValueError('ttrl_verl Ray placement-group integration is deferred; use executor=local')
        super().bind_execution(context)

    def submit_fit(self, model, rows, directory, checkpoint, *, votes, call_id):
        with self.resources.acquire('trainer') as lease:
            root = checkout(self.config['upstream_root'], TTRL_REVISION)
            directory.mkdir(parents=True, exist_ok=False)
            output, request = directory / 'response.json', directory / 'request.json'
            atomic_json(request, dict(config={**self.config, 'upstream_root': str(root)}, world_size=len(lease.devices), call_id=call_id,
                model=model, items=rows, checkpoint=str(checkpoint), votes=votes, output=str(output)))
            env = dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(map(str, lease.devices)),
                VLLM_USE_V1='1', WANDB_MODE='disabled')
            cuda_home = self.config.get('training_cuda_home')
            if cuda_home:
                env.update(CUDA_HOME=cuda_home, PATH=cuda_home + '/bin' + os.pathsep + env.get('PATH', ''),
                    LD_LIBRARY_PATH=cuda_home + '/lib64' + os.pathsep + env.get('LD_LIBRARY_PATH', ''))
            command = [self.config.get('training_python', sys.executable), '-m',
                       self.worker_module, str(request)]
            with (directory / 'worker.log').open('xb') as log:
                process = self.executor.launch(WorkerSpec(command, env, stdout=log,
                    detached=True), lease)
            atomic_json(directory / 'job.json', dict(call_id=call_id, pid=process.pid,
                start_ticks=process_identity(process.pid), submitted_at=time.time()))
            return JobRef(str(directory), self.name)

    def _descriptor(self, job):
        if job.backend != self.name:
            raise ValueError('Native job backend does not match this engine')
        directory = Path(job.job_id)
        descriptor = json.loads((directory / 'job.json').read_text())
        return directory, descriptor

    def _alive(self, descriptor):
        identity = process_identity(descriptor['pid'])
        return identity is not None and identity == descriptor['start_ticks']

    def reattach_fit(self, job):
        directory, descriptor = self._descriptor(job)
        if not (directory / 'status.json').exists() and not self._alive(descriptor):
            raise RuntimeError('Native worker is gone or its PID identity changed; no job was resubmitted')

    def poll_fit(self, job):
        directory, descriptor = self._descriptor(job)
        status_path = directory / 'status.json'
        progress = latest_progress(directory)
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['status'] == 'succeeded':
                return JobOutcome(job, JobStatus.SUCCEEDED, json.loads((directory / 'response.json').read_text()),
                                  simulated=self.simulated, progress=progress)
            return JobOutcome(job, JobStatus.FAILED, error=status['error'], simulated=self.simulated, progress=progress)
        if not self._alive(descriptor):
            return JobOutcome(job, JobStatus.FAILED, error='Native worker exited without a terminal publication',
                              simulated=self.simulated, progress=progress)
        return JobOutcome(job, JobStatus.RUNNING, simulated=self.simulated, progress=progress)

    def fit_events(self, job, *, after):
        directory, _ = self._descriptor(job)
        return read_events(directory, after=after)

    def control_fit(self, job, control):
        directory, descriptor = self._descriptor(job)
        if control.call_id != descriptor['call_id']:
            raise ValueError('Control belongs to a different native call')
        control_checkpoint(directory, control.call_id, control.sequence, control.action)


class TTRLVerlLab(TTRLLab):
    engine_methods = ('generate', 'submit_fit', 'poll_fit', 'fit_events', 'control_fit', 'reattach_fit')
    fields = {'evaluate': {'dataset': 'string', 'split': 'string'},
              'fit': {'dataset': 'string', 'samples_per_prompt': 'integer', 'reference': 'string'}}
    allow_empty_transfer = True
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config, *, engine=None, verifier=None):
        self.native_jobs = {}
        self.artifacts = context.artifacts
        config = {**DEFAULTS, **config}
        validate_config(config)
        if verifier is None:
            verifier = TTRLReward(config['upstream_root'])
        super().__init__(context, config, engine=engine or VerlTTRLEngine(config), verifier=verifier)

    def profiles(self):
        # Native fit is a continuous upstream job, not a generate/train split.
        return ()

    def invariants(self):
        return ()

    def _extra_state(self):
        return {**super()._extra_state(), 'native_jobs': deepcopy(self.native_jobs)}

    def _restore_extra(self, state):
        super()._restore_extra(state)
        self.native_jobs = deepcopy(state['native_jobs'])

    def _execute_operation(self, call):
        if call.name == 'fit':
            return self._fit(call.arguments, call.call_id)
        return super()._execute_operation(call)

    def _fit(self, args, call_id):
        if self.path(args['reference']) != self.path(args['parent']):
            raise ValueError('Native TTRL initializes actor and reference from the same explicit model')
        rows = self.read(args['dataset'])
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) or set(r) != {'id', 'prompt'}
                or any(not isinstance(r[k], str) or not r[k].strip() for k in ('id', 'prompt')) for r in rows):
            raise ValueError('TTRL learning file must contain only id and prompt; gold labels are forbidden')
        prompts = {r['prompt'].strip() for r in rows}
        if len(prompts) != len(rows) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate adaptation questions')
        if self.target_prompts is None or prompts != self.target_prompts or prompts & self.transfer_prompts:
            raise ValueError('Adaptation must exactly match target questions and exclude transfer questions')
        if len(rows) < self.config['train_batch_size']:
            raise ValueError('Native drop-last dataloader would have no training batch')
        if args['samples_per_prompt'] < self.config['samples_per_prompt']:
            raise ValueError('Vote count must be at least the training sample count')
        checkpoint = self.root / f'round-{args["round"]}' / 'model'
        directory = self.root / f'round-{args["round"]}' / 'native-workers'
        job = self.engine.submit_fit(str(self.path(args['parent'])), rows,
            directory, checkpoint, votes=args['samples_per_prompt'], call_id=call_id)
        if not isinstance(job, JobRef):
            raise TypeError('Native engine submission must return JobRef')
        self.native_jobs[job.job_id] = dict(job=job, call_id=call_id, request=deepcopy(args),
            directory=str(directory), checkpoint=str(checkpoint), artifacts={})
        return job

    def _job(self, job):
        entry = self.native_jobs[job.job_id]
        if entry['job'] != job:
            raise ValueError('Native job reference changed')
        return entry

    def reattach_job(self, job):
        self._job(job)
        self.engine.reattach_fit(job)

    def job_events(self, job, *, after):
        entry = self._job(job)
        events = []
        for event in self.engine.fit_events(job, after=after):
            ref = None
            if event['kind'] == 'checkpoint':
                key = str(event['sequence'])
                # Keep the identity stable even if the host dies before checkpointing its Lab state.
                marker = Path(entry['directory']) / f'artifact-{key}.json'
                if key not in entry['artifacts']:
                    if marker.exists():
                        ref = ArtifactRef(**json.loads(marker.read_text()))
                        self.artifacts.read(ref)
                    else:
                        ref = self.artifacts.put(json.dumps(event['checkpoint'], sort_keys=True).encode(),
                                                 kind='ttrl-verl-checkpoint')
                        atomic_json(marker, asdict(ref))
                    entry['artifacts'][key] = ref
                ref = entry['artifacts'][key]
            events.append(OperationProgress(entry['call_id'], job, event['sequence'], event['kind'],
                          JobProgress(**event['progress']), ref))
        return tuple(events)

    def control_job(self, job, control):
        entry = self._job(job)
        if control.call_id != entry['call_id']:
            raise ValueError('Control belongs to a different native operation')
        self.engine.control_fit(job, control)

    def poll_job(self, job):
        entry = self._job(job)
        outcome = self.engine.poll_fit(job)
        if not outcome.terminal:
            return outcome
        args = entry['request']
        try:
            if outcome.status != JobStatus.SUCCEEDED:
                result = OperationResult('Native TTRL failed', (), self.simulated, ExecutionStatus.FAILED,
                    outcome.error or outcome.status.value, data={'request': args})
            else:
                output = self._publish(args, Path(entry['checkpoint']), outcome.result)
                result = OperationResult('Native TTRL completed', (), self.simulated, data={'request': args, **output})
        except Exception as exc:
            result = OperationResult('Native TTRL publication failed', (), self.simulated, ExecutionStatus.FAILED,
                f'{type(exc).__name__}: {exc}', data={'request': args})
        atomic_json(self.root / f'round-{args["round"]}' / 'fit-receipt.json',
                    dict(status=result.status.value, error=result.error, data=result.data))
        return JobOutcome(job, JobStatus.SUCCEEDED, result, simulated=self.simulated, progress=outcome.progress)

    def _publish(self, args, checkpoint, metrics):
        weights = sorted(checkpoint.glob('*.safetensors'))
        if not weights or not metrics.get('weights_changed') or metrics.get('steps', 0) < 1:
            raise ValueError('Native TTRL did not publish updated weights')
        manifest = dict(parent=str(self.path(args['parent'])), reference=str(self.path(args['reference'])), initial_model=self.initial_model,
            dataset_sha256=file_hash(self.path(args['dataset'])), engine=self.engine.name,
            algorithm='ttrl-native-grpo', recipe=self.config, metrics=metrics,
            weights={p.name: file_hash(p) for p in weights}, verifier=self.verifier.name)
        atomic_json(checkpoint / 'provenance.json', manifest)
        self.models[str(checkpoint)] = manifest
        return {'model': str(checkpoint), 'training_performed': not self.simulated, 'training': metrics,
                'provenance_sha256': file_hash(checkpoint / 'provenance.json')}
