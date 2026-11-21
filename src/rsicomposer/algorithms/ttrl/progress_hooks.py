"""Observe the pinned trainer without changing its optimizer or training loop.

Installed in the official TaskRunner via Ray's public worker setup hook. All
training still runs in the upstream body. A checkpoint pause happens after the
original logger records that step, before the next optimizer step can start.
"""
from dataclasses import asdict
import json
import math
import numbers
import os
from pathlib import Path
import time

from rsicomposer.api import JobProgress
from .native_jobs import atomic_json, read_events


class StopAtCheckpoint(Exception):
    """Only this acknowledged checkpoint boundary is a successful early stop."""


class ProgressBridge:
    def __init__(self, directory, call_id, total_steps):
        self.directory, self.call_id = Path(directory), call_id
        self.total_steps = total_steps
        self.started = time.time()
        existing = read_events(directory)
        self.sequence = existing[-1]['sequence'] if existing else 0
        self.saved = {}

    def saved_checkpoint(self, trainer):
        step = trainer.global_steps
        actor = self.directory / 'checkpoints' / f'global_step_{step}' / 'actor'
        if not actor.is_dir():
            raise RuntimeError('Official save did not publish the expected actor checkpoint')
        self.saved[step] = str(actor)

    def emit(self, kind, progress, checkpoint=None):
        self.sequence += 1
        event = dict(sequence=self.sequence, kind=kind, progress=asdict(progress), checkpoint=checkpoint)
        atomic_json(self.directory / 'events' / f'{self.sequence:012d}.json', event)
        return self.sequence

    def log(self, data, step):
        # Evaluation log calls at step zero are not optimizer progress.
        if type(step) is not int or step < 1:
            return
        metrics = {key: float(value) for key, value in data.items()
                   if isinstance(key, str) and key.strip() and isinstance(value, numbers.Real)
                   and not isinstance(value, bool) and math.isfinite(value)}
        now = time.time()
        progress = JobProgress(step=step, total_steps=self.total_steps, metrics=metrics,
                               elapsed_seconds=now - self.started, observed_at=now)
        self.emit('progress', progress)
        if step not in self.saved:
            return
        sequence = self.emit('checkpoint', progress, dict(step=step, actor=self.saved.pop(step),
            format='verl-fsdp', upstream_managed=True))
        control_path = self.directory / 'controls' / f'{sequence:012d}.json'
        while not control_path.exists():
            time.sleep(.1)
        control = json.loads(control_path.read_text())
        if control.get('call_id') != self.call_id or control.get('sequence') != sequence:
            raise ValueError('Checkpoint control belongs to a different job or event')
        if control.get('action') == 'stop':
            atomic_json(self.directory / 'stop-ack.json', dict(call_id=self.call_id, sequence=sequence, step=step))
            raise StopAtCheckpoint(f'Controller stopped native schedule at step {step}')
        if control.get('action') != 'continue':
            raise ValueError('Unknown native checkpoint control')


def install():
    directory = os.environ.get('RSISYS_TTRL_JOB')
    if not directory:
        return
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.tracking import Tracking
    if getattr(RayPPOTrainer, '_rsisys_progress_installed', False):
        return
    request = json.loads((Path(directory) / 'request.json').read_text())
    total_steps = len(request['items']) // request['config']['train_batch_size'] * request['config']['total_epochs']
    bridge = ProgressBridge(directory, request['call_id'], total_steps)
    original_save, original_log = RayPPOTrainer._save_checkpoint, Tracking.log

    def save(trainer, *args, **kwargs):
        result = original_save(trainer, *args, **kwargs)
        bridge.saved_checkpoint(trainer)
        return result

    def log(tracking, data, step, *args, **kwargs):
        result = original_log(tracking, data, step, *args, **kwargs)
        bridge.log(data, step)
        return result

    RayPPOTrainer._save_checkpoint = save
    Tracking.log = log
    RayPPOTrainer._rsisys_progress_installed = True
