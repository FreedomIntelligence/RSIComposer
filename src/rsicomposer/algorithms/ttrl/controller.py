"""TTRL majority-reward GRPO and native continuous-training decisions."""
from rsicomposer.algorithms.shared.training_schedule import TrainingScheduleController
from rsicomposer.api import Decision, JobControl, emit_observation, observation_id


class TTRLController(TrainingScheduleController):
    """One on-policy majority-reward GRPO update per round, no gold rewards."""
    method, option = 'ttrl', 'samples_per_prompt'
    default_width = 16
    minimum_width = 2
    phases = ('generate', 'reward', 'train')
    default_temperature = .6
    requires_reference = True


class TTRLVerlController(TTRLController):
    """Run one native schedule without resetting its optimizer/reference state."""
    phases = ('fit',)

    def required_profiles(self):
        # Keep one continuous native schedule rather than a generate/train split.
        return {}

    _STATE = (*TTRLController._STATE, 'checkpoints')

    def __init__(self, *, iterations=1, stop_after_checkpoints=None, **config):
        self._validate_stop(stop_after_checkpoints)
        self.stop_after_checkpoints = stop_after_checkpoints
        self.checkpoints = []
        super().__init__(iterations=iterations, **config)

    @staticmethod
    def _validate_stop(value):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('stop_after_checkpoints must be positive or null')

    def decide(self, window):
        if self.pending is not None and self.phase == 'fit':
            feedbacks = [result for result in window.feedbacks if result.call_id == self.pending.call_id]
            if not feedbacks:
                for event in getattr(window, 'progresses', ()):
                    if event.call_id == self.pending.call_id and event.kind == 'checkpoint':
                        if any(item['sequence'] == event.sequence for item in self.checkpoints):
                            continue
                        action = ('stop' if self.stop_after_checkpoints is not None
                                  and len(self.checkpoints) + 1 >= self.stop_after_checkpoints else 'continue')
                        self.checkpoints.append(dict(sequence=event.sequence, step=event.progress.step, action=action))
                        ident = observation_id('ttrl_verl.checkpoint', event.call_id, event.sequence)
                        emit_observation('ttrl_verl.checkpoint', ident, scope=event.call_id, subject=ident,
                                         fields={'step': event.progress.step, 'sequence': event.sequence})
                        emit_observation('ttrl_verl.control', observation_id('ttrl_verl.control', ident),
                                         scope=event.call_id, subject=ident, links={'controls': [ident]},
                                         fields={'action': action, 'step': event.progress.step})
                        return Decision('control', control=JobControl(event.call_id, event.sequence, action))
                return Decision('wait', reason='Native TTRL schedule is running')
        return super().decide(window)

    def report(self):
        return {**super().report(), 'backend': 'official-verl', 'checkpoints': list(self.checkpoints)}

    @classmethod
    def validate_config(cls, config):
        config = dict(config)
        cls._validate_stop(config.pop('stop_after_checkpoints', None))
        super().validate_config({'iterations': 1, **config})
        if config.get('iterations', 1) != 1:
            raise ValueError('ttrl_verl runs one continuous native schedule; configure total_epochs in the Lab')
