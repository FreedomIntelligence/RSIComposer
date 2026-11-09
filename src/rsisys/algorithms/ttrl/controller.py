"""TTRL majority-reward GRPO and native continuous-training decisions."""
from rsisys.controllers.receipt_schedule import ReceiptScheduleController


class TTRLController(ReceiptScheduleController):
    """One on-policy majority-reward GRPO update per round, no gold rewards."""
    method, option = 'ttrl', 'samples_per_prompt'
    default_width = 16
    minimum_width = 2
    phases = ('rollout', 'reward', 'train')


class TTRLVerlController(TTRLController):
    """Run one native schedule without resetting its optimizer/reference state."""
    phases = ('fit',)

    def __init__(self, *, iterations=1, **config):
        super().__init__(iterations=iterations, **config)

    def report(self):
        return {**super().report(), 'backend': 'official-verl'}

    @classmethod
    def validate_config(cls, config):
        super().validate_config({'iterations': 1, **config})
        if config.get('iterations', 1) != 1:
            raise ValueError('ttrl_verl runs one continuous native schedule; configure total_epochs in the Lab')
