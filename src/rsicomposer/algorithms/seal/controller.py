"""SEAL self-edit, inner adaptation, selection and outer-SFT decisions."""
from rsicomposer.algorithms.shared.training_schedule import TrainingScheduleController


class SEALController(TrainingScheduleController):
    """Top-1 self-edit ReST-EM; outer SFT starts from the CURRENT policy."""
    method, option = 'seal', 'candidates'
    default_width = 2
    phases = ('generate', 'adapt', 'select', 'train')
