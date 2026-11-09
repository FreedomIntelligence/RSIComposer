"""SEAL self-edit, inner adaptation, selection and outer-SFT decisions."""
from rsisys.loops.decision_making.training_schedule import TrainingScheduleController


class SEALController(TrainingScheduleController):
    """Top-1 self-edit ReST-EM; outer SFT starts from the CURRENT policy."""
    method, option = 'seal', 'candidates'
    default_width = 2
    phases = ('generate', 'adapt', 'select', 'train')
