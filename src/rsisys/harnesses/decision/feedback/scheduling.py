"""A default scheduling policy; executor and host admission are separate."""
class FifoFeedbackScheduler:
    def select(self, pending, running):
        return pending[0].event.event_id if pending else None


def validate_scheduler(scheduler):
    if not callable(getattr(scheduler, 'select', None)):
        raise TypeError('Feedback scheduler requires select(pending, running)')
