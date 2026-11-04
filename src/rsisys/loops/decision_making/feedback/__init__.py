"""Feedback learning independent of any Decision harness implementation."""
from .service import FeedbackAdapter, FeedbackEvidence, FeedbackLearningService
from .learning import FeedbackLearning, SerializedLearningBackend
from .knowledge import WorkingMemoryPolicy
from .components import (ContextMemoryOrganizer, TrajectoryArchive, FeedbackProcessor,
                         JournalContextMemory, JournalTrajectoryArchive)
from .pipeline import FeedbackPipeline, enable_feedback_pipeline
from .proposals import ModelHarnessProposer


def enable_langmem_feedback(*args, **kwargs):
    """Load the optional upstream integration only when explicitly installed."""
    from .langmem import enable_langmem_feedback as install
    return install(*args, **kwargs)

__all__ = ["FeedbackAdapter", "FeedbackEvidence", "FeedbackLearningService",
           "FeedbackLearning", "SerializedLearningBackend", "WorkingMemoryPolicy",
           "ContextMemoryOrganizer", "TrajectoryArchive", "JournalContextMemory", "JournalTrajectoryArchive",
           "FeedbackProcessor", "FeedbackPipeline", "ModelHarnessProposer",
           "enable_feedback_pipeline", "enable_langmem_feedback"]
