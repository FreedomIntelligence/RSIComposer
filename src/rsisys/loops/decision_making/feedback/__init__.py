"""Feedback learning independent of any Decision harness implementation."""
from .service import FeedbackAdapter, FeedbackEvidence, FeedbackLearningService
from .learning import FeedbackLearning, SerializedLearningBackend
from .knowledge import WorkingMemoryPolicy
from .components import (ContextMemoryOrganizer, TrajectoryArchive, FeedbackProcessor,
                         JournalContextMemory, JournalTrajectoryArchive)
from rsisys.loops.contracts.feedback_context import ContextMemory, ContextQuery, FeedbackContextEvent
from .context_memory import CallbackContextMemory
from rsisys.loops.contracts.feedback_modules import (
    ArchiveEvent, FeedbackProposer, FeedbackModel, FeedbackScheduler, PendingFeedback, FeedbackFailurePolicy)
from .stages import CallbackFeedbackProcessor
from .scheduling import FifoFeedbackScheduler
from .pipeline import FeedbackPipeline, enable_feedback_pipeline
from .proposals import ModelHarnessProposer, ModelImprovementProposer


def enable_langmem_feedback(*args, **kwargs):
    """Load the optional upstream integration only when explicitly installed."""
    from .langmem import enable_langmem_feedback as install
    return install(*args, **kwargs)

__all__ = ["FeedbackAdapter", "FeedbackEvidence", "FeedbackLearningService",
           "FeedbackLearning", "SerializedLearningBackend", "WorkingMemoryPolicy",
           "ContextMemoryOrganizer", "TrajectoryArchive", "JournalContextMemory", "JournalTrajectoryArchive",
           "ContextMemory", "ContextQuery", "FeedbackContextEvent", "CallbackContextMemory",
           "ArchiveEvent", "FeedbackProposer", "FeedbackModel", "FeedbackScheduler", "PendingFeedback",
           "FeedbackFailurePolicy", "CallbackFeedbackProcessor", "FifoFeedbackScheduler",
           "FeedbackProcessor", "FeedbackPipeline", "ModelHarnessProposer", "ModelImprovementProposer",
           "enable_feedback_pipeline", "enable_langmem_feedback"]
