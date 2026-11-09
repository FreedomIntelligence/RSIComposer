"""Feedback learning independent of any Decision harness implementation."""
from rsisys.harnesses.decision.feedback.components import (
    ContextMemoryOrganizer,
    JournalContextMemory,
    JournalTrajectoryArchive,
)
from rsisys.harnesses.decision.feedback.context_memory import CallbackContextMemory
from rsisys.harnesses.decision.feedback.knowledge import WorkingMemoryPolicy
from rsisys.harnesses.decision.feedback.learning import (
    FeedbackLearning,
    SerializedLearningBackend,
)
from rsisys.harnesses.decision.feedback.pipeline import (
    FeedbackPipeline,
    enable_feedback_pipeline,
)
from rsisys.harnesses.decision.feedback.proposals import (
    ModelHarnessProposer,
    ModelImprovementProposer,
)
from rsisys.harnesses.decision.feedback.scheduling import FifoFeedbackScheduler
from rsisys.harnesses.decision.feedback.service import (
    FeedbackAdapter,
    FeedbackEvidence,
    FeedbackLearningService,
)
from rsisys.harnesses.decision.feedback.stages import CallbackFeedbackProcessor
from rsisys.loops.contracts.feedback_context import (
    ContextMemory,
    ContextQuery,
    FeedbackContextEvent,
)
from rsisys.loops.contracts.feedback_modules import (
    ArchiveEvent,
    FeedbackFailurePolicy,
    FeedbackModel,
    FeedbackProcessor,
    FeedbackProposer,
    FeedbackScheduler,
    PendingFeedback,
    TrajectoryArchive,
)


def enable_langmem_feedback(*args, **kwargs):
    """Load the optional upstream integration only when explicitly installed."""
    from rsisys.harnesses.decision.feedback.langmem import (
        enable_langmem_feedback as install,
    )
    return install(*args, **kwargs)

__all__ = ["FeedbackAdapter", "FeedbackEvidence", "FeedbackLearningService",
           "FeedbackLearning", "SerializedLearningBackend", "WorkingMemoryPolicy",
           "ContextMemoryOrganizer", "TrajectoryArchive", "JournalContextMemory", "JournalTrajectoryArchive",
           "ContextMemory", "ContextQuery", "FeedbackContextEvent", "CallbackContextMemory",
           "ArchiveEvent", "FeedbackProposer", "FeedbackModel", "FeedbackScheduler", "PendingFeedback",
           "FeedbackFailurePolicy", "CallbackFeedbackProcessor", "FifoFeedbackScheduler",
           "FeedbackProcessor", "FeedbackPipeline", "ModelHarnessProposer", "ModelImprovementProposer",
           "enable_feedback_pipeline", "enable_langmem_feedback"]
