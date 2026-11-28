"""Small context window shared by memory providers and consuming harnesses.

Providers own organization, retrieval and storage. These inputs contain public
observations, never a feedback worker's private job/accounting representation.
"""
from dataclasses import dataclass, field
import re
from typing import Protocol


EVIDENCE_SOURCES = frozenset({'lab', 'operation', 'controller'})


@dataclass(frozen=True)
class EvidenceEvent:
    """One fact recorded for a run: a Lab receipt or a Controller-side event.

    source is 'lab' or 'operation' for execution receipts
    (kind='receipt'), and 'controller' for events a policy records through
    AssemblyContext.record, e.g. kind='candidate' or 'interpretation'.
    output and references keep their native shape; the host supplies metadata.
    event_id is stable across delivery attempts, so a Memory may deduplicate by it.
    For product processors, evidence_packet is the host's bounded projection,
    source_refs names admitted citations, and candidate_evidence binds supplied
    evaluations to their exact candidates. These do not expose worker state,
    accounting reservations or execution authority.
    """
    event_id: str
    source: str
    kind: str = 'receipt'
    task_id: str = ''
    status: str = ''
    output: object = None
    request: object = None
    references: object = ()
    decision_id: str | None = None
    identity: dict = field(default_factory=dict)
    evidence_packet: dict = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    candidate_evidence: dict | None = None

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError('event_id must be a nonempty string')
        if self.source not in EVIDENCE_SOURCES:
            raise ValueError(f'Unknown evidence source: {self.source!r}')
        if not isinstance(self.kind, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', self.kind):
            raise ValueError('kind must be a short lowercase identifier')
        if self.decision_id is not None and (not isinstance(self.decision_id, str) or not self.decision_id.strip()):
            raise ValueError('decision_id must be a nonempty string or None')
        if (not isinstance(self.evidence_packet, dict) or not isinstance(self.source_refs, tuple)
                or any(not isinstance(ref, str) or not ref for ref in self.source_refs)
                or len(set(self.source_refs)) != len(self.source_refs)
                or (self.candidate_evidence is not None and not isinstance(self.candidate_evidence, dict))):
            raise TypeError('Evidence packet, source references and candidate evidence have invalid shapes')


@dataclass(frozen=True)
class ContextQuery:
    objective: str = ''
    latest_feedback: EvidenceEvent | None = None


class ControllerMemory(Protocol):
    """A Controller's memory: written from evidence, read as decision context.

    context(query) is the only required method; an empty string is a valid view.
    Optional methods:
      observe(event: EvidenceEvent) -> None   receives Lab receipts on a later host
                                              tick and Controller events as recorded;
      capabilities() -> Iterable[Capability]  contributes tools with validators and handlers;
                                              the Host enforces ordinary operation permissions.
      query(request: dict) -> dict            typed retrieval for policies that need
                                              more than text;
      rebuild(events) -> None                 replays evidence into an empty memory.
      export() / restore(state) -> None       capture and replace owned state.
      restore_checkpoint(state) -> None       optional same-workspace recovery;
                                              verify any later committed state.
      seed_version(state) -> None          explicitly imports candidate-version
                                              memory into its new logical scope;
                                              durable restore remains separate.
    Internal model use goes through the deployment's accounted dependencies.
    """
    def context(self, query: ContextQuery) -> str: ...
