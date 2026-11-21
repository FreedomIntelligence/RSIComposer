"""Run work briefs and the original Harbor deliverable verifiers."""

from rsisys.labs.coding.lab import CodingLab


class ProfessionalWorkLab(CodingLab):
    domain = 'professional_work'
    benchmark_kind = 'professional-deliverable'
    label = 'Professional work'
    protocol_version = 'professional-work-environment/v1'
