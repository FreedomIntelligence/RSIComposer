"""Harbor-backed evaluation of professional deliverables."""


def register_components(registry):
    from .lab import ProfessionalWorkLab
    registry.register_lab('professional_work', ProfessionalWorkLab, version='1',
                          validate_config=ProfessionalWorkLab.validate_config)
