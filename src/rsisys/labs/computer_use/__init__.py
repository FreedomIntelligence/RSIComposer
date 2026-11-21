"""Browser application state benchmarks."""


def register_components(registry):
    from .lab import ComputerUseLab
    registry.register_lab('computer_use', ComputerUseLab, version='1',
                          validate_config=ComputerUseLab.validate_config)
