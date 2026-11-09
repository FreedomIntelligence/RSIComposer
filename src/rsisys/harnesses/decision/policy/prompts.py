"""Replaceable prompt text; action admission and execution remain host code."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from string import Template

_NAMES = ('research', 'agent_memory', 'decision', 'review', 'summary', 'control', 'scheduling',
          'attempt_optional', 'attempt_required', 'validation_correction')


@dataclass(frozen=True, init=False)
class DecisionPrompts:
    """Capture a bundle at assembly time, with individual text overrides.

    Templates use $name placeholders so JSON examples retain literal braces.
    Only the placeholders supported by each built-in template are accepted.
    Overrides cannot grant capabilities or change parser/executor constraints.
    """
    _templates: tuple[tuple[str, str], ...]
    is_default: bool

    def __init__(self, overrides: Mapping[str, str] | None = None):
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError('Prompt overrides must be a mapping')
        overrides = dict(overrides) if overrides is not None else {}
        if set(overrides) - set(_NAMES):
            raise ValueError('Unknown decision prompt name')
        folder = resources.files(__package__).joinpath('prompts')
        defaults = {name: folder.joinpath(name + '.txt').read_text(encoding='utf-8').rstrip('\n')
                    for name in _NAMES}
        templates = {**defaults, **overrides}
        for name, value in templates.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Decision prompt {name!r} must be nonempty text')
            template = Template(value)
            if (not template.is_valid() or set(template.get_identifiers()) -
                    set(Template(defaults[name]).get_identifiers())):
                raise ValueError(f'Unsupported placeholder in decision prompt {name!r}')
        object.__setattr__(self, '_templates', tuple(templates.items()))
        object.__setattr__(self, 'is_default', templates == defaults)

    def render(self, name: str, **values: str) -> str:
        try:
            text = dict(self._templates)[name]
        except KeyError:
            raise ValueError(f'Unknown decision prompt {name!r}') from None
        return Template(text).substitute(values)

    def manifest(self) -> dict:
        """A fresh audit manifest; full effective text remains in each request."""
        return {'provider': 'decision_prompts', 'version': 1, 'is_default': self.is_default,
                'sha256': {name: sha256(text.encode('utf-8')).hexdigest()
                           for name, text in self._templates}}
