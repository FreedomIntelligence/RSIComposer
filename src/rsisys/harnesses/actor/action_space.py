"""Echo and local-command actions provided by the bundled Actor harness."""
from pathlib import Path

from rsisys.loops.exploration.action_space import Action, ActionProposal, ActionSpec


class EchoActionSpace:
    def describe(self) -> tuple[ActionSpec, ...]:
        return (ActionSpec("echo", "Echo a string in the task environment", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False,
        }),)

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action:
        if proposal.name != "echo" or proposal.name not in allowed_actions:
            raise ValueError(f"Unsupported or disallowed action: {proposal.name}")
        if set(proposal.arguments) != {"text"} or not isinstance(proposal.arguments["text"], str):
            raise ValueError("echo requires exactly one string argument: text")
        return Action(name=proposal.name, text=proposal.arguments["text"])


class CommandActionSpace(EchoActionSpace):
    def describe(self):
        return super().describe() + (ActionSpec('execute', 'Execute argv in a local workspace', {
            'type': 'object', 'properties': {'argv': {'type': 'array', 'minItems': 1,
                                                   'items': {'type': 'string', 'pattern': r'^[^\u0000]*$'}},
                                           'output_paths': {'type': 'array', 'items': {
                                               'type': 'string', 'minLength': 1,
                                               'description': 'Relative workspace path; absolute paths and .. components are rejected.'}}},
            'required': ['argv'], 'additionalProperties': False,
        }),)

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action:
        if proposal.name != 'execute':
            return super().validate(proposal, allowed_actions)
        args = proposal.arguments
        if proposal.name not in allowed_actions or not isinstance(args, dict):
            raise ValueError('execute is not allowed')
        if set(args) - {'argv', 'output_paths'} or 'argv' not in args:
            raise ValueError('execute requires argv and optional output_paths')
        argv, outputs = args['argv'], args.get('output_paths', [])
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and '\0' not in x for x in argv):
            raise ValueError('argv must be a nonempty list of strings')
        if not isinstance(outputs, list) or not all(isinstance(x, str) and x and not Path(x).is_absolute()
                                                   and '..' not in Path(x).parts for x in outputs):
            raise ValueError('output_paths must stay within the workspace')
        return Action('execute', arguments={'argv': list(argv), 'output_paths': list(outputs)})
