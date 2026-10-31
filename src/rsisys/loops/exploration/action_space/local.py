from pathlib import Path

from . import Action, ActionProposal, ActionSpec, EchoActionSpace


class CommandActionSpace(EchoActionSpace):
    def describe(self):
        return super().describe() + (ActionSpec('execute', 'Execute argv in a local workspace', {
            'type': 'object', 'properties': {'argv': {'type': 'array', 'items': {'type': 'string'}},
                                           'output_paths': {'type': 'array', 'items': {'type': 'string'}}},
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
