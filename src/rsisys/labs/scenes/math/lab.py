"""Reserved entry point for the reusable math task environment."""


class MathLab:
    """An assembly placeholder with no executable operations yet."""

    def __init__(self, context, config):
        self.validate_config(config)

    @staticmethod
    def validate_config(config):
        if config:
            raise ValueError('The math Lab placeholder accepts no configuration')

    def describe(self):
        return ()

    def execute(self, call):
        raise NotImplementedError('The math Lab is a placeholder and has no operations yet')
