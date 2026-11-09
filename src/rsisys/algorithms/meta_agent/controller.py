"""Assemble a model-driven controller with an already-adapted harness."""


def validate_config(config):
    if set(config) - {'memory', 'harness', 'dsh'}:
        raise ValueError('Unknown meta_agent option')
    if type(config.get('memory', True)) is not bool:
        raise ValueError('meta_agent memory must be boolean')
    harness = config.get('harness', 'default')
    if harness not in ('default', 'dsh'):
        raise ValueError('meta_agent harness must be default or dsh')
    if harness == 'default' and 'dsh' in config:
        raise ValueError('dsh configuration requires harness=dsh')
    if harness == 'dsh':
        from rsisys.loops.decision_making.harness.dsh import DSHConfig
        if not isinstance(config.get('dsh'), dict):
            raise ValueError('harness=dsh requires an explicit dsh configuration object')
        DSHConfig(**config['dsh'])


def build_meta_agent(context, execution, config):
    validate_config(config)
    if config.get('harness', 'default') == 'dsh':
        from rsisys.loops.decision_making.harness.dsh import DSHConfig, dsh_decision_factory
        return dsh_decision_factory(DSHConfig(**config['dsh']))(context, execution)
    from rsisys.assembly import build_default_decision
    return build_default_decision(context, execution)
