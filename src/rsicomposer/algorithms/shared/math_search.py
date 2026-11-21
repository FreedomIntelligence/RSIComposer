"""MathLab over the artifact-search protocol, for methods that search for a math artifact.

The data, verifiers, scoring and evidence are MathLab's: the Lab config names a
MathLab dataset manifest with development search splits and one sealed split,
and each split selects its verifier there. MathSearchLab adds the candidate
registry (register, evaluate, select), evaluation by integer offsets into a
split, and the per-example fields artifact-search Controllers read: scores,
feedback, trace_refs and receipt_id (the digest of the stored evaluation trace).

A method Lab implements _candidate_id() for its candidate shape and solve() for
how a candidate answers, and maps its own config to MathLab's in math_config().
PromptSearchLab is the case where the candidate is prompt text for the actor
(GEPA, OPRO); the harness Lab runs candidate programs against a model pool.
"""
from dataclasses import asdict
import json

from rsicomposer.api import MathLab, MetricSpec, OperationResult, PromptTemplateRuntime

from .experiment import ModelCalls
from .search_registry import SearchRegistry


class MathSearchLab(SearchRegistry, MathLab):
    # Prefix of the receipts stored for register, select and other bookkeeping operations.
    receipt_kind = 'search'

    def __init__(self, context, config):
        self._search_init()
        self.config = config
        MathLab.__init__(self, context, self.math_config(config))

    def math_config(self, config):
        """The MathLab config this Lab's config describes; split roles follow the search splits."""
        return {'manifest': config['manifest'],
                'split_roles': {**dict.fromkeys(self.search_splits, 'dev'), self.sealed_split: 'sealed'}}

    def state(self):
        return {'binding': self.binding(), 'search': self._search_state()}

    def restore(self, state):
        if state['binding'] != self.binding():
            raise ValueError('Math search Lab input binding changed since the checkpoint')
        self._restore_search(state['search'])

    def primary_metric(self):
        # The sealed split's verifier defines what the reported accuracy means.
        return MetricSpec('accuracy', 'maximize', self.datasets[self.sealed_split][0].verifier.name)

    def describe(self):
        return self._search_specs()

    def execute(self, call):
        request = call.arguments
        if call.name == 'evaluate':
            if request.get('split') not in self.datasets:
                raise ValueError('Unknown evaluation split')
            result = self.evaluate(call.call_id, {'split': request['split'], 'request': request,
                                                  'candidate': self._candidate(request)})
            self._record_baseline(request)
            return result
        if call.name == 'register':
            return self.receipt(call, self._register(request))
        if call.name == 'select':
            return self.receipt(call, self._select(request))
        raise ValueError(f'Unknown {self.receipt_kind} operation: {call.name}')

    def receipt(self, call, values):
        """Store a bookkeeping operation's request and result; its digest is the receipt_id."""
        payload = {'operation': call.name, 'request': call.arguments, 'result': values}
        ref = self.artifacts.put(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                                 kind=f'{self.receipt_kind}-{call.name}')
        return OperationResult(f'{self.receipt_kind} {call.name}', (ref,), False,
            data={'request': call.arguments, 'receipt_id': ref.digest, **values}, call_id=call.call_id)

    def offsets(self, request, size):
        """Example offsets an evaluation answers, in order; all of the split when indices is null."""
        return self._indices(request, size)

    def problems(self, arguments):
        rows = self.datasets[arguments['split']]
        return [rows[i] for i in self.offsets(arguments['request'], len(rows))]

    def feedback(self, record):
        return 'Correct answer' if record['correct'] else f"Incorrect answer; reference: {record['expected']}"

    def result_data(self, records, ref, arguments):
        return {'request': arguments['request'], 'receipt_id': ref.digest, 'valid': True,
                'scores': [int(row['correct']) for row in records],
                'trace_refs': [asdict(ref)] * len(records),
                'feedback': [self.feedback(row) for row in records]}


class PromptSearchLab(MathSearchLab):
    """Prompt domain agents answered by the actor role under a model-call budget.

    The Lab runs a PromptTemplateRuntime, so the template name, system text, temperature, top_p,
    seed, token limit and call budget that decide a score are all recorded in each benchmark's
    protocol and are the values that execute. A method supplies render(), the render function
    that reproduces its pinned upstream prompt exactly.

    Config: {"manifest": ..., "seed": 17, "max_model_calls": 256,
             "model": {"max_output_tokens": 2048, "scorer_temperature": 0.0}}
    """
    source_budgets = False  # Generation is set by this Lab, not by a benchmark.
    template_name = None    # Names the method's upstream prompt format, e.g. 'gepa-instruction/v1'.
    system = ''

    def __init__(self, context, config):
        self.calls = ModelCalls(context, config['max_model_calls'])
        super().__init__(context, config)

    def make_runtime(self, _math_config):
        # Built from this Lab's own config (MathLab only passes its derived one).
        config = self.config
        return PromptTemplateRuntime(
            template=self.template_name, system=self.system_text(config), temperature=config['model']['scorer_temperature'],
            top_p=1.0, seed=self.seed_policy(config), max_output_tokens=config['model']['max_output_tokens'],
            max_model_calls=config['max_model_calls'], render=self.render)

    def system_text(self, config):
        return self.system

    def seed_policy(self, config):
        return config['seed']

    def render(self, problem, instruction, request):
        """User prompt for one problem under one domain agent, matching the pinned upstream prompt."""
        raise NotImplementedError

    def solve(self, problems, arguments):
        return self.run_runtime(self.domain_agent(arguments), problems, arguments['request'])

    def domain_agent(self, arguments):
        """The submitted domain agent as the runtime accepts it."""
        return {'instruction': arguments['candidate']['instruction']}

    def math_config(self, config):
        return {**super().math_config(config), 'max_output_tokens': config['model']['max_output_tokens']}

    def state(self):
        return {**super().state(), 'model_calls': self.calls.count}

    def restore(self, state):
        super().restore(state)
        self.calls.count = state['model_calls']

    def ask_actor(self, problem, text, **options):
        # Every actor call spends the Lab's model-call budget.
        return super().ask_actor(problem, text, generate=self.calls.generate, **options)
