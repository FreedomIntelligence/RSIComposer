"""Bind evaluation and storage I/O around the unmodified paper optimizer."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

from .protocol import paper_modules


class PaperGEPA:
    def __init__(self, config, protocol):
        self.benchmark, self.component, _, _, self.dspy = protocol
        self.artifact_dir = config['artifact_dir']
        self.config = config

    def candidate(self, program):
        return {self.component: dict(program.named_predictors())[self.component].signature.instructions}

    def run(self, train, validation, exchange, observe):
        engine, utilities = paper_modules(self.artifact_dir)
        dspy = self.dspy
        bridge = self
        optimizer = None
        feedback_batch = {}
        evaluated_candidate = None

        def state():
            return optimizer.gepa_state

        def evaluate(program, batch, *, capture=False):
            nonlocal evaluated_candidate
            split = batch[0].split
            if any(row.split != split for row in batch):
                raise RuntimeError('GEPA evaluation batch mixes splits')
            current = state()
            iteration = current.i + 1 if current else 0
            trace = current.full_program_trace[-1] if current and current.full_program_trace else {}
            parents = (trace['merged_entities'][:2] if trace.get('merged') else
                       (trace['selected_program_candidate'],) if 'selected_program_candidate' in trace else ())
            candidate = bridge.candidate(program)
            if not capture:
                evaluated_candidate = candidate
            return exchange('evaluate', {'candidate': candidate, 'split': split,
                'indices': [row.index for row in batch], 'capture_traces': capture,
                'round': iteration, 'baseline': current is None},
                [bridge.candidate(current.program_candidates[i]) for i in parents])

        class Evaluate:
            def __init__(self, **options):
                self.__dict__.update(options)

            def __call__(self, program):
                # GEPA mutates its baseline scores in place to track the Pareto front.
                scores = list(evaluate(program, self.devset)['scores'])
                # The paper stores these output tuples for best-output logs. Full
                # responses are already stored in the corresponding RSI Lab trace.
                outputs = [(row, dspy.Prediction(), score) for row, score in zip(self.devset, scores, strict=True)]
                return round(100 * sum(scores) / len(scores), 2), outputs, scores

        class DSPyIO:
            def __getattr__(self, name):
                return getattr(dspy, name)

        def traces(program, dataset, **options):
            results = evaluate(program, dataset, capture=True)
            feedback_batch.clear()
            feedback_batch.update(usable_feedback=sum(record is not None for record in results['trajectories']),
                                  parent_correct=sum(results['scores']))
            predictor = dict(program.named_predictors())[bridge.component]
            # Rehydrate only parsed predictions. The artifact's default excludes
            # failed parses before scoring the reflection batch.
            for row, record, score in zip(dataset, results['trajectories'], results['scores'], strict=True):
                if record is None:
                    continue
                example = dspy.Example(**record['inputs'], feedback=record['feedback']).with_inputs('problem')
                prediction = dspy.Prediction(**record['generated_output'])
                yield {'example': example, 'prediction': prediction,
                       'trace': [(predictor, record['inputs'], prediction)], 'score': score}

        def feedback(*, module_inputs, **_):
            return {'feedback_score': None, 'feedback_text': module_inputs.feedback}

        def reflect(prompt, **_):
            text = exchange('reflect', {'prompt': prompt, 'round': state().i + 1}, ())['text']
            if not text.strip():
                raise _EmptyReflection('Reflection response was empty or truncated')
            return [text]

        def screen(status, child=None, *, pool_index=None):
            current = state()
            if not current or not current.full_program_trace:
                return
            trace = current.full_program_trace[-1]
            merged = trace.get('merged', False)
            fields = {'round': current.i + 1, 'status': status,
                      'strategy': 'merge' if merged else 'reflection',
                      'split': 'validation' if merged else 'search',
                      'batch_size': len(trace['subsample_ids'])}
            if merged:
                parents = trace['merged_entities'][:2]
                fields.update(parent_correct=sum(trace['id1_subsample_scores']),
                              second_parent_correct=sum(trace['id2_subsample_scores']),
                              child_correct=sum(trace['new_program_subsample_scores']))
            else:
                parents = (trace['selected_program_candidate'],)
                fields.update(feedback_batch)
                if 'new_subsample_scores' in trace:
                    fields['child_correct'] = sum(trace['new_subsample_scores'])
            if pool_index is not None:
                fields['pool_index'] = pool_index
                fields['validation_correct'] = sum(current.prog_candidate_val_subscores[pool_index])
                fields['validation_size'] = len(validation)
            observe(fields, [bridge.candidate(current.program_candidates[i]) for i in parents], child)

        def log(message):
            error = sys.exception()
            if 'Exception during ' in message and error is not None and not isinstance(error, _EmptyReflection):
                raise _UpstreamFailure(message) from error
            # These are branch markers from the pinned artifact. Read scores and
            # ancestry from optimizer state; never parse them out of log text.
            if not message.startswith('Iteration '):
                return
            _, _, event = message.partition(': ')
            if event.startswith('No valid predictions found for module '):
                screen('no_usable_feedback')
            elif event.startswith('All scores are perfect. Skipping module '):
                screen('perfect_feedback')
            elif event.startswith('Exception during instruction proposal:') and isinstance(error, _EmptyReflection):
                screen('empty_reflection')
            elif (event == 'New subsample score is not better, skipping'
                    or event.endswith(', skipping merge') and event.startswith('New program subsample score ')):
                screen('rejected', evaluated_candidate)

        # The optimizer and utility modules are private instances of the pinned
        # author's source. Replace transport and redundant pickle checkpoints only.
        engine.dspy = DSPyIO()
        engine.dspy.Evaluate = Evaluate
        engine.print = lambda *args, **kwargs: None
        utilities.bootstrap_trace_data = traces
        utilities.GEPAState.save = lambda self, run_dir: None
        engine.capture_module_trace_with_feedback = utilities.capture_module_trace_with_feedback
        engine.initialize_gepa_state = utilities.initialize_gepa_state
        p = self.config['parameters']

        with TemporaryDirectory(prefix='rsi-gepa-paper-') as scratch:
            optimizer = engine.GEPA(named_predictor_to_feedback_fn_map={self.component: feedback},
                knowledgebase_qe=None, metric=self.benchmark.metric, logger=SimpleNamespace(log=log),
                run_dir=scratch, run_linearized_gepa=False,
                seed=self.config['seed'], use_merge=p.get('use_merge', False),
                max_merge_invocations=p.get('max_merge_invocations', 5),
                track_scores_on='val', set_for_merge_minibatch='val',
                num_dspy_examples_per_gepa_step=p['minibatch_size'], max_metric_calls=p['max_metric_calls'])
            add_to_pool = optimizer.run_full_eval_add_new_program_to_gepa_tree

            def observe_pool(new_program, **kwargs):
                child = bridge.candidate(new_program)
                screen('validating', child)
                added = add_to_pool(new_program, **kwargs)
                screen('admitted', child, pool_index=added[0])
                return added

            optimizer.run_full_eval_add_new_program_to_gepa_tree = observe_pool
            base = self.benchmark.program_cot.deepcopy()
            def examples(batch):
                return [dspy.Example(split=split, index=index).with_inputs() for split, index in batch]

            try:
                with dspy.context(lm=reflect):
                    result = optimizer.gepa(base, examples(train), examples(validation))
            except _UpstreamFailure as error:
                raise RuntimeError(str(error)) from error.__cause__
        selected = utilities.idxmax(result.per_program_tracked_scores)
        return {'candidates': [self.candidate(program) for program in result.program_candidates],
                'parents': result.parent_program_for_candidate, 'trace': result.full_program_trace,
                'optimizer_metric_calls': result.total_num_evals,
                'selected_candidate': self.candidate(result.program_candidates[selected]),
                'validation_score': result.program_full_scores_val_set[selected] / 100}


class _EmptyReflection(Exception):
    pass


class _UpstreamFailure(BaseException):
    pass
