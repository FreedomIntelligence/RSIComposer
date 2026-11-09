"""Regression checks for protocol differences found in the reproduction audit."""
import json
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rsisys.api import Application, ApplicationConfig, Registry
from rsisys.labs.math_verifiers import StrictNumericVerifier
from rsisys.labs.seal import SEALLab, train_sequences
from rsisys.labs.seal_grader import SEALSemanticGrader
from rsisys.labs.seal_worker import inner_rows
from rsisys.labs.ttrl_verl import TTRLVerlLab, validate_config
from rsisys.labs.verl_worker import native_settings, unlabeled_records, tensor_hashes


class PaperAdapterTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'safetensors')),
                         'Optional tensor publication dependencies are not installed')
    def test_publication_hash_detects_numeric_updates_but_not_dtype_conversion(self):
        import torch
        from safetensors.torch import save_file
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('before', 'converted', 'changed'):
                (root/name).mkdir()
            value = torch.tensor([1., 2.], dtype=torch.bfloat16)
            save_file({'weight': value}, root/'before/model.safetensors')
            save_file({'weight': value.float()}, root/'converted/model.safetensors')
            save_file({'weight': value.float()+1}, root/'changed/model.safetensors')
            self.assertEqual(tensor_hashes(root/'before'), tensor_hashes(root/'converted'))
            self.assertNotEqual(tensor_hashes(root/'before'), tensor_hashes(root/'changed'))

    def test_newline_construction_handles_preamble_and_multiline_qa(self):
        self.assertEqual(train_sequences('Introduction\n1. first\n2. second', 'article', 'Title', split_newlines=True),
                         ['Title\n1. first', 'Title\n2. second', 'Title\narticle'])
        self.assertEqual(train_sequences('Question 1: why?\nAnswer: A\nQuestion 2: how?\nAnswer: B',
            'article', 'Title', split_newlines=True),
            ['Title\nQuestion 1: why?\nAnswer: A', 'Title\nQuestion 2: how?\nAnswer: B', 'Title\narticle'])

    def test_inner_tokenization_matches_upstream_padding_labels_without_added_eos(self):
        calls = []
        def tokenize(text, **kwargs):
            calls.append(kwargs)
            return {'input_ids': [8, 9, 0, 0], 'attention_mask': [1, 1, 0, 0]}
        rows = inner_rows(['fact'], tokenize, 4)
        self.assertEqual(calls, [dict(truncation=True, max_length=4, padding='max_length')])
        self.assertEqual(rows[0]['labels'], [8, 9, 0, 0])

    def test_seed_rewards_are_averaged_before_candidate_selection(self):
        with TemporaryDirectory() as tmp:
            engine = SimpleNamespace(name='fixture', simulated=True, generate=lambda: None,
                                     adapt=lambda: None, train=lambda: None)
            lab = SEALLab(SimpleNamespace(data_dir=tmp), {}, engine=engine)
            row = dict(title='T', context='C', completion='E', questions=[dict(question='Q', answer='yes')])
            episodes = lab._episodes([row], 3)
            self.assertEqual(len({r['seed'] for r in episodes}), 3)
            self.assertTrue(all(list(q)==['prompt'] for ep in episodes for q in ep['questions']))
            responses = [dict(weights_changed=True, steps=1, baseline=['no'], predictions=[p]) for p in ('yes', 'no', 'no')]
            scored = lab._score_adaptations([row], responses, 3)
            self.assertAlmostEqual(scored[0]['adapter_mean'], 1/3)
            self.assertEqual(len(scored[0]['inner_runs']), 3)
            self.assertAlmostEqual(scored[0]['scores'][0]['exact_match'], 1/3)
            with self.assertRaisesRegex(ValueError, 'seeds'):
                lab._score_adaptations([row], responses[:2], 3)

    def test_semantic_grade_uses_question_and_fails_on_ambiguous_service_result(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='yes' if len(calls)==1 else 'uncertain')
        grader = SEALSemanticGrader(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        question = dict(question='Who won?', answer='Broncos')
        self.assertEqual(grader.score_question('Denver Broncos', question, 'Football')['f1'], 1)
        self.assertIn('Topic: Football\nWho won?', calls[0]['input'])
        self.assertEqual(calls[0]['model'], 'gpt-4.1')
        grader.score_question('Denver Broncos', question, 'Football')
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            grader.score_question('unknown', question, 'Football')

    def test_paper_accuracy_uses_document_mean_and_retains_question_micro_metric(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'base').mkdir()
            docs = [dict(title=str(i), context='document-'+str(i),
                questions=[dict(question='q', answer='correct') for _ in range(n)])
                for i, n in enumerate((1, 3))]
            (root/'eval.json').write_text(json.dumps(docs))
            engine = SimpleNamespace(name='fixture', simulated=True, train=lambda: None,
                generate=lambda *a, **k: [dict(completion='edit') for _ in docs],
                adapt=lambda model, episodes, directory, **kwargs: [dict(weights_changed=True, steps=1,
                    baseline=['wrong']*len(ep['questions']),
                    predictions=['correct' if i==0 else 'wrong']*len(ep['questions']))
                    for i, ep in enumerate(episodes)])
            from rsisys.labs.seal import squad_scores
            lab = SEALLab(SimpleNamespace(data_dir=root/'run'), dict(profile='paper',
                gpu_ids=[0,1], data_root=tmp), engine=engine, scorer=squad_scores, scorer_name='fixture-only')
            result = lab._evaluate(dict(round=0, model='base', dataset='eval.json'))
            self.assertEqual(result['quality_metrics']['adapted_accuracy'], .5)
            self.assertEqual(result['quality_metrics']['question_micro_accuracy'], .25)

    def test_native_overrides_cannot_inject_labels_or_change_checkpoint_selection(self):
        for key in ('data', 'trainer', 'actor_rollout_ref', 'data.train_files', 'ttrl.enable', 'trainer.total_epochs', 'trainer.default_local_dir'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(dict(upstream_root='unused', native_overrides={key: 'bad'}))
        records = unlabeled_records([dict(id='q', prompt='2+2?')])
        self.assertEqual(records[0]['reward_model']['ground_truth'], 'None')
        settings = native_settings(dict(config=dict(total_epochs=10, train_batch_size=32,
            samples_per_prompt=32, max_length=4096, max_tokens=3072, temperature=.6, top_p=1.,
            eval_samples=16, evaluation_top_p=.95, evaluation_temperature=.6), model='base', votes=64),
            Path('/run'), Path('/upstream/verl'))
        self.assertEqual(settings['actor_rollout_ref.actor.loss_agg_mode'], 'token-mean')
        self.assertEqual(settings['actor_rollout_ref.actor.optim.warmup_style'], 'cosine')
        self.assertEqual(settings['ttrl.n_votes_per_prompt'], 64)
        self.assertEqual(settings['ttrl.n_samples_per_prompt'], 32)
        self.assertEqual(settings['data.seed'], 1)
        self.assertEqual(settings['actor_rollout_ref.rollout.seed'], 0)

    def test_registered_native_schedule_evaluates_then_fits_then_evaluates_final_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'base').mkdir()
            (root/'train.json').write_text(json.dumps([dict(id='q', prompt='2+2?')]))
            (root/'eval.json').write_text(json.dumps(dict(target=[dict(id='q', prompt='2+2?', answer=r'\boxed{4}')], transfer=[])))
            calls = []
            class Engine:
                name, simulated = 'native-fixture', True
                def generate(self, model, prompts, directory, **kwargs):
                    calls.append(('evaluate', model, len(prompts), kwargs))
                    return [dict(completion=r'\boxed{4}' if i%2 else r'\boxed{5}') for i, _ in enumerate(prompts)]
                def fit(self, model, rows, directory, checkpoint, **kwargs):
                    calls.append(('fit', model, rows, kwargs))
                    checkpoint.mkdir(parents=True)
                    (checkpoint/'model.safetensors').write_bytes(b'simulated')
                    return dict(steps=1, weights_changed=True)
            registry = Registry()
            registry.register_lab('native_fixture', lambda context, config: TTRLVerlLab(context, config,
                engine=Engine(), verifier=StrictNumericVerifier()), version='test')
            modules = registry.build(dict(controller=dict(name='ttrl_verl', config=dict(
                initial_model=str(root/'base'), train_data=[str(root/'train.json')], eval_data=str(root/'eval.json'),
                samples_per_prompt=4)), labs=dict(ttrl=dict(name='native_fixture', config=dict(
                    upstream_root='unused', train_batch_size=1, samples_per_prompt=2)))))
            with Application(ApplicationConfig('native', root/'run', {}, {}, max_decision_steps=4), modules=modules) as app:
                app.run()
                report = app.controller_report()
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['rounds_completed'], 1)
            self.assertFalse(report['training_performed'])
            self.assertEqual([c[0] for c in calls], ['evaluate', 'fit', 'evaluate'])
            self.assertEqual(calls[1][2], [dict(id='q', prompt='2+2?')])
            self.assertNotEqual(calls[0][1], calls[2][1])
            self.assertEqual(calls[2][2], 16)
            self.assertEqual(report['evaluations'][-1]['metrics']['target_accuracy'], .5)
            self.assertEqual(report['evaluations'][-1]['metrics']['target_questions'], 1)


if __name__ == '__main__':
    unittest.main()
