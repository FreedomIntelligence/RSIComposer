"""Preparation integrity tests with symbolic fixtures, never medical evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.medical import prepare_benchmarks as prep
from examples.medical.task import prepare, samples


def native(benchmark, index, *, question=None, ident=None):
    labels = 'ABCDEFGHIJ' if benchmark == 'medxpertqa' else 'ABCD'
    options = {k: f'symbol {k}' for k in labels}
    question = question or f'{benchmark} complete symbolic context {index}\nSecond context line.'
    if benchmark == 'medxpertqa':
        return dict(id=ident or f'Text-{index}', question=question + '\nAnswer Choices: ' +
                    ' '.join(f'({k}) {v}' for k, v in options.items()), options=options,
                    label='J', medical_task='symbol task', question_type='Reasoning')
    row = dict(question=question, options=options, answer_idx='D', answer=options['D'],
               meta_info='not prompt context', metamap_phrases='unused metadata')
    if ident:
        row['id'] = ident
    return row


class MedicalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.native = dict(
            medxpertqa_test=[native('medxpertqa', i) for i in range(12)],
            medqa_test=[native('medqa', i) for i in range(12)],
            medqa_train=[native('medqa', i + 100) for i in range(20)])
        self.specs = deepcopy(prep.SOURCES)
        self.addCleanup(patch.stopall)
        patch.object(prep, 'SOURCES', self.specs).start()
        self.refresh()

    def refresh(self):
        for name, rows in self.native.items():
            path = self.source / self.specs[name]['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            content = prep.jsonl_bytes(rows)
            path.write_bytes(content)
            self.specs[name].update(sha256=sha256(content).hexdigest(), rows=len(rows))

    def plan(self, **kwargs):
        return prep.build_plan(self.source, **dict(train_count=5, dev_count=3,
                               guardrail_count=2, **kwargs))

    def test_real_pins_and_source_splits_are_explicit(self):
        # Test immutable production declarations, not the patched fixture specs.
        code = Path(prep.__file__).read_text()
        self.assertIn('07ea977f7e326909b51c883b65f6ad7f399fc6db07408ef3630f4e3048ec71b2', code)
        self.assertEqual(self.specs['medqa_test']['repository'], 'GBaker/MedQA-USMLE-4-options')
        self.assertEqual(self.specs['medxpertqa_test']['split'], 'test')

    def test_conversion_preserves_context_and_removes_only_verified_choices(self):
        raw = native('medxpertqa', 8)
        raw['question'] = raw['question'].replace('Answer Choices: ', 'Answer Choices:  ')
        entry = prep.convert(raw, 'medxpertqa_test', 9)
        self.assertEqual(entry.row['question'], raw['question'].rsplit('\nAnswer Choices:', 1)[0])
        self.assertEqual(entry.row['gold'], 'J')
        self.assertEqual(list(entry.row['options']), list('ABCDEFGHIJ'))
        self.assertEqual(set(entry.row), {'id', 'question', 'options', 'gold', 'benchmark_id'})
        medqa = native('medqa', 1)
        result = prep.convert(medqa, 'medqa_train', 1).row
        self.assertEqual(result['gold'], 'D')
        self.assertEqual(result['question'], medqa['question'])
        self.assertEqual(set(result), set(entry.row))

    def test_ambiguous_or_invalid_labels_fail(self):
        raw = native('medqa', 1)
        for value in (0, 1, 'd', 'E', None):
            raw['answer_idx'] = value
            with self.assertRaises(ValueError):
                prep.convert(raw, 'medqa_train', 1)
        raw['answer_idx'] = 'D'
        raw['answer'] = 'not the selected option'
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            prep.convert(raw, 'medqa_train', 1)
        mx = native('medxpertqa', 1)
        mx['images'] = ['not-a-text-only-row']
        with self.assertRaises(ValueError):
            prep.convert(mx, 'medxpertqa_test', 1)

    def test_choices_conflict_quarantines_and_blocks_stem_from_training(self):
        raw = self.native['medxpertqa_test'][0]
        stem = raw['question'].rsplit('\nAnswer Choices:', 1)[0]
        raw['options']['J'] = 'conflicting source option'
        self.native['medqa_train'].append(native('medqa', 999, question=stem))
        self.refresh()
        plan = self.plan()
        self.assertEqual(plan['counts']['medxpertqa_test:choices_conflict'], 1)
        self.assertEqual(plan['counts']['medqa_train:excluded'], 1)
        self.assertEqual(plan['audit'][0]['reason'], 'choices_conflict')
        self.assertNotIn('Text-0', {e.source_id for pool in plan['private'].values() for e in pool})

    def test_entire_test_pools_are_protected_before_train_selection(self):
        # Deliberately duplicate an evaluation stem with another option set.
        # An independent source question/label conflict must not be guessed away.
        test = self.native['medqa_test'][0]
        self.native['medqa_train'].append(deepcopy(test))
        self.refresh()
        plan = self.plan()
        self.assertEqual(plan['counts']['medqa_train:duplicate'], 1)
        all_rows = [e.row for pool in (*plan['public'].values(), *plan['private'].values()) for e in pool]
        self.assertEqual(len(all_rows), len({r['id'] for r in all_rows}))
        self.assertEqual(len(all_rows), len({prep.canonical(r['question']) for r in all_rows}))

    def test_unicode_question_duplicate_and_source_id_conflicts(self):
        copy = deepcopy(self.native['medqa_test'][0])
        copy['question'] = copy['question'].upper().replace(' ', '  ')
        self.native['medqa_train'].append(copy)
        self.refresh()
        self.assertEqual(self.plan()['counts']['medqa_train:duplicate'], 1)
        self.native['medqa_test'][0]['id'] = 'shared-id'
        self.native['medqa_train'][0]['id'] = 'shared-id'
        self.refresh()
        plan = self.plan()
        self.assertEqual(plan['counts']['medqa_test:duplicate_conflict'], 1)
        self.assertEqual(plan['counts']['medqa_train:duplicate_conflict'], 1)
        for pool in (*plan['public'].values(), *plan['private'].values()):
            self.assertNotIn('medqa:shared-id', {e.row['id'] for e in pool})

    def test_duplicate_id_cannot_hide_changed_gold(self):
        duplicate = deepcopy(self.native['medqa_test'][0])
        duplicate.update(answer_idx='A', answer=duplicate['options']['A'])
        self.native['medqa_train'].append(duplicate)
        self.refresh()
        plan = self.plan()
        self.assertEqual(plan['counts']['medqa_test:duplicate_conflict'], 1)
        self.assertEqual(plan['counts']['medqa_train:duplicate_conflict'], 1)
        self.assertEqual(len(plan['audit']), 2)

    def test_cross_benchmark_stem_dedup_ignores_incomparable_label_spaces(self):
        raw = self.native['medxpertqa_test'][0]
        stem = raw['question'].rsplit('\nAnswer Choices:', 1)[0]
        self.native['medqa_test'].append(native('medqa', 998, question=stem))
        self.native['medqa_train'].append(native('medqa', 999, question=stem))
        self.refresh()
        plan = self.plan()
        self.assertEqual(plan['counts']['medqa_test:duplicate'], 1)
        self.assertEqual(plan['counts']['medqa_train:duplicate'], 1)
        self.assertEqual(plan['audit'], [])
        all_rows = [e.row for pool in (*plan['public'].values(), *plan['private'].values()) for e in pool]
        self.assertEqual(sum(prep.canonical(r['question']) == prep.canonical(stem) for r in all_rows), 1)

    def test_exclusions_apply_to_all_pools_and_are_private(self):
        entry = prep.convert(self.native['medxpertqa_test'][0], 'medxpertqa_test', 1)
        # Existing domain files need not have adopted this script's namespace.
        entry.row['id'] = 'Text-0'
        exclusion = self.root / 'operator-exclusion.jsonl'
        exclusion.write_bytes(prep.jsonl_bytes([entry.row]))
        plan = self.plan(exclusions=[exclusion])
        self.assertEqual(plan['counts']['medxpertqa_test:excluded'], 1)
        public, private = self.root / 'public', self.root / 'private'
        prep.write_plan(plan, public, private)
        visible = b''.join(p.read_bytes() for p in public.iterdir())
        self.assertNotIn(b'Text-0"', visible)
        self.assertNotIn(str(exclusion).encode(), visible)
        self.assertIn(str(exclusion), (private / 'hashmanifest.json').read_text())

    def test_selection_reproducible_train_prefix_and_fixed_evaluation(self):
        one = self.plan()
        two = prep.build_plan(self.source, train_count=8, dev_count=3, guardrail_count=2)
        self.assertEqual(one['public']['train.medqa.jsonl'], two['public']['train.medqa.jsonl'][:5])
        for name in ('dev.medxpertqa.jsonl', 'guardrail.medqa.jsonl'):
            self.assertEqual(one['public'][name], two['public'][name])
        self.assertEqual(one['private'], two['private'])
        self.assertEqual(one['public'], self.plan()['public'])

    def test_pinned_source_change_and_shortage_fail_without_output(self):
        path = self.source / self.specs['medqa_train']['path']
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.plan()
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            prep.build_plan(self.source, train_count=25, dev_count=3, guardrail_count=2)
        self.assertFalse((self.root / 'public').exists())

    def test_materialization_binds_frozen_task_and_withholds_private_rows(self):
        plan = self.plan()
        before = {p: p.read_bytes() for p in self.source.rglob('*.jsonl')}
        public, private = self.root / 'public', self.root / 'private'
        prep.write_plan(plan, public, private)
        task = prep.verify_public(public)
        self.assertEqual(len(task.train.read()), 5)
        self.assertEqual(len(task.check()), 5)
        self.assertEqual(task, prepare(json.loads((public / 'task.json').read_bytes()), base_directory=public))
        manifest = json.loads((public / 'hashmanifest.json').read_bytes())
        self.assertEqual(manifest['implementation_sha256'], sha256(Path(prep.__file__).read_bytes()).hexdigest())
        self.assertEqual(manifest['partitions']['dev.medxpertqa.jsonl']['kind'], 'custom_development')
        self.assertEqual(manifest['partitions']['dev.medxpertqa.jsonl']['source_split'], 'test')
        self.assertFalse(manifest['partitions']['dev.medxpertqa.jsonl']['official_development'])
        self.assertFalse(manifest['partitions']['dev.medxpertqa.jsonl']['new_holdout_claim'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.rglob('*.jsonl')})
        visible = b''.join(p.read_bytes() for p in public.iterdir())
        self.assertNotIn(str(private).encode(), visible)
        self.assertNotIn(str(self.source).encode(), visible)
        self.assertNotIn(b'heldout.', visible)
        self.assertNotIn(b'lineage', visible)
        for pool in plan['private'].values():
            for entry in pool:
                self.assertNotIn(entry.row['id'].encode() + b'"', visible)
        for path in private.glob('heldout.*.jsonl'):
            self.assertTrue(samples(path.read_bytes()))
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        with self.assertRaises(FileExistsError):
            prep.write_plan(plan, public, private)

    def test_output_digest_detects_tampering(self):
        public, private = self.root / 'public', self.root / 'private'
        prep.write_plan(self.plan(), public, private)
        data = public / 'train.medqa.jsonl'
        data.chmod(0o600)
        data.write_bytes(data.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            prep.verify_public(public)

    def test_source_and_private_directory_overlap_rejected(self):
        plan = self.plan()
        for public, private in ((self.source / 'out', self.root / 'private'),
                                (self.root / 'public', self.source / 'private'),
                                (self.root / 'public', self.root / 'public' / 'private'),
                                (self.root / 'same', self.root / 'same')):
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                prep.write_plan(plan, public, private)

    def test_relocation_does_not_silently_rebind_frozen_paths(self):
        public, private = self.root / 'public', self.root / 'private'
        prep.write_plan(self.plan(), public, private)
        moved = self.root / 'relocated'
        public.rename(moved)
        with self.assertRaisesRegex(ValueError, 'Frozen task identity'):
            prep.verify_public(moved)


if __name__ == '__main__':
    unittest.main()
