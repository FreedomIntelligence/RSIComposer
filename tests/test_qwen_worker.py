"""Worker device and lifecycle checks without torch, models, sockets or GPUs."""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.qwen_toy import worker


class Tensor:
    """Only the tiny tensor operations used by the worker's HTTP contract."""
    def __init__(self, values, moves):
        self.values, self.moves = values, moves
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def to(self, device):
        self.moves.append(device)
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, columns = index
            value = ([row[columns] for row in self.values[rows]] if isinstance(rows, slice)
                     else self.values[rows][columns])
        else:
            value = self.values[index]
        return Tensor(value, self.moves) if isinstance(value, list) else value

    def __ne__(self, value):
        return Tensor([[item != value for item in row] for row in self.values], self.moves)

    def sum(self):
        return sum(sum(row) for row in self.values)


class Batch(dict):
    @property
    def input_ids(self):
        return self['input_ids']

    def to(self, device):
        for value in self.values():
            value.to(device)
        return self


class ForbiddenCUDA:
    def __getattribute__(self, name):
        raise AssertionError(f'CPU worker touched CUDA: {name}')


class WorkerFixture:
    def __init__(self, *, cuda=False, devices=1):
        self.moves, self.pending = [], []
        self.model = Mock(config=SimpleNamespace(use_cache=False),
                          generation_config=SimpleNamespace(eos_token_id=[3]))
        self.model.to.return_value = self.model
        self.model.disable_adapter.side_effect = nullcontext
        self.model.return_value = SimpleNamespace(loss=0.5)
        self.model.generate.side_effect = lambda **kw: Tensor([[1, 2, 7, 3]], self.moves)
        self.tokenizer = Mock(eos_token_id=3, eos_token='<eos>', pad_token_id=0)
        self.tokenizer.side_effect = lambda text, **kw: (
            Batch(input_ids=Tensor([[1, 2]], self.moves)) if kw.get('return_tensors') else {'input_ids': [1, 2]})
        self.tokenizer.apply_chat_template.return_value = 'prompt'
        self.tokenizer.decode.return_value = '{"parity":"odd"}'
        self.trainer = Mock(state=SimpleNamespace(global_step=24))
        self.trainer.train.return_value = SimpleNamespace(metrics={'train_loss': 0.5})
        self.torch = SimpleNamespace(
            __version__='fixture', version=SimpleNamespace(cuda='fixture'),
            cuda=Mock(is_available=Mock(return_value=True), device_count=Mock(return_value=devices),
                      get_device_name=Mock(return_value='fixture')) if cuda else ForbiddenCUDA(),
            float32='float32', bfloat16='bfloat16', set_num_threads=Mock(),
            random=SimpleNamespace(default_generator=Mock()),
            manual_seed=Mock() if cuda else Mock(side_effect=AssertionError('CPU used all-device seed')),
            tensor=lambda values: Tensor(values, self.moves), no_grad=nullcontext,
        )
        self.transformers = SimpleNamespace(
            AutoTokenizer=Mock(from_pretrained=Mock(return_value=self.tokenizer)),
            Qwen3_5ForConditionalGeneration=Mock(from_pretrained=Mock(return_value=self.model)),
            Trainer=Mock(return_value=self.trainer),
            TrainingArguments=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)), TrainerCallback=object,
        )
        self.peft = SimpleNamespace(LoraConfig=Mock(), get_peft_model=Mock(return_value=self.model))
        self.smi = Mock(return_value='0, GPU-other\n2, GPU-allocated\n') if cuda else Mock(
            side_effect=AssertionError('CPU invoked nvidia-smi'))

    def post(self, path, payload):
        handler = self.handler.__new__(self.handler)
        data = json.dumps(payload).encode()
        handler.path, handler.headers = path, {'Content-Length': str(len(data))}
        handler.rfile = io.BytesIO(data)
        response = []
        handler.reply = lambda code, value: response.append((code, value))
        handler.do_POST()
        return response[-1]

    def get(self, path):
        handler = self.handler.__new__(self.handler)
        handler.path = path
        response = []
        handler.reply = lambda code, value: response.append(value)
        handler.do_GET()
        return response[-1]

    def job(self, ident):
        return self.get('/jobs/' + ident)

    def drain(self):
        target, args = self.pending.pop(0)
        target(*args)  # Fake Trainer only, synchronously outside the submission lock.

    def run(self, scenario, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            self.root = Path(directory)
            for name in ('train.jsonl', 'eval.jsonl'):
                (self.root / name).write_text(json.dumps({'prompt': 'p', 'answer': 'a'}) + '\n')
            def serve(address, handler):
                self.address = address
                self.handler = handler
                server = MagicMock()
                server.__enter__.return_value = server
                server.serve_forever.side_effect = scenario
                return server
            def thread(*, target, args, daemon):
                return SimpleNamespace(start=lambda: self.pending.append((target, args)))
            with patch.dict(sys.modules, torch=self.torch, transformers=self.transformers, peft=self.peft), \
                    patch.dict(os.environ, NVIDIA_VISIBLE_DEVICES='GPU-allocated', CUDA_VISIBLE_DEVICES='GPU-allocated'), \
                    patch.object(worker.subprocess, 'check_output', self.smi), \
                    patch.object(worker.threading, 'Thread', side_effect=thread), \
                    patch.object(worker, 'ThreadingHTTPServer', side_effect=serve), \
                    patch.object(worker.traceback, 'print_exc'), patch('sys.stdout', new_callable=io.StringIO) as output:
                worker.main(['--model', '/local/Qwen3.5-small', '--run-dir', directory, *arguments])
                self.log = output.getvalue()
                self.ready = json.loads((self.root / 'worker-ready.json').read_text())


class QwenWorkerTests(unittest.TestCase):
    def test_default_cpu_train_evaluate_and_serve_never_touch_cuda(self):
        fixture = WorkerFixture()
        def scenario():
            self.assertEqual(fixture.address, ('127.0.0.1', 8766))
            health = fixture.get('/health')
            self.assertEqual((health['device'], health['threads']), ('cpu', 4))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['ACCELERATE_USE_CPU'], 'true')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '4')
            fixture.torch.set_num_threads.assert_called_once_with(4)
            fixture.torch.random.default_generator.manual_seed.assert_called_once_with(73)
            code, receipt = fixture.post('/train', {'idempotency_key': 'train'})
            self.assertEqual(code, 200)
            self.assertEqual(fixture.post('/train', {'idempotency_key': 'train'}), (code, receipt))
            self.assertEqual(len(fixture.pending), 1)
            fixture.drain()
            self.assertEqual(fixture.job(receipt['job_id'])['status'], 'succeeded')
            options = fixture.transformers.TrainingArguments.call_args.kwargs
            self.assertEqual(options['max_steps'], 24)
            self.assertTrue(options['use_cpu'])
            self.assertFalse(options['bf16'])
            fixture.model.gradient_checkpointing_disable.assert_called_once()
            self.assertTrue(fixture.model.config.use_cache)
            self.assertEqual(fixture.post('/train', {'idempotency_key': 'another'})[0], 400)
            code, receipt = fixture.post('/evaluate', {'idempotency_key': 'eval'})
            self.assertEqual(code, 200)
            fixture.drain()
            self.assertEqual(fixture.job(receipt['job_id'])['status'], 'succeeded')
            code, response = fixture.post('/v1/chat/completions', {
                'model': 'Qwen3.5-9B', 'messages': [{'role': 'user', 'content': 'p'}], 'max_tokens': 2})
            self.assertEqual(code, 200)
            self.assertEqual(response['choices'][0]['finish_reason'], 'stop')
            self.assertTrue(fixture.moves)
            self.assertEqual(set(fixture.moves), {'cpu'})
        fixture.run(scenario)
        self.assertEqual(fixture.ready['host'], '127.0.0.1')
        self.assertEqual(json.loads(fixture.log), {'event': 'worker_ready', **fixture.ready})
        fixture.smi.assert_not_called()
        fixture.torch.manual_seed.assert_not_called()
        fixture.model.to.assert_called_once_with('cpu')
        self.assertEqual(fixture.transformers.Qwen3_5ForConditionalGeneration.from_pretrained.call_args.kwargs['dtype'], 'float32')
        for loader in (fixture.transformers.AutoTokenizer, fixture.transformers.Qwen3_5ForConditionalGeneration):
            self.assertTrue(loader.from_pretrained.call_args.kwargs['local_files_only'])

    def test_steps_and_threads_are_configurable_on_cpu(self):
        fixture = WorkerFixture()
        fixture.trainer.state.global_step = 3
        def scenario():
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '2')
            self.assertEqual(fixture.get('/health')['threads'], 2)
            _, receipt = fixture.post('/train', {'idempotency_key': 'train'})
            fixture.drain()
            self.assertEqual(fixture.job(receipt['job_id'])['status'], 'succeeded')
        fixture.run(scenario, '--device', 'cpu', '--steps', '3', '--threads', '2')
        self.assertEqual(json.loads(fixture.log)['threads'], 2)
        self.assertEqual(fixture.transformers.TrainingArguments.call_args.kwargs['max_steps'], 3)
        fixture.torch.set_num_threads.assert_called_once_with(2)

    def test_cuda_keeps_allocation_mapping_and_bfloat16(self):
        fixture = WorkerFixture(cuda=True)
        def scenario():
            self.assertEqual(fixture.address, ('0.0.0.0', 8766))
            self.assertEqual(fixture.get('/health')['device'], 'cuda')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(os.environ['ACCELERATE_USE_CPU'], 'false')
            _, receipt = fixture.post('/train', {'idempotency_key': 'train'})
            fixture.drain()
            self.assertEqual(fixture.job(receipt['job_id'])['status'], 'succeeded')
        fixture.run(scenario, '--device', 'cuda')
        self.assertEqual(json.loads(fixture.log.splitlines()[-1])['device'], 'cuda')
        fixture.smi.assert_called_once()
        fixture.model.to.assert_called_once_with('cuda')
        options = fixture.transformers.TrainingArguments.call_args.kwargs
        self.assertFalse(options['use_cpu'])
        self.assertTrue(options['bf16'])
        self.assertEqual(fixture.transformers.Qwen3_5ForConditionalGeneration.from_pretrained.call_args.kwargs['dtype'], 'bfloat16')

    def test_explicit_host_overrides_device_default(self):
        for device, host in (('cpu', '0.0.0.0'), ('cuda', '127.0.0.1')):
            fixture = WorkerFixture(cuda=device == 'cuda')
            with self.subTest(device=device):
                fixture.run(lambda: self.assertEqual(fixture.address, (host, 9876)),
                            '--device', device, '--host', host, '--port', '9876')
                self.assertEqual(fixture.ready['host'], host)

    def test_cuda_rejects_non_single_gpu_before_loading_model(self):
        for count in (0, 2):
            fixture = WorkerFixture(cuda=True, devices=count)
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, 'exactly one'):
                fixture.run(lambda: self.fail('Server must not start'), '--device', 'cuda')
            fixture.transformers.Qwen3_5ForConditionalGeneration.from_pretrained.assert_not_called()

    def test_cpu_cancellation_and_incomplete_training_never_publish_success(self):
        for phase in ('queued', 'training', 'saving', 'incomplete'):
            fixture = WorkerFixture()
            def scenario():
                _, receipt = fixture.post('/train', {'idempotency_key': 'train'})
                ident = receipt['job_id']
                def cancel(*unused):
                    fixture.post('/cancel', {'job_id': ident})
                if phase == 'queued':
                    cancel()
                elif phase == 'training':
                    def train():
                        cancel()
                        callback = fixture.transformers.Trainer.call_args.kwargs['callbacks'][0]
                        control = SimpleNamespace(should_training_stop=False, should_epoch_stop=False)
                        self.assertTrue(callback.on_substep_end(None, None, control).should_training_stop)
                        return SimpleNamespace(metrics={})
                    fixture.trainer.train.side_effect = train
                elif phase == 'saving':
                    fixture.model.save_pretrained.side_effect = cancel
                else:
                    fixture.trainer.state.global_step = 1
                fixture.drain()
                outcome = fixture.job(ident)
                self.assertEqual(outcome['status'], 'failed' if phase == 'incomplete' else 'cancelled')
                self.assertNotIn('result', outcome)
                self.assertFalse((fixture.root / 'train-result.json').exists())
                if phase == 'queued':
                    fixture.trainer.train.assert_not_called()
                else:
                    fixture.model.gradient_checkpointing_disable.assert_called_once()
            with self.subTest(phase=phase):
                fixture.run(scenario)

    def test_invalid_cli_options_fail_before_model_or_server_initialization(self):
        for arguments in (('--steps', '0'), ('--steps', '-1'), ('--threads', '0'),
                          ('--threads', '-1'), ('--device', 'gpu'), ('--host', '  ')):
            fixture = WorkerFixture()
            with self.subTest(arguments=arguments), patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                fixture.run(lambda: self.fail('Server must not start'), *arguments)
            fixture.transformers.Qwen3_5ForConditionalGeneration.from_pretrained.assert_not_called()
            fixture.smi.assert_not_called()


if __name__ == '__main__':
    unittest.main()
