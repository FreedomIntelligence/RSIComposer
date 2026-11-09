"""Optional real tensor gradients, runnable on CPU without model downloads."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires optional torch')
class WeightObjectiveTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('peft'), 'requires optional peft')
    def test_merge_check_rejects_updates_lost_to_bfloat16_rounding(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.layer import LoraLayer

        from rsisys.labs.weight_worker import merged_change_count

        base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).to(torch.bfloat16)
        with torch.no_grad():
            base[0].weight.fill_(1)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=['0']))
        self.assertEqual(merged_change_count(model), 0)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoraLayer):
                    module.lora_A['default'].weight.fill_(1)
                    module.lora_B['default'].weight.fill_(1e-10)
        self.assertEqual(merged_change_count(model), 0)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoraLayer):
                    module.lora_B['default'].weight.fill_(0.1)
        expected = merged_change_count(model)
        self.assertEqual(expected, 16)
        merged = model.merge_and_unload()
        self.assertEqual(int(torch.count_nonzero(merged[0].weight != 1)), expected)

    def test_advantage_sign_mask_and_clipped_gradient(self):
        import torch

        from rsisys.labs.weight_objectives import grpo_objective
        current = torch.tensor([[-1., -1., -1.]], requires_grad=True)
        old = current.detach().clone()
        mask = torch.tensor([[0., 1., 1.]])
        loss, ratio, kl = grpo_objective(current, old, old, 1., mask, 0.)
        loss.backward()
        self.assertEqual(current.grad[0, 0], 0)
        self.assertLess(current.grad[0, 1], 0)
        self.assertTrue(torch.equal(ratio, torch.ones_like(ratio)))
        current.grad.zero_()
        loss, _, _ = grpo_objective(current, old, old, -1., mask, 0.)
        loss.backward()
        self.assertGreater(current.grad[0, 1], 0)
        high = torch.tensor([[-0.1]], requires_grad=True)
        loss, _, _ = grpo_objective(high, torch.tensor([[-1.]]), torch.tensor([[-1.]]), 1., torch.ones(1, 1), 0.)
        loss.backward()
        self.assertEqual(high.grad.item(), 0)

    def test_reference_kl_minimized_at_reference(self):
        import torch

        from rsisys.labs.weight_objectives import grpo_objective
        ref = torch.tensor([[-2.]])
        for value in (-3., -2., -1.):
            current = torch.tensor([[value]], requires_grad=True)
            loss, _, kl = grpo_objective(current, current.detach(), ref, 0., torch.ones(1, 1), 1.)
            loss.backward()
            self.assertGreaterEqual(kl.item(), 0)
            if value == -2.:
                self.assertEqual(current.grad.item(), 0)
            else:
                self.assertGreater(current.grad.item() * (value + 2.), 0)

    def test_chunked_log_likelihood_matches_dense_and_prompt_mask(self):
        from types import SimpleNamespace

        import torch

        from rsisys.labs.weight_worker import tensor_batch, token_logps
        class Model:
            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(logits=torch.arange(input_ids.numel() * 7, dtype=torch.float32).reshape(*input_ids.shape, 7) / 7)
        rows = [{'ids': [1, 2, 3, 4], 'prompt_len': 2}, {'ids': [1, 3, 5], 'prompt_len': 1}]
        tokenizer = SimpleNamespace(pad_token_id=0)
        ids, attention, labels = tensor_batch(rows, tokenizer, 'cpu')
        model = Model()
        logits = model(ids, attention).logits[:, :-1] / 0.6
        expected = logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(token_logps(model, ids, attention, 0.6), expected, atol=1e-6))
        self.assertEqual(labels.tolist(), [[-100, -100, 3, 4], [-100, 3, 5, -100]])
