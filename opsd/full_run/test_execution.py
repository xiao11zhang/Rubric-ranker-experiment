import ast
import json
from pathlib import Path
import random
import tempfile
import types
import unittest

from opsd.dev_eval.prepare import ROOT
from .utils import order_for_epoch, rng_state, restore_rng, select_best


class ExecutionTest(unittest.TestCase):
    def test_exhaustive_order_does_not_consume_global_rng(self):
        before = random.getstate()
        order = order_for_epoch(4000, 42)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(order, order_for_epoch(4000, 42))
        self.assertEqual(sorted(order), list(range(4000)))
        self.assertEqual(len({i for start in range(0, 4000, 4) for i in order[start:start + 4]}), 4000)

    def test_original_per_response_update_ast_is_identical(self):
        old = ast.parse((ROOT / 'opsd/train.py').read_text())
        new = ast.parse((ROOT / 'opsd/full_run/train.py').read_text())
        def micro_body(tree):
            for node in ast.walk(tree):
                if isinstance(node, ast.For) and len(node.body) > 1:
                    second = node.body[1]
                    if isinstance(second, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sp' for t in second.targets):
                        return ast.dump(ast.Module(body=node.body[1:], type_ignores=[]), include_attributes=False)
            raise AssertionError('Response update not found')
        self.assertEqual(micro_body(old), micro_body(new))
        def calls(tree, suffix):
            return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(suffix)]
        for suffix in ('clip_grad_norm_', 'optim.AdamW', 'LoraConfig'):
            self.assertEqual(calls(old, suffix), calls(new, suffix))

    def test_best_rule_includes_sft_and_breaks_ties(self):
        def item(step, primary, f1):
            return {'step': step, 'metrics': {'balanced_macro': {'all_annotated_supports_at_5': primary, 'support_f1_selected': f1}}}
        self.assertEqual(select_best([item(0, .8, .9), item(100, .7, .95)])['step'], 0)
        self.assertEqual(select_best([item(100, .8, .9), item(200, .8, .91)])['step'], 200)
        self.assertEqual(select_best([item(100, .8, .9), item(200, .8, .9)])['step'], 100)

    def test_evaluation_restores_rng_and_training_mode(self):
        import torch
        from .evaluation import evaluate
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = torch.device('cpu')
                self.config = types.SimpleNamespace(max_position_embeddings=1024)
            def generate(self, input_ids, **kwargs):
                torch.rand(20)
                random.random()
                return torch.cat([input_ids, torch.tensor([[4, 0]] * len(input_ids))], dim=1)
        class ToyTokenizer:
            pad_token_id = 0
            eos_token_id = 0
            def apply_chat_template(self, *args, **kwargs):
                return [2, 3]
            def decode(self, *args, **kwargs):
                return '[1]'
        row = {'id': 'toy', 'dataset': 'hotpot', 'query': 'Which?', 'docs': [{'content': 'Answer'}],
               'gold_support_ids': [1], 'answer_aliases': ['Answer'], 'candidate_hash': 'toy'}
        model = ToyModel().train()
        before = rng_state()
        with tempfile.TemporaryDirectory() as folder:
            evaluate(model, ToyTokenizer(), [row], Path(folder) / 'evaluation', model_key='toy')
        after = rng_state()
        self.assertTrue(model.training)
        self.assertEqual(before['python'], after['python'])
        self.assertTrue(torch.equal(before['torch_cpu'], after['torch_cpu']))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before['torch_cuda'], after['torch_cuda'])))

    def test_optimizer_rng_resume_is_exact_on_cpu(self):
        import torch
        from .checkpoint import save_checkpoint, load_checkpoint
        class Serializable(torch.nn.Linear):
            def save_pretrained(self, path):
                torch.save(self.state_dict(), Path(path) / 'adapter_model.safetensors')
        class Tokenizer:
            def save_pretrained(self, path):
                (Path(path) / 'tokenizer.json').write_text('{}')
        def update(model, optimizer):
            optimizer.zero_grad()
            x = torch.randn(2, 3) * random.random()
            model(x).square().mean().backward()
            optimizer.step()
        model = Serializable(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        update(model, optimizer)
        plan = {'config': {'gradient_accumulation': 4}, 'config_sha256': 'toy', 'order_ids': list(range(8))}
        with tempfile.TemporaryDirectory() as folder:
            path = save_checkpoint(model, Tokenizer(), optimizer, folder, 1, plan)
            update(model, optimizer)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            loaded = load_checkpoint(path, plan)
            other = Serializable(3, 1)
            other.load_state_dict(torch.load(path / 'adapter_model.safetensors', weights_only=True))
            opt = torch.optim.AdamW(other.parameters(), lr=.01)
            opt.load_state_dict(loaded['optimizer'])
            restore_rng(loaded['rng'])
            update(other, opt)
            self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in expected.items()))


if __name__ == '__main__':
    unittest.main()
