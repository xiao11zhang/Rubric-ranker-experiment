"""Group 05 runner: original OPSD updates, approved deentity PI, fixed 500-step stop."""
import argparse
import json
from pathlib import Path
import random
import time

from opsd.common import encode_prompt, parse_output, read_rows
from opsd.objective import loss, sampled_log_probs
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from opsd.common import digest
from .prepare import runtime
from .protocol import verify
from .entities import teacher_row
from .reporting import actual_report, training_report
from opsd.full_run.utils import atomic_json, restore_rng, select_best, utc
from opsd.full_run.checkpoint import save_checkpoint, load_checkpoint, reconcile_log
from opsd.full_run.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path('opsd/group05_deentity/plans/deentity-001.json'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    c = plan['config']
    train, derived = runtime(plan)
    limit = plan['execution_stop_after_step']
    if not args.execute:
        print(json.dumps({'status': 'plan_verified', 'training_started': False, 'stop_after_step': limit}))
        return
    small = jsonl(plan['small_dev_path'])
    order = plan['order_indices']
    if len(order) != len(train) or set(order) != set(range(len(train))):
        raise ValueError('Not an exhaustive permutation')
    root = ROOT / c['output']
    if args.resume:
        if args.resume.resolve().parent != root.resolve():
            raise ValueError('Resume must belong to this run')
        if json.loads((root / 'plan.json').read_text(encoding='utf-8'))['config_sha256'] != plan['config_sha256']:
            raise ValueError('Resume belongs to another experiment')
        resume = load_checkpoint(args.resume, plan)
        if resume['step'] > limit:
            raise ValueError('Resume is beyond the frozen stopping boundary')
        reconcile_log(root, resume['step'])
    else:
        if root.exists():
            raise FileExistsError('Run exists; use explicit --resume for latest complete checkpoint')
        root.mkdir(parents=True)
        resume = None
        atomic_json(root / 'config.json', c)
        atomic_json(root / 'plan.json', plan)
    def status(stage, **fields):
        record = {'utc': utc(), 'stage': stage, **fields}
        atomic_json(root / 'status.json', record)
        print(json.dumps(record), flush=True)
    try:
        status('loading_models', resume_step=resume['step'] if resume else 0)
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from transformers import AutoModelForCausalLM, AutoTokenizer
        random.seed(c['seed'])
        torch.manual_seed(c['seed'])
        tokenizer = AutoTokenizer.from_pretrained(c['model'], revision=c['revision'], local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        kwargs = dict(revision=c['revision'], torch_dtype=getattr(torch, c['dtype']), attn_implementation='sdpa', local_files_only=True)
        student = AutoModelForCausalLM.from_pretrained(c['model'], **kwargs).to(c['student_device'])
        teacher = AutoModelForCausalLM.from_pretrained(c['model'], **kwargs).to(c['teacher_device'])
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        from peft import LoraConfig, get_peft_model, PeftModel
        if resume:
            student = PeftModel.from_pretrained(student, str(args.resume), is_trainable=True)
        else:
            student = get_peft_model(student, LoraConfig(r=c['lora_rank'], lora_alpha=c['lora_alpha'],
                lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                task_type='CAUSAL_LM'))
        student.enable_input_require_grads()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        student.train()
        params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=c['learning_rate'])
        if resume:
            optimizer.load_state_dict(resume['optimizer'])
            restore_rng(resume['rng'])
        writer = SummaryWriter(str(root / 'tensorboard'), purge_step=resume['step'] + 1 if resume else None)
        history_path = root / 'small-validation-history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if resume:
            history = [r for r in history if r['step'] <= resume['step']]
        def validate(step, checkpoint=None):
            model_key = sha(checkpoint / 'adapter_model.safetensors') if checkpoint else sha(Path(c['model']) / 'merge-manifest.json')
            status('small_validation', step=step, queries=200)
            context = student.disable_adapter() if step == 0 else __import__('contextlib').nullcontext()
            with context:
                result = evaluate(student, tokenizer, small, root / 'validation' / f'step-{step}', model_key=model_key)
            item = {'step': step, 'metrics': result['metrics'], 'model_key': model_key}
            history[:] = [r for r in history if r['step'] != step] + [item]
            history.sort(key=lambda r: r['step'])
            atomic_json(history_path, history)
            best = select_best(history)
            atomic_json(root / 'best-small-validation.json', best)
            for dataset in ('hotpot', 'musique', 'balanced_macro'):
                for key, value in result['metrics'][dataset].items():
                    if key != 'queries' and not key.endswith('_queries') and value is not None:
                        writer.add_scalar(f'validation/{dataset}/{key}', value, step)
            writer.flush()
            print(json.dumps({'validation_step': step, 'metrics': result['metrics']['balanced_macro'], 'best_step': best['step']}), flush=True)
        if not resume:
            validate(0)
        elif resume['step'] in plan['small_eval_steps']:
            validate(resume['step'], args.resume)
        start = time.monotonic()
        first = resume['step'] + 1 if resume else 1
        for step in range(first, limit + 1):
            optimizer.zero_grad(set_to_none=True)
            records = []
            # Only sampling changed: each training row is visited once, without replacement.
            begin = (step - 1) * c['gradient_accumulation']
            for index in order[begin:begin + c['gradient_accumulation']]:
                row = train[index]
                sp = encode_prompt(tokenizer, row, top_k=c['top_k'])
                tp = encode_prompt(tokenizer, teacher_row(row, derived), teacher=True, mode='correct', top_k=c['top_k'])
                s_input = torch.tensor([sp], device=c['student_device'])
                student.eval()
                with torch.no_grad():
                    generated = student.generate(input_ids=s_input, attention_mask=torch.ones_like(s_input),
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                        max_new_tokens=c['max_response_tokens'], pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id, use_cache=True)
                student.train()
                response = generated[0, len(sp):].tolist()
                if not response:
                    raise RuntimeError('Empty response')
                t_input = torch.tensor([tp + response], device=c['teacher_device'])
                with torch.no_grad():
                    tlogits = teacher(input_ids=t_input, use_cache=False).logits
                    tlp = sampled_log_probs(tlogits, t_input, len(tp)).to(c['student_device'])
                    del tlogits
                slogits = student(input_ids=generated, use_cache=False).logits
                slp = sampled_log_probs(slogits, generated, len(sp))
                objective, adv = loss(slp, tlp)
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite distillation loss')
                (objective / c['gradient_accumulation']).backward()
                text = tokenizer.decode(response, skip_special_tokens=True)
                records.append({'id': row['id'], 'teacher_rubric_hash': derived[row['id']]['rubric_hash'],
                                'original_rubric_hash': derived[row['id']]['original_rubric_hash'], 'output': text,
                                'valid': parse_output(text, len(row['docs']), c['top_k'])['valid'],
                                'tokens': len(response), 'ended_eos': response[-1] == tokenizer.eos_token_id,
                                'loss': objective.item(), 'mean_logp_gap': adv.mean().item()})
                del slogits, slp, tlp, objective, adv
            grad_norm = torch.nn.utils.clip_grad_norm_(params, c['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            entry = {'step': step, 'seconds': time.monotonic() - start, 'grad_norm': float(grad_norm), 'samples': records}
            with (root / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
            metrics = {'loss': sum(r['loss'] for r in records) / len(records),
                'teacher_student_logp_gap': sum(r['mean_logp_gap'] for r in records) / len(records),
                'valid_fraction': sum(r['valid'] for r in records) / len(records),
                'eos_fraction': sum(r['ended_eos'] for r in records) / len(records),
                'mean_response_tokens': sum(r['tokens'] for r in records) / len(records), 'grad_norm': float(grad_norm),
                'records_consumed': step * c['gradient_accumulation']}
            for key, value in metrics.items():
                writer.add_scalar('train/' + key, value, step)
            writer.flush()
            status('training', step=step, total_steps=limit, **metrics)
            if step in plan['checkpoint_steps']:
                checkpoint = save_checkpoint(student, tokenizer, optimizer, root, step, plan)
                if step in plan['small_eval_steps']:
                    validate(step, checkpoint)
        entries = jsonl(root / 'train.jsonl')
        actual = actual_report(plan, entries, derived)
        if ([r['step'] for r in history] != plan['small_eval_steps'] or
                not (root / f'step-{limit}/checkpoint.json').is_file()):
            raise ValueError('Checkpoint or scheduled dev200 evaluation incomplete')
        atomic_json(root / 'coverage.json', actual)
        training_report(root, plan, actual, history)
        writer.close()
        verify(plan)
        status('training_complete_at_frozen_500_step_boundary', step=limit,
               records_consumed=actual['records'], full_epoch_completed=False,
               best_step=select_best(history)['step'])
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
