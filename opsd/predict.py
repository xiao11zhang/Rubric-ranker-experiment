"""Student-only inference, for both the base checkpoint and distilled adapters."""
import argparse
import json
from pathlib import Path

from .common import encode_prompt, parse_output, read_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base model or a full checkpoint")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter", help="Optional saved LoRA checkpoint directory")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-prompt-tokens", type=int, default=12000)
    p.add_argument("--max-response-tokens", type=int, default=128)
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = read_rows(args.data)
    prompts = [encode_prompt(tokenizer, row, top_k=args.top_k) for row in rows]
    if max(map(len, prompts)) > args.max_prompt_tokens:
        raise ValueError("Prompt exceeds limit; no truncation performed")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream, torch.no_grad():
        for row, prompt in zip(rows, prompts):
            tokens = torch.tensor([prompt], device=args.device)
            output = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                max_new_tokens=args.max_response_tokens, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id)
            response = output[0, len(prompt):].tolist()
            text = tokenizer.decode(response, skip_special_tokens=True)
            record = {"id": row["id"], "candidate_hash": row["candidate_hash"], "output": text,
                      "ended_eos": bool(response and response[-1] == tokenizer.eos_token_id),
                      **parse_output(text, len(row["docs"]), args.top_k)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
