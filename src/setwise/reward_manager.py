"""Fail-fast parallel manager: unlike upstream, never converts judge errors to zero."""
from concurrent.futures import ThreadPoolExecutor
import os

import torch
from verl.workers.reward_manager.abstract import AbstractRewardManager


class StrictRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.key = reward_fn_key

    def __call__(self, data, return_dict=True):
        cached = self._extract_reward_from_rm_scores(data, return_dict)
        if cached is not None:
            return cached
        jobs = []
        for i in range(len(data)):
            item = data[i]
            prompt_len = item.batch["prompts"].shape[-1]
            length = int(item.batch["attention_mask"][prompt_len:].sum().item())
            if length < 1:
                raise ValueError("Empty rollout response")
            text = self.tokenizer.decode(item.batch["responses"][:length], skip_special_tokens=True)
            jobs.append((i, length, text, item.non_tensor_batch))

        def score(job):
            _, _, text, metadata = job
            return self.compute_score(data_source=metadata[self.key], solution_str=text,
                ground_truth=metadata["reward_model"]["ground_truth"], extra_info=metadata["extra_info"])

        with ThreadPoolExecutor(max_workers=int(os.environ.get("SETWISE_JUDGE_WORKERS", "4"))) as pool:
            results = list(pool.map(score, jobs))  # Exceptions propagate: never update on broken rewards.
        tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        keys = set().union(*(r.keys() for r in results))
        extra = {key: [] for key in keys}
        for (i, length, _, _), result in zip(jobs, results):
            tensor[i, length - 1] = result["score"]
            for key in keys:
                extra[key].append(result.get(key, ""))
        return {"reward_tensor": tensor, "reward_extra_info": extra} if return_dict else tensor
