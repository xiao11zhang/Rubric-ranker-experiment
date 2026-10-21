"""Configuration bridge to ONE pinned AdaTutoRank/ATO verl checkout."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from setwise.config import fingerprint, validate


def overrides(config, ato_root):
    c, root = config, Path(ato_root).resolve()
    train, model, rt, ato = c["training"], c["model"], c["runtime"], c["ato"]
    method = c["experiment"]["method"]
    if method == "b0":
        raise ValueError("B0 is the shared SFT checkpoint: use evaluate, no RL run")
    plugin = Path(__file__).parent.resolve()
    values = {
        "data.train_files": str(Path(c["data"]["train_parquet"]).resolve()),
        "data.val_files": str(Path(c["data"]["val_parquet"]).resolve()),
        "data.train_batch_size": train["query_batch_size"],
        "data.max_prompt_length": c["data"]["max_prompt_tokens"],
        "data.max_response_length": c["data"]["max_response_tokens"],
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "+data.apply_chat_template_kwargs": {"enable_thinking": model["enable_thinking"]},
        "actor_rollout_ref.model.path": str(Path(model["sft_checkpoint"]).resolve()),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": train["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": train["warmup_steps"],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "cosine",
        "actor_rollout_ref.actor.ppo_mini_batch_size": train["mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": rt["micro_batch_size"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_kl_loss": train["kl_coefficient"] > 0,
        "actor_rollout_ref.actor.kl_loss_coef": train["kl_coefficient"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.clip_ratio": train["clip_ratio"],
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "sdpo" if method == "b2" else "vanilla",
        "actor_rollout_ref.actor.fsdp_config.param_offload": rt["param_offload"],
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": rt["optimizer_offload"],
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": rt["micro_batch_size"],
        "actor_rollout_ref.rollout.name": "sglang",
        "actor_rollout_ref.rollout.n": train["group_size"],
        "actor_rollout_ref.rollout.temperature": train["temperature"],
        "actor_rollout_ref.rollout.top_p": train["top_p"],
        "actor_rollout_ref.rollout.top_k": train["top_k"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": rt["tensor_parallel_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": rt["rollout_memory_fraction"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": rt["micro_batch_size"],
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "custom_reward_function.path": str(plugin / "rewards.py"),
        "custom_reward_function.name": "compute_score",
        "+val_reward_function.path": str(plugin / "rewards.py"),
        "+val_reward_function.name": "compute_score",
        "reward_model.use_reward_loop": False,
        "reward_manager.source": "importlib",
        "reward_manager.name": "StrictRewardManager",
        "reward_manager.module.path": str(plugin / "reward_manager.py"),
        "trainer.nnodes": rt["nodes"],
        "trainer.n_gpus_per_node": rt["gpus_per_node"],
        "trainer.use_legacy_worker_impl": "enable",
        "trainer.total_epochs": train["epochs"],
        "trainer.total_training_steps": train["max_steps"],
        "trainer.val_before_train": True,
        "trainer.test_freq": train["eval_every"],
        "trainer.save_freq": train["save_every"],
        "data.seed": c["experiment"]["seed"],
        "trainer.project_name": "setwise-controlled",
        "trainer.experiment_name": c["experiment"]["name"],
        "trainer.default_local_dir": str(Path(rt["output_dir"]).resolve() / "checkpoints"),
        "trainer.rollout_data_dir": str(Path(rt["output_dir"]).resolve() / "rollouts"),
        "+trainer.validation_data_dir": str(Path(rt["output_dir"]).resolve() / "validation"),
        "trainer.distill_data_dir": str(Path(rt["output_dir"]).resolve() / "tutoring"),
        "trainer.logger": ["console", "tensorboard"],
        "max_model_len": ato["max_teacher_prompt_tokens"] + c["data"]["max_response_tokens"],
    }
    if method == "b2":
        prefix = "actor_rollout_ref.actor.self_distillation."
        settings = {"full_logit_distillation": False, "joint_grpo_distill": True,
            "opsd_combine_mode": "advantage", "opsd_combine_form": "convex",
            "opsd_convex_lambda": ato["grpo_weight"], "alpha": 1.0,
            "teacher_regularization": "ema", "teacher_update_rate": 0.0,
            # Upstream generic routing is >=threshold; move only the lower cut
            # to the next float so exactly .3 remains in the low bucket, on ALL Ray workers.
            "joint_exclude_incorrect_format": True, "hint_thresholds": [math.nextafter(0.3, math.inf), 0.6],
            "hint_bucket_types": ["demonstration", "diff", "rubric"],
            "rubric_set_bar_low": 0.3, "rubric_set_bar_medium": 0.6,
            "max_reprompt_len": ato["max_teacher_prompt_tokens"],
            "include_environment_feedback": False, "use_rubric_set_hint": True,
            "rubric_set_source": "policy", "rubric_set_max_response_length": c["data"]["max_response_tokens"],
            "diff_max_response_length": ato["reflection_tokens"],
            "rubric_set_gate_relative": True, "rubric_set_gate_margin": 0.0,
            "rubric_template_path": str(plugin / "prompts/adaptive_hints.yaml"),
            "rubric_set_select_template_path": str(plugin / "prompts/self_selector.yaml"),
            "diff_prompt_template_path": str(plugin / "prompts/self_reflector.yaml")}
        values.update({prefix + k: v for k, v in settings.items()})
    # Hydra accepts JSON-compatible literals, including quoted strings and lists.
    return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]


def command(config, ato_root):
    return [sys.executable, "-m", "setwise.training_entry", "--config-name", "ato", *overrides(config, ato_root)]


def checkpoint_hash(directory):
    root = Path(directory)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty SFT checkpoint")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def launch(config, config_path, ato_root, execute=False):
    validate(config, production=execute)
    cmd = command(config, ato_root)
    if not execute:
        return {"status": "plan_only", "argv": cmd, "cwd": str(Path(ato_root).resolve()),
                "config_sha256": fingerprint(config)}
    if sys.platform != "linux":
        raise RuntimeError("GPU training is supported on Linux only")
    root = Path(ato_root).resolve()
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual_commit != config["runtime"]["backend_commit"]:
        raise ValueError("Backend commit mismatch; pin the audited checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Backend checkout must be clean")
    if checkpoint_hash(config["model"]["sft_checkpoint"]) != config["model"]["sft_sha256"]:
        raise ValueError("SFT checkpoint hash mismatch")
    manifest_path = Path(config["data"]["manifest_path"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config["data"]["manifest_sha256"]:
        raise ValueError("Dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    query_ids = []
    for split in ("train", "val"):
        path = Path(config["data"][f"{split}_parquet"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[split]["parquet_sha256"]:
            raise ValueError(f"{split} dataset hash mismatch")
        query_ids.append(set(manifest[split]["query_ids"]))
    if query_ids[0] & query_ids[1]:
        raise ValueError("Train/validation query ID overlap")
    output = Path(config["runtime"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "config.toml"
    # Snapshot is flattened to remove mutable inheritance dependencies.
    from setwise.cli import dump_toml
    import copy
    snapshot_config = copy.deepcopy(config)
    snapshot_config["judge"]["cache_path"] = str(Path(config["judge"]["cache_path"]).resolve())
    snapshot.write_text(dump_toml(snapshot_config), encoding="utf-8")
    (output / "launch.json").write_text(json.dumps({"argv": cmd, "backend_commit": actual_commit,
        "config_sha256": fingerprint(config)}, indent=2), encoding="utf-8")
    env = dict(os.environ, SETWISE_CONFIG=str(snapshot), SETWISE_JUDGE_WORKERS=str(config["judge"]["workers"]))
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["TENSORBOARD_DIR"] = str(output / "tensorboard")
    subprocess.run(cmd, cwd=root, env=env, check=True)
    return {"status": "completed", "output": str(output)}
