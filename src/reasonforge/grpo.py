import json
from pathlib import Path
import numpy as np
from trl import GRPOTrainer
from .data import training_rows
from .modeling import load_model, load_tokenizer, adapter_state, state_hash, weight_delta, ResourceMeter, release
from .prompts import grpo_record
from .rewards import correctness_reward, format_reward
from .training import common_args, log_callback
from .utils import write_json, write_jsonl, file_hash
from .recovery import prepare_stage
from .storage import storage_callback


class ResumeSafeGRPOTrainer(GRPOTrainer):
    """Restore root default adapter even when a frozen ref/ subdirectory exists.

    Transformers 5.22.1's generic PEFT loader otherwise enters the subdirectory
    branch and skips the default adapter stored at the checkpoint root.
    """
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target = self.model if model is None else model
        target.load_adapter(str(resume_from_checkpoint), adapter_name="default", is_trainable=True)
        target.set_adapter("default")
        from safetensors.torch import load_file
        expected = {name: tensor.float() for name, tensor in
                    load_file(str(Path(resume_from_checkpoint) / "adapter_model.safetensors")).items()}
        if state_hash(adapter_state(target)) != state_hash(expected):
            raise RuntimeError("GRPO checkpoint failed to restore the trained default adapter")


def train(config, run, *, correctness_only=False, resume=None):
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig
    from peft import get_peft_model_state_dict
    set_seed(config["seed"])
    stage = "grpo_correctness_only" if correctness_only else "grpo"
    directory = prepare_stage(run, stage, config, resume)
    parent = Path(run) / "sft" / "final"
    sft_summary = json.loads((Path(run) / "sft" / "summary.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(config)
    records = [grpo_record(row, tokenizer) for row in training_rows(run)]
    write_jsonl(directory / "train_formatted.jsonl", records[:2])
    model = load_model(config, parent, trainable=True)
    before = adapter_state(model)
    if state_hash(before) != sft_summary["weight_update"]["after_hash"]:
        raise RuntimeError("GRPO is not starting from the saved SFT weights")
    c = config["grpo"]
    args = GRPOConfig(**common_args(config, directory, "grpo"),
                      num_generations=c["num_generations"], max_completion_length=c["max_completion_length"],
                      temperature=c["temperature"], top_p=1.0, top_k=0,
                      beta=c["beta"], loss_type="grpo", scale_rewards="group", use_vllm=False,
                      reward_weights=[c["reward_weights"][0]] if correctness_only else c["reward_weights"],
                      mask_truncated_completions=False, log_completions=False)
    rewards = [correctness_reward] if correctness_only else [correctness_reward, format_reward]
    # Extra rollout audit is local only. Never inject rewards into generated prompts.
    def audited_correctness(completions, ground_truth, **kwargs):
        scores = correctness_reward(completions, ground_truth)
        formats = format_reward(completions)
        total = np.array(scores) * c["reward_weights"][0]
        if not correctness_only:
            total += np.array(formats) * c["reward_weights"][1]
        groups = total.reshape(-1, c["num_generations"])
        group_variance = groups.var(axis=1)
        audit = {"call": audited_correctness.calls, "n": len(scores),
                 "correctness_rate": float(np.mean(scores)), "format_rate": float(np.mean(formats)),
                 "reward_mean": float(total.mean()), "reward_variance": float(total.var()),
                 "zero_variance_group_fraction": float(np.mean(group_variance == 0)),
                 "all_rewards_identical": bool(np.all(total == total[0]))}
        with (directory / "rollout_audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit) + "\n")
        from .rewards import completion_text
        from .metrics import extract_answer
        question_ids = kwargs.get("question_id", [None] * len(completions))
        with (directory / "rollouts.jsonl").open("a", encoding="utf-8") as f:
            for index, completion in enumerate(completions):
                text = completion_text(completion)
                record = {"call": audited_correctness.calls, "question_id": question_ids[index],
                          "ground_truth": ground_truth[index], "prediction": text,
                          "extracted_answer": extract_answer(text), "correctness_reward": scores[index],
                          "format_reward": formats[index], "weighted_reward": float(total[index]),
                          "stage": stage, "seed": config["seed"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        audited_correctness.calls += 1
        return scores
    from .utils import read_jsonl
    audit_file = directory / "rollout_audit.jsonl"
    audited_correctness.calls = len(read_jsonl(audit_file)) if audit_file.exists() else 0
    audited_correctness.__name__ = "correctness_reward"
    rewards[0] = audited_correctness
    # A loaded trainable PeftModel is passed without peft_config. TRL copies its
    # default adapter into a frozen ref adapter for beta > 0 (verified v1.17.1).
    trainer = ResumeSafeGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(records),
                          processing_class=tokenizer, reward_funcs=rewards,
                          callbacks=[log_callback(directory / "train_log.jsonl"), storage_callback(run)])
    if state_hash(adapter_state(trainer.model)) != state_hash(before):
        raise RuntimeError("TRL reinitialized SFT adapter")
    if c["beta"] > 0:
        import torch
        reference = get_peft_model_state_dict(trainer.model, adapter_name="ref", save_embedding_layers=False)
        default = get_peft_model_state_dict(trainer.model, adapter_name="default", save_embedding_layers=False)
        if set(reference) != set(default) or not all(torch.equal(reference[k], default[k]) for k in default):
            raise RuntimeError("KL reference does not match the initial SFT policy")
        if any(p.requires_grad for n, p in trainer.model.named_parameters() if ".ref." in n):
            raise RuntimeError("KL reference adapter must remain frozen")
    if resume:
        from safetensors.torch import load_file
        before = {name: tensor.float() for name, tensor in
                  load_file(str(Path(resume) / "adapter_model.safetensors")).items()}
    with ResourceMeter() as meter:
        result = trainer.train(resume_from_checkpoint=resume)
    after = adapter_state(trainer.model)
    changes = weight_delta(before, after)
    trainer.model.save_pretrained(directory / "final", selected_adapters=["default"], save_embedding_layers=False)
    tokenizer.save_pretrained(directory / "final")
    from .utils import read_jsonl
    audits = read_jsonl(directory / "rollout_audit.jsonl")
    zero_variance = all(row["all_rewards_identical"] for row in audits)
    warnings = []
    if not changes["parameters_changed"]:
        warnings.append("No adapter parameter update")
    if zero_variance:
        warnings.append("All rollout rewards identical: no relative-reward learning signal; changes may be KL/numerical")
    summary = {"stage": stage, "resources": meter.result, "train": result.metrics,
               "weight_update": changes, "checkpoint": f"{stage}/final", "parent_checkpoint": "sft/final",
               "parent_adapter_sha256": file_hash(parent / "adapter_model.safetensors"),
               "sft_start_hash": sft_summary["weight_update"]["after_hash"],
               "warning": "; ".join(warnings) if warnings else None,
               "train_samples": len(records), "correctness_only": correctness_only,
               "generations_seen": sum(row["n"] for row in audits), "resumed": resume is not None,
               "resource_scope": "current invocation only",
               "resume_adapter_sha256": file_hash(Path(resume) / "adapter_model.safetensors") if resume else None}
    write_json(directory / "summary.json", summary)
    del trainer, model
    release()
    return summary
