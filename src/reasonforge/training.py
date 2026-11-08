"""Shared TRL logging and training settings."""
import json
import math
from pathlib import Path


def common_args(config, stage_dir, section):
    from .modeling import device_dtype
    device, dtype = device_dtype(config)
    import torch
    c = config[section]
    return dict(
        output_dir=str(stage_dir), learning_rate=c["learning_rate"], max_steps=c["max_steps"],
        per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, optim="adamw_torch", lr_scheduler_type="constant",
        bf16=dtype == torch.bfloat16, fp16=False, use_cpu=device == "cpu",
        logging_steps=1, save_steps=c["save_steps"], save_strategy="steps", save_total_limit=2,
        report_to="none", push_to_hub=False, seed=config["seed"], data_seed=config["seed"],
        dataloader_num_workers=0, dataloader_pin_memory=device == "cuda",
        disable_tqdm=True,
    )


def log_callback(path):
    from transformers import TrainerCallback

    class LocalLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            record = {"step": state.global_step, **(logs or {})}
            for value in record.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise FloatingPointError("Non-finite training metric")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with Path(path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return LocalLog()
