"""Guard checkpoint recovery without replacing completed experiment artifacts."""
import json
from pathlib import Path


def prepare_stage(run, stage, config, resume=None):
    directory = Path(run) / stage
    if resume is None:
        directory.mkdir(exist_ok=False)
        return directory
    checkpoint = Path(resume).resolve()
    required = ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete recovery checkpoint; missing {missing}")
    if (directory / "final").exists() or (directory / "summary.json").exists():
        raise FileExistsError("Completed stage is protected; use a fresh run for a new experiment")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    section = "sft" if stage == "sft" else "grpo"
    if not 0 < state["global_step"] < config[section]["max_steps"]:
        raise ValueError("Checkpoint must be a partial run before the configured final step")
    parent_config = checkpoint.parent.parent / "config.json"
    if not parent_config.is_file():
        raise ValueError("Recovery checkpoint must retain its source run config.json")
    original = json.loads(parent_config.read_text(encoding="utf-8"))
    for key in ("model", "data", "seed", "lora", section):
        if original[key] != config[key]:
            raise ValueError(f"Recovery configuration differs for {key}")
    if checkpoint.parent.name != stage:
        raise ValueError("Recovery checkpoint stage mismatch")
    if directory.exists():
        for existing in directory.glob("checkpoint-*"):
            suffix = existing.name.rsplit("-", 1)[-1]
            if suffix.isdigit() and int(suffix) > state["global_step"]:
                raise FileExistsError("A newer checkpoint exists; refusing to overwrite its history")
    directory.mkdir(exist_ok=True)
    return directory
