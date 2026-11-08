"""Pinned model loading, adapter continuity and actual resource measurements."""
import gc
import hashlib
from pathlib import Path
import time
from .utils import configure_runtime


def device_dtype(config):
    import torch
    requested = config["device"]
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use an explicit CPU config for CPU tests")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    return device, dtype


def load_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config, adapter=None, trainable=False):
    configure_runtime()
    from transformers import AutoModelForCausalLM
    device, dtype = device_dtype(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], dtype=dtype,
        attn_implementation="sdpa",
    )
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    model.to(device)
    return model


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {name: value.detach().float().cpu().clone() for name, value in
            get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False).items()}


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].contiguous().numpy().tobytes())
    return h.hexdigest()


def weight_delta(before, after):
    if set(before) != set(after):
        raise RuntimeError("Adapter structure changed during training")
    delta = max(float((after[k] - before[k]).abs().max()) for k in before)
    return {"before_hash": state_hash(before), "after_hash": state_hash(after),
            "max_abs_delta": delta, "parameters_changed": delta > 0}


class ResourceMeter:
    def __enter__(self):
        import torch
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import torch
        if self.cuda:
            torch.cuda.synchronize()
        self.result = {"seconds": time.perf_counter() - self.start,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if self.cuda else None,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if self.cuda else None}


def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
