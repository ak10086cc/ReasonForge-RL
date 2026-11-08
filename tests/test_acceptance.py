"""Audit explicit real run paths; never manufacture experimental evidence."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from reasonforge.metrics import summarize, extract_answer
from reasonforge.modeling import state_hash
from reasonforge.utils import digest, file_hash, read_jsonl


def test_real_run_integrity(artifact_run):
    run = artifact_run
    manifest = json.loads((run / "data/manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((run/"provenance.json").read_text(encoding="utf-8"))
    if (run/"source_snapshot.zip").is_file():
        import zipfile, hashlib
        with zipfile.ZipFile(run/"source_snapshot.zip") as archive:
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == value
                       for name,value in provenance["source_hashes"].items())
        if "source_snapshot_sha256" in provenance:
            assert provenance["source_snapshot_sha256"] == file_hash(run/"source_snapshot.zip")
    splits = {split: read_jsonl(run / f"data/{split}.jsonl") for split in ("train", "validation", "test")}
    ids = {split: {row["question_id"] for row in rows} for split, rows in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert all(row["source"] == "official_train" for split in ("train", "validation") for row in splits[split])
    assert all(row["source"] == "official_test" for row in splits["test"])
    for split, rows in splits.items():
        assert digest(rows) == manifest["splits"][split]["selected_hash"]
    contract_hashes = set()
    stages = ["base", "sft", "grpo"] + (["grpo_correctness_only"] if config["ablation"] else [])
    for stage in stages:
        directory = run / "evaluation" / f"{stage}_test"
        rows = read_jsonl(directory / "predictions.jsonl")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        assert [r["question_id"] for r in rows] == [r["question_id"] for r in splits["test"]]
        for row in rows:
            extracted = extract_answer(row["prediction"])
            assert row["extracted_answer"] == extracted
            assert row["parse_success"] == (extracted is not None)
            assert row["correct"] == (extracted == row["ground_truth"])
        assert metrics["prediction_sha256"] == file_hash(directory / "predictions.jsonl")
        assert all(metrics[k] == v for k, v in summarize(rows).items())
        contract_hashes.add(metrics["evaluation_contract_hash"])
        if stage != "base":
            assert metrics["adapter_reload_verified"]
            summary = json.loads((run / stage / "summary.json").read_text(encoding="utf-8"))
            saved = {k: v.float() for k, v in load_file(str(run / stage / "final/adapter_model.safetensors")).items()}
            assert state_hash(saved) == summary["weight_update"]["after_hash"]
            assert summary["resources"]["seconds"] > 0
            assert summary["weight_update"]["parameters_changed"]
    assert len(contract_hashes) == 1
    sft = json.loads((run / "sft/summary.json").read_text(encoding="utf-8"))
    for stage in stages:
        if not stage.startswith("grpo"):
            continue
        summary = json.loads((run / stage / "summary.json").read_text(encoding="utf-8"))
        assert summary.get("sft_start_hash", summary["weight_update"]["before_hash"]) == sft["weight_update"]["after_hash"]
        assert summary["parent_adapter_sha256"] == file_hash(run / "sft/final/adapter_model.safetensors")
