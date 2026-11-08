"""Synthetic artifact fixtures exercise reporting only, never experiment claims."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from reasonforge.modeling import state_hash
from reasonforge.metrics import summarize
from reasonforge.utils import load_config, write_json, write_jsonl, file_hash, digest

spec = importlib.util.spec_from_file_location("make_report", Path(__file__).resolve().parents[1] / "scripts/make_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def artifact_fixture(root, name, stages=("base", "sft", "grpo")):
    run = root / name
    config = load_config("configs/full.yaml")
    config["name"] = name
    write_json(run / "config.json", config)
    write_json(run / "provenance.json", {"hardware": {"gpu_name": "SYNTHETIC TEST DEVICE", "gpu_total_mib": None},
               "dependencies": {"torch": "TEST-VERSION"}, "command": ["TEST FIXTURE ONLY"]})
    rows = [{"question_id": "test:0", "question": "TEST FIXTURE", "ground_truth": "2", "split": "test"}]
    for split in ("train", "validation", "test"):
        write_jsonl(run / f"data/{split}.jsonl", rows)
    write_json(run / "data/manifest.json", {"fixture": True})
    contract = {"test_hash": file_hash(run / "data/test.jsonl")}
    write_json(run / "evaluation_contract.json", contract)
    for index, stage in enumerate(stages):
        directory = run / "evaluation" / f"{stage}_test"
        predictions = [{**rows[0], "prediction": "Final Answer: 2", "extracted_answer": "2", "correct": True,
                        "parse_success": True, "generated_tokens": 3, "reached_eos": True,
                        "stage": stage, "seed": config["seed"]}]
        write_jsonl(directory / "predictions.jsonl", predictions)
        write_json(directory / "metrics.json", {**summarize(predictions), "stage": stage, "completion_rate": 1,
                   "resources": {"seconds": 1}, "prediction_sha256": file_hash(directory / "predictions.jsonl"),
                   "evaluation_contract_hash": digest(contract), "adapter_reload_verified": stage != "base"})
        if stage != "base":
            weights = {"TEST_ONLY": torch.tensor([float(index)])}
            target = run / stage / "final/adapter_model.safetensors"
            target.parent.mkdir(parents=True)
            save_file(weights, str(target))
            summary = {"resources": {"seconds": 1, "peak_allocated_mib": None, "peak_reserved_mib": None},
                       "train_samples": 1, "weight_update": {"after_hash": state_hash(weights), "max_abs_delta": 1}}
            if stage.startswith("grpo"):
                summary["parent_adapter_sha256"] = file_hash(run / "sft/final/adapter_model.safetensors")
                summary["sft_start_hash"] = json.loads((run / "sft/summary.json").read_text(encoding="utf-8"))["weight_update"]["after_hash"]
            write_json(run / stage / "summary.json", summary)
    return run


def test_full_report_uses_actual_artifact_status_and_settings(tmp_path):
    run = artifact_fixture(tmp_path, "full")
    before = {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "preview"
    report.build_reports([run], output)
    text = (output / "experiment_report.md").read_text(encoding="utf-8")
    resume = (output / "resume_bullets.md").read_text(encoding="utf-8")
    assert "Full 核心三阶段已有完整评测产物" in text
    assert "| full | 全部阶段 | 未运行" not in text
    assert "TEST-VERSION" in text and "SYNTHETIC TEST DEVICE" in text
    assert "完整数据规模核心评测已有证据" in resume
    assert "Small 单 seed 固定测试子集结果" not in resume
    assert before == {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}


def test_partial_full_is_never_reported_as_complete(tmp_path):
    run = artifact_fixture(tmp_path, "full", stages=("base",))
    (run / "evaluation/sft_test").mkdir()
    output = tmp_path / "partial-preview"
    report.build_reports([run], output)
    text = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert "未完成（不计指标）" in text
    assert "Full 核心三阶段尚无完整产物" in text
    assert "完整规模核心评测证据尚不齐全" in (output / "resume_bullets.md").read_text(encoding="utf-8")


def test_small_report_does_not_invent_full_or_training_results(tmp_path):
    run = artifact_fixture(tmp_path, "small", stages=("base",))
    output = tmp_path / "small-preview"
    report.build_reports([run], output)
    text = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert "本报告未提供的规模：smoke、full" in text
    assert "| full |" not in text
    assert "未提供 pytest XML" in text


def test_report_rejects_tampered_prediction_and_output_inside_run(tmp_path):
    run = artifact_fixture(tmp_path, "small")
    with pytest.raises(ValueError, match="outside"):
        report.build_reports([run], run / "preview")
    path = run / "evaluation/base_test/predictions.jsonl"
    rows = [json.loads(path.read_text(encoding="utf-8"))]
    rows[0]["correct"] = False
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="parser"):
        report.build_reports([run], tmp_path / "bad-preview")


def test_legacy_grpo_summary_uses_recorded_initial_weight_hash(tmp_path):
    run = artifact_fixture(tmp_path, "smoke")
    path = run / "grpo/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["weight_update"]["before_hash"] = summary.pop("sft_start_hash")
    write_json(path, summary)
    report.build_reports([run], tmp_path / "legacy-preview")
    assert (tmp_path / "legacy-preview/experiment_report.md").exists()
