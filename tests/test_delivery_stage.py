"""Recovery curves, ablation boundaries and unattended delivery failure guards."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from reasonforge.utils import write_json, write_jsonl

spec = importlib.util.spec_from_file_location("delivery_stage", Path(__file__).parents[1] / "scripts/run_delivery_stage.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_loss_curve_discards_abandoned_branch_without_averaging():
    rows = [{"step": 1, "loss": 1}, {"step": 2, "loss": 20}, {"step": 3, "loss": 30},
            {"step": 2, "loss": 2}, {"step": 3, "eval_loss": 99}, {"step": 3, "loss": 3},
            {"step": 4, "loss": 4}]
    accepted, discarded = worker.accepted_loss_rows(rows)
    assert [r["loss"] for r in accepted] == [1, 2, 3, 4]
    assert discarded == 2


def test_paired_counts_distinguish_equal_accuracy_and_changed_answers():
    left = [{"question_id": str(i), "correct": v} for i, v in enumerate([True, True, False, False])]
    right = [{"question_id": str(i), "correct": v} for i, v in enumerate([True, False, True, False])]
    assert worker.paired_counts(left, right) == {
        "both_correct": 1, "left_only_correct": 1, "right_only_correct": 1, "both_wrong": 1}
    with pytest.raises(ValueError, match="identical ordered"):
        worker.paired_counts(left, right[::-1])


def test_only_ablation_and_planned_evaluation_pauses_are_resumed(tmp_path, monkeypatch):
    (tmp_path / "stage6").mkdir()
    monkeypatch.setattr(worker, "verify_inputs", lambda *a: {})
    monkeypatch.setattr(worker, "storage_inventory", lambda *a: {})
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            write_json(tmp_path / "evaluation/grpo_correctness_only_test/progress.json", {"completed": 3})
            write_json(tmp_path / "status.json", {"status": "paused", "phase": "ablation"})
        else:
            write_json(tmp_path / "status.json", {"status": "completed", "completed_phase": "ablation"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    worker.run_batches(Path("full.yaml"), tmp_path, Path("checkpoint-64"))
    assert len(commands) == 2
    assert all(c[c.index("--phase") + 1] == "ablation" for c in commands)
    assert "--resume-checkpoint" in commands[0] and "--resume-checkpoint" not in commands[1]


@pytest.mark.parametrize("code,state", [(1, {"status": "failed"}),
    (0, {"status": "paused", "phase": "ablation"}),
    (0, {"status": "completed", "completed_phase": "grpo"})])
def test_failure_wrong_phase_or_no_progress_never_retried(tmp_path, monkeypatch, code, state):
    (tmp_path / "stage6").mkdir()
    monkeypatch.setattr(worker, "verify_inputs", lambda *a: {})
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        write_json(tmp_path / "status.json", state)
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    with pytest.raises(RuntimeError):
        worker.run_batches(Path("full.yaml"), tmp_path)
    assert len(calls) == 1


@pytest.fixture
def ablation_fixture(tmp_path):
    config = {"seed": 42, "grpo": {"max_steps": 1, "num_generations": 4, "reward_weights": [2.0, 0.2]}}
    write_json(tmp_path / "stage4/acceptance.json", {
        "final_adapter_sha256": "parent-file", "training": {"weight_update": {"after_hash": "parent-state"}}})
    write_json(tmp_path / "grpo_correctness_only/summary.json", {
        "parent_checkpoint": "sft/final", "parent_adapter_sha256": "parent-file", "sft_start_hash": "parent-state",
        "weight_update": {"before_hash": "parent-state", "parameters_changed": True}, "resumed": False,
        "correctness_only": True, "train_samples": 7345, "generations_seen": 4})
    write_jsonl(tmp_path / "grpo_correctness_only/train_log.jsonl", [{"step": 1, "loss": 0.1}])
    write_jsonl(tmp_path / "grpo_correctness_only/rollout_audit.jsonl", [
        {"call": 0, "n": 4, "reward_mean": 2.0, "reward_variance": 0.0,
         "correctness_rate": 1.0, "format_rate": 1.0, "zero_variance_group_fraction": 1.0}])
    row = {"call": 0, "question_id": "train:1", "ground_truth": "2", "prediction": "Final Answer: 2",
           "extracted_answer": "2", "stage": "grpo_correctness_only", "seed": 42,
           "correctness_reward": 1.0, "format_reward": 1.0, "weighted_reward": 2.0}
    write_jsonl(tmp_path / "grpo_correctness_only/rollouts.jsonl", [row] * 4)
    write_jsonl(tmp_path / "grpo/rollouts.jsonl", [row] * 4)
    write_jsonl(tmp_path / "data/train.jsonl", [{"question_id": "train:1", "ground_truth": "2"}])
    return config, {"grpo_correctness_only": {"adapter_reload_verified": True}}


def test_ablation_acceptance_checks_actual_reward_not_just_config(tmp_path, ablation_fixture):
    config, results = ablation_fixture
    assert worker.validate_ablation(config, tmp_path, results)["same_training_question_sequence"]
    path = tmp_path / "grpo_correctness_only/rollouts.jsonl"
    rows = worker.read_jsonl(path)
    rows[0]["weighted_reward"] = 2.2
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="exclude format"):
        worker.validate_ablation(config, tmp_path, results)


def test_ablation_rejects_changed_question_order(tmp_path, ablation_fixture):
    config, results = ablation_fixture
    write_jsonl(tmp_path / "grpo/rollouts.jsonl", [{"question_id": "different"}] * 4)
    with pytest.raises(ValueError, match="different training question"):
        worker.validate_ablation(config, tmp_path, results)


@pytest.fixture
def recovery_fixture(tmp_path):
    write_json(tmp_path / "stage6/recovery.json", {
        "checkpoint_files": {"adapter_model.safetensors": {"sha256": "checkpoint-file"}},
        "checkpoint_state_hash": "checkpoint-state", "rollout_batches_at_checkpoint": 2,
        "rollout_batches_before_resume": 3, "rollout_records_before_resume": 12})
    summary = {"resumed": True, "resume_adapter_sha256": "checkpoint-file",
               "weight_update": {"before_hash": "checkpoint-state"}}
    audits = [{"call": i, "n": 4} for i in range(5)]
    rollouts = [{"call": i, "question_id": str(i)} for i in range(5) for _ in range(4)]
    return summary, audits, rollouts


def test_resumed_audit_selects_checkpoint_prefix_plus_new_attempt(tmp_path, recovery_fixture):
    summary, audits, rollouts = recovery_fixture
    selected_audits, selected_rollouts = worker.accepted_ablation_records(tmp_path, summary, audits, rollouts, 4)
    assert [a["call"] for a in selected_audits] == [0, 1, 3, 4]
    assert [r["call"] for r in selected_rollouts] == [i for i in [0, 1, 3, 4] for _ in range(4)]
    assert len(audits) == 5 and len(rollouts) == 20  # Raw evidence remains intact.


def test_resumed_audit_rejects_wrong_checkpoint(tmp_path, recovery_fixture):
    summary, audits, rollouts = recovery_fixture
    summary["resume_adapter_sha256"] = "wrong-checkpoint"
    with pytest.raises(ValueError, match="verified checkpoint"):
        worker.accepted_ablation_records(tmp_path, summary, audits, rollouts, 4)


def test_resumed_audit_rejects_missing_call(tmp_path, recovery_fixture):
    summary, audits, rollouts = recovery_fixture
    with pytest.raises(ValueError, match="boundary"):
        worker.accepted_ablation_records(tmp_path, summary, audits, rollouts[:-1], 4)


def test_document_updates_refuse_changes_made_after_launch(tmp_path, monkeypatch):
    write_json(tmp_path / "stage6/provenance.json", {"delivery_inputs_sha256": {"README.md": "previous-hash"}})
    (tmp_path / "README.md").write_text("user edit", encoding="utf-8")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="document changed"):
        worker.update_documents(tmp_path, {}, {}, {})
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "user edit"


def test_document_delivery_uses_real_report_builder_in_isolated_fixture(tmp_path, monkeypatch):
    """Rehearse report generation without touching any actual project document."""
    import shutil
    from test_report import artifact_fixture
    from reasonforge.utils import file_hash
    source = worker.ROOT
    workspace = tmp_path / "workspace"
    # Private delivery documents need not be distributed in a public checkout.
    source_documents = [n for n in worker.DOCUMENTS if (source / n).is_file()]
    for name in worker.DOCUMENTS:
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Delivery fixture\n", encoding="utf-8")
    (workspace / "scripts").mkdir()
    shutil.copy2(source / "scripts/make_report.py", workspace / "scripts/make_report.py")
    # This one-shot finalizer consumes a pre-delivery README. The repository's
    # current README is already finalized and may be edited for publication.
    # Keep the contract in the fixture, independent of the live document text.
    (workspace / "README.md").write_text(
        "# Fixture\n\n阶段 1 已完成运行准备\n\n阶段 2 的记录位于 fixture。\n",
        encoding="utf-8")
    run = artifact_fixture(tmp_path, "full", stages=worker.STAGES)
    write_json(run / "storage_policy.json", {"max_run_bytes": 5 * 2**30, "min_free_bytes": 1})
    write_json(run / "stage6/provenance.json", {
        "delivery_inputs_sha256": {n: file_hash(workspace / n) for n in worker.DOCUMENTS}})
    xml = run / "stage6/validation/final/pytest.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', encoding="utf-8")
    monkeypatch.setattr(worker, "ROOT", workspace)
    monkeypatch.setattr(worker, "RUNS", ())
    from reasonforge.analyze import collect_results
    before = {n: file_hash(source / n) for n in source_documents}
    pairs = worker.update_documents(run, collect_results(run), {"passed": 1}, {
        "unique_training_questions_seen": 1, "same_training_question_sequence": True})
    assert pairs["sft_vs_grpo"]["both_correct"] == 1
    assert "阶段 1–6 已完成" in (workspace / "README.md").read_text(encoding="utf-8")
    assert "Full 实验解释与恢复记录" in (workspace / "reports/experiment_report.md").read_text(encoding="utf-8")
    assert "Codex" in (workspace / "reports/resume_bullets.md").read_text(encoding="utf-8")
    assert before == {n: file_hash(source / n) for n in source_documents}
