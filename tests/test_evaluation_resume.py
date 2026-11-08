"""Real tiny-model generation checks; these are not GSM8K experiment results."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed

from reasonforge import evaluate as evaluation
from reasonforge.utils import load_config, read_jsonl, write_jsonl, file_hash
from reasonforge.utils import write_json


@pytest.fixture
def tiny_evaluation(monkeypatch, tmp_path):
    set_seed(42)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "one": 3, "two": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             pad_token_id=0, eos_token_id=1))
    calls = []

    def load_model(config, adapter=None):
        result = copy.deepcopy(model)
        original = result.generate

        def generate(**kwargs):
            calls.append(1)
            return original(**kwargs)
        result.generate = generate
        return result

    monkeypatch.setattr(evaluation, "load_tokenizer", lambda config: tokenizer)
    monkeypatch.setattr(evaluation, "load_model", load_model)
    config = load_config("configs/smoke.yaml")
    config["device"] = "cpu"
    config["evaluation"]["max_new_tokens"] = 3

    def make_run(name):
        run = tmp_path / name
        for split in ("test", "validation"):
            rows = [{"question_id": f"{split}:{i}", "question": f"one two {i}", "ground_truth": str(i),
                     "split": split} for i in range(3)]
            write_jsonl(run / "data" / f"{split}.jsonl", rows)
        evaluation.freeze_evaluation(config, run)
        return run
    return config, make_run, calls


def test_chunked_real_generation_equals_uninterrupted(tiny_evaluation):
    config, make_run, calls = tiny_evaluation
    full, chunked = make_run("full"), make_run("chunked")
    expected = evaluation.evaluate(config, full)
    paused = evaluation.evaluate(config, chunked, max_samples=1)
    directory = chunked / "evaluation/base_test"
    assert paused["status"] == "paused" and paused["completed"] == 1
    assert not (directory / "metrics.json").exists()
    actual = evaluation.evaluate(config, chunked)
    assert read_jsonl(full / "evaluation/base_test/predictions.jsonl") == read_jsonl(directory / "predictions.jsonl")
    for key in ("n", "accuracy", "parse_rate", "correct", "mean_generated_tokens"):
        assert actual[key] == expected[key]
    assert len(calls) == 6  # Finished questions are never generated again.
    before = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}
    assert evaluation.evaluate(config, chunked) == actual
    assert before == {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}
    assert len(calls) == 6
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    assert sum(s["new_predictions"] for s in progress["sessions"]) == 3


def test_time_budget_finishes_current_question(tiny_evaluation, monkeypatch):
    config, make_run, calls = tiny_evaluation
    run = make_run("deadline")
    ticks = iter([100.0, 101.0])
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert evaluation.evaluate(config, run, max_seconds=1e-9)["completed"] == 1
    assert len(calls) == 1


def test_keyboard_interrupt_retains_completed_predictions(tiny_evaluation, monkeypatch):
    config, make_run, calls = tiny_evaluation
    run = make_run("interrupt")
    original = evaluation.load_model

    def load(config, adapter):
        model = original(config, adapter)
        generate = model.generate

        def interrupted(**kwargs):
            if len(calls) == 1:
                raise KeyboardInterrupt
            return generate(**kwargs)
        model.generate = interrupted
        return model
    monkeypatch.setattr(evaluation, "load_model", load)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate(config, run)
    assert len(read_jsonl(run / "evaluation/base_test/predictions.jsonl")) == 1
    monkeypatch.setattr(evaluation, "load_model", original)
    assert evaluation.evaluate(config, run)["n"] == 3


@pytest.mark.parametrize("change", ["config", "data", "manifest", "prediction", "order"])
def test_resume_rejects_changed_identity_or_prefix(tiny_evaluation, change):
    config, make_run, calls = tiny_evaluation
    run = make_run(change)
    evaluation.evaluate(config, run, max_samples=2)
    directory = run / "evaluation/base_test"
    if change == "config":
        config["evaluation"]["max_new_tokens"] += 1
    elif change == "data":
        rows = read_jsonl(run / "data/test.jsonl")
        rows[0]["question"] = "changed"
        write_jsonl(run / "data/test.jsonl", rows)
    elif change == "manifest":
        path = directory / "resume_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["evaluator_source_sha256"] = "changed"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path = directory / "predictions.jsonl"
        rows = read_jsonl(path)
        if change == "prediction":
            rows[0]["prediction"] += " changed"
        else:
            rows.reverse()
        write_jsonl(path, rows)
    with pytest.raises(ValueError):
        evaluation.evaluate(config, run)
    assert len(calls) == 2


def test_incomplete_legacy_and_torn_json_are_preserved(tiny_evaluation):
    config, make_run, calls = tiny_evaluation
    run = make_run("legacy")
    directory = run / "evaluation/base_test"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="no resume manifest"):
        evaluation.evaluate(config, run)
    run = make_run("torn")
    evaluation.evaluate(config, run, max_samples=1)
    path = run / "evaluation/base_test/predictions.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"question_id":')
    before = path.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        evaluation.evaluate(config, run)
    assert path.read_bytes() == before


@pytest.mark.parametrize("options", [{"max_samples": 0}, {"max_seconds": float("nan")}, {"max_seconds": -1}])
def test_invalid_budgets_create_no_evaluation(tiny_evaluation, options):
    config, make_run, calls = tiny_evaluation
    run = make_run("invalid")
    with pytest.raises(ValueError):
        evaluation.evaluate(config, run, **options)
    assert not (run / "evaluation").exists()


def _pipeline():
    spec = importlib.util.spec_from_file_location("pipeline_resume_test", Path(__file__).resolve().parents[1] / "scripts/run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_baseline_pause_stops_before_test_then_resumes(tiny_evaluation, monkeypatch):
    import yaml
    config, make_run, calls = tiny_evaluation
    run = make_run("cli")
    # Start before the baseline validation/freeze boundary.
    (run / "evaluation_contract.json").unlink()
    write_json(run / "config.json", config)
    config_file = run / "test-config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    pipeline = _pipeline()
    argv = ["run_pipeline.py", "--config", str(config_file), "--phase", "baseline", "--run-dir", str(run)]
    monkeypatch.setattr(sys, "argv", argv + ["--eval-max-samples", "1"])
    pipeline.main()
    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "paused"
    assert not (run / "evaluation_contract.json").exists()
    assert not (run / "evaluation/base_test").exists()
    assert not (run / "failures.jsonl").exists()
    monkeypatch.setattr(sys, "argv", argv)
    pipeline.main()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed" and status["completed_phase"] == "baseline"
    assert len(calls) == 6  # 3 validation and 3 test, without repeats.


def test_cli_finished_sft_reuses_model_without_training(tiny_evaluation, monkeypatch):
    import yaml
    from reasonforge import sft, analyze
    config, make_run, calls = tiny_evaluation
    run = make_run("sft-eval-routing")
    write_json(run / "config.json", config)
    write_json(run / "sft/summary.json", {"test_fixture_only": True})
    (run / "sft/final").mkdir()
    config_file = run / "test-config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    invoked = []

    def forbidden_train(*args, **kwargs):
        pytest.fail("Finished training was restarted")

    def evaluation_only(*args, **kwargs):
        invoked.append(args[2])
        return {"status": "paused", "completed": 1}
    monkeypatch.setattr(sft, "train", forbidden_train)
    monkeypatch.setattr(analyze, "collect_results", lambda run: {"base": {"test_fixture_only": True}})
    monkeypatch.setattr(evaluation, "evaluate", evaluation_only)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(config_file), "--phase", "sft", "--run-dir", str(run)])
    _pipeline().main()
    assert invoked == ["sft"]
    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "paused"


def test_cli_rejects_concurrent_run_without_modifying_status(tiny_evaluation, monkeypatch):
    import yaml
    from filelock import FileLock
    config, make_run, calls = tiny_evaluation
    run = make_run("locked")
    write_json(run / "config.json", config)
    config_file = run / "test-config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(config_file), "--phase", "baseline", "--run-dir", str(run)])
    with FileLock(run / ".run.lock", timeout=0):
        with pytest.raises(SystemExit) as error:
            _pipeline().main()
        assert error.value.code == 2
    assert not (run / "status.json").exists()
    assert not calls


def test_frozen_contract_rejects_inference_code_and_dependency_changes(tiny_evaluation, monkeypatch):
    config, make_run, calls = tiny_evaluation
    run = make_run("code-pin")
    original = evaluation.file_hash
    monkeypatch.setattr(evaluation, "file_hash", lambda path: "changed" if Path(path).name == "modeling.py" else original(path))
    with pytest.raises(ValueError, match="differ"):
        evaluation.evaluate(config, run)
    monkeypatch.setattr(evaluation, "file_hash", original)
    monkeypatch.setattr(evaluation, "versions", lambda: {"transformers": "changed"})
    with pytest.raises(ValueError, match="differ"):
        evaluation.evaluate(config, run)
    assert not calls


def test_sft_cannot_start_before_verified_base(tiny_evaluation, monkeypatch):
    import yaml
    config, make_run, calls = tiny_evaluation
    run = make_run("missing-base")
    write_json(run / "config.json", config)
    config_file = run / "test-config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(config_file), "--phase", "sft", "--run-dir", str(run)])
    with pytest.raises(SystemExit) as error:
        _pipeline().main()
    assert error.value.code == 1
    assert not (run / "sft").exists() and not calls
    assert "requires completed" in (run / "failures.jsonl").read_text(encoding="utf-8")
