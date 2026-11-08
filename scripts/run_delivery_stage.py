"""Finish authorized ablation, evidence-based reports and final acceptance locally."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reasonforge.utils import (configure_runtime, digest, file_hash, load_config,
                              read_jsonl, versions, write_json)
from reasonforge.storage import check_storage, directory_bytes, storage_inventory

STAGES = ("base", "sft", "grpo", "grpo_correctness_only")
RUNS = ("outputs/runs/smoke-20261106T035154814916Z",
        "outputs/runs/small-20261106T032826964295Z")
DOCUMENTS = ("README.md", "reports/experiment_report.md", "reports/resume_bullets.md",
             "reports/interview_qa.md", "reports/environment.md")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_text(path, text):
    """Keep an existing document's newline convention."""
    path = Path(path)
    newline = "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", newline).encode("utf-8"))


def verify_inputs(config_path, run):
    from reasonforge.analyze import collect_results
    from reasonforge.evaluate import evaluation_contract
    config = load_config(config_path)
    parent = read(run / "stage5/acceptance.json")
    if (config != read(run / "config.json") or config["name"] != "full"
            or parent["status"] != "complete" or parent["stopped_before_stage"] != 6):
        raise ValueError("Unchanged Full config and complete Stage 5 are required")
    for name, key in (("config.json", "configuration_sha256"),
                      ("data/manifest.json", "data_manifest_sha256"),
                      ("evaluation_contract.json", "evaluation_contract_sha256")):
        if file_hash(run / name) != parent[key]:
            raise ValueError(f"Accepted input changed: {name}")
    original = read(run / "provenance.json")
    if file_hash(run / "source_snapshot.zip") != original["source_snapshot_sha256"]:
        raise ValueError("Original source archive changed")
    hashes = dict(original["source_hashes"])
    previous = run / "provenance.json"
    for stage in (4, 5, 6):
        addition = read(run / f"stage{stage}/provenance.json")
        if (file_hash(previous) != addition["parent_provenance_sha256"]
                or file_hash(run / f"stage{stage}/source_additions.zip") != addition["source_snapshot_sha256"]):
            raise ValueError("Stage source provenance changed")
        hashes.update(addition["source_hashes"])
        previous = run / f"stage{stage}/provenance.json"
    recovery_path = run / "stage6/recovery.json"
    if recovery_path.exists():
        recovery = read(recovery_path)
        replacement = recovery.get("source_after_fix_sha256", {})
        if replacement:
            if file_hash(run / "stage6/recovery_sources.zip") != recovery["recovery_sources_sha256"]:
                raise ValueError("Recovery source archive changed")
            for name, expected in replacement.items():
                if hashes.get(name) != recovery["source_before_fix_sha256"].get(name):
                    raise ValueError("Recovery amendment does not match original Stage 6 source")
                hashes[name] = expected
    for name, expected in hashes.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen source/config changed: {name}")
    if versions() != original["dependencies"]:
        raise ValueError("Dependencies changed since Base")
    manifest = read(run / "data/manifest.json")
    question_sets = []
    for split, count in (("train", 7345), ("validation", 128), ("test", 1319)):
        rows = read_jsonl(run / "data" / f"{split}.jsonl")
        expected = manifest["splits"][split]
        source = "official_test" if split == "test" else "official_train"
        if (len(rows) != count or expected["selected_count"] != count
                or digest(rows) != expected["selected_hash"]
                or [r["question_id"] for r in rows] != expected["selected_ids"]
                or any(r["split"] != split or r["source"] != source for r in rows)):
            raise ValueError(f"Full {split} membership or contents changed")
        question_sets.append({r["question_hash"] for r in rows})
    if any(question_sets[i] & question_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Data splits overlap")
    if read(run / "evaluation_contract.json") != evaluation_contract(config, run):
        raise ValueError("Frozen evaluation contract changed")
    results = collect_results(run)
    if any(results.get(key) != parent[key] for key in STAGES[:3]):
        raise ValueError("Accepted Base/SFT/GRPO results changed")
    if (file_hash(run / "grpo/final/adapter_model.safetensors") != parent["final_adapter_sha256"]
            or read(run / "grpo/summary.json") != parent["training"]):
        raise ValueError("Accepted GRPO weights or summary changed")
    sft = read(run / "stage4/acceptance.json")
    if (file_hash(run / "sft/final/adapter_model.safetensors") != sft["final_adapter_sha256"]
            or read(run / "sft/summary.json") != sft["training"]):
        raise ValueError("Accepted SFT parent changed")
    check_storage(run)
    return config


def completed_questions(run):
    path = run / "evaluation/grpo_correctness_only_test/progress.json"
    return read(path)["completed"] if path.exists() else 0


def status(run, state, **fields):
    write_json(run / "stage6/worker_status.json", {
        "stage": 6, "status": state, "pid": os.getpid(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})


def run_batches(config_path, run, resume=None):
    stage = run / "stage6"
    invocation = len(list(stage.glob("ablation-*.log")))
    while True:
        verify_inputs(config_path, run)
        before = completed_questions(run)
        invocation += 1
        command = [sys.executable, "-u", str(ROOT / "scripts/run_pipeline.py"),
                   "--config", str(config_path), "--phase", "ablation", "--run-dir", str(run),
                   "--eval-max-seconds", "3600"]
        if resume is not None:
            command.extend(["--resume-checkpoint", str(resume)])
        log = stage / f"ablation-{invocation:04d}.log"
        status(run, "running", phase="ablation", invocation=invocation,
               command=command, log=str(log.relative_to(run)))
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Ablation exited {result.returncode}; no automatic failure retry; inspect {log.name}")
        saved = read(run / "status.json")
        if saved.get("status") == "completed" and saved.get("completed_phase") == "ablation":
            return
        if (saved.get("status") != "paused" or saved.get("phase") != "ablation"
                or completed_questions(run) <= before):
            raise RuntimeError("Unexpected ablation status or no evaluation progress")
        resume = None
        write_json(stage / "storage_latest.json", storage_inventory(run.parent))


def accepted_ablation_records(run, summary, audits, rollouts, group_size):
    if not summary["resumed"]:
        return audits, rollouts
    recovery = read(run / "stage6/recovery.json")
    if (summary["resume_adapter_sha256"] != recovery["checkpoint_files"]["adapter_model.safetensors"]["sha256"]
            or summary["weight_update"]["before_hash"] != recovery["checkpoint_state_hash"]):
        raise ValueError("Resumed ablation did not start from the verified checkpoint")
    at_checkpoint = recovery["rollout_batches_at_checkpoint"]
    before = recovery["rollout_batches_before_resume"]
    before_records = recovery["rollout_records_before_resume"]
    if (not 0 < at_checkpoint <= before < len(audits) or before_records != before * group_size
            or [a["call"] for a in audits] != list(range(len(audits)))
            or [r["call"] for r in rollouts] != [a["call"] for a in audits for _ in range(a["n"])]):
        raise ValueError("Recovery rollout boundary is inconsistent with preserved logs")
    return (audits[:at_checkpoint] + audits[before:],
            rollouts[:at_checkpoint * group_size] + rollouts[before_records:])


def validate_ablation(config, run, results):
    from reasonforge.metrics import extract_answer
    from reasonforge.rewards import correctness_reward, format_reward
    name = "grpo_correctness_only"
    summary = read(run / name / "summary.json")
    sft = read(run / "stage4/acceptance.json")
    logs = read_jsonl(run / name / "train_log.jsonl")
    steps = [r["step"] for r in logs if "loss" in r]
    audits = read_jsonl(run / name / "rollout_audit.jsonl")
    rollouts = read_jsonl(run / name / "rollouts.jsonl")
    main_rollouts = read_jsonl(run / "grpo/rollouts.jsonl")
    if (summary["parent_checkpoint"] != "sft/final"
            or summary["parent_adapter_sha256"] != sft["final_adapter_sha256"]
            or summary["sft_start_hash"] != sft["training"]["weight_update"]["after_hash"]
            or (not summary["resumed"] and summary["weight_update"]["before_hash"] != summary["sft_start_hash"])
            or not summary["weight_update"]["parameters_changed"] or not summary["correctness_only"]
            or summary["train_samples"] != 7345 or not results[name]["adapter_reload_verified"]
            or not steps or max(steps) != config["grpo"]["max_steps"]
            or not audits or summary["generations_seen"] != sum(a["n"] for a in audits)
            or len(rollouts) != summary["generations_seen"]
            or any(a["n"] != config["grpo"]["num_generations"] for a in audits)):
        raise ValueError("Ablation parent, training or rollout counts failed acceptance")
    accepted_audits, accepted_rollouts = accepted_ablation_records(
        run, summary, audits, rollouts, config["grpo"]["num_generations"])
    if (len(accepted_audits) != config["grpo"]["max_steps"]
            or len(accepted_rollouts) != config["grpo"]["max_steps"] * config["grpo"]["num_generations"]):
        raise ValueError("Successful ablation branch does not match the training budget")
    same_order = [r["question_id"] for r in accepted_rollouts] == [r["question_id"] for r in main_rollouts]
    if not same_order:
        raise ValueError("Ablation visited a different training question sequence")
    train = {r["question_id"]: r for r in read_jsonl(run / "data/train.jsonl")}
    grouped = {}
    for row in rollouts:
        if (row["question_id"] not in train or row["ground_truth"] != train[row["question_id"]]["ground_truth"]
                or row["stage"] != name or row["seed"] != config["seed"]
                or row["extracted_answer"] != extract_answer(row["prediction"])):
            raise ValueError("Ablation rollout is outside the training split or inconsistent")
        correct = correctness_reward([row["prediction"]], [row["ground_truth"]])[0]
        formatted = format_reward([row["prediction"]])[0]
        if (row["correctness_reward"] != correct or row["format_reward"] != formatted
                or abs(row["weighted_reward"] - correct * config["grpo"]["reward_weights"][0]) > 1e-8):
            raise ValueError("Ablation weighted rewards must exclude format reward")
        grouped.setdefault(row["call"], []).append(row)
    if len(grouped) != len(audits):
        raise ValueError("Rollout calls and audit count differ")
    for audit in audits:
        rows = grouped[audit["call"]]
        rewards = [r["weighted_reward"] for r in rows]
        mean = sum(rewards) / len(rewards)
        variance = sum((v - mean)**2 for v in rewards) / len(rewards)
        if (len(rows) != audit["n"] or abs(mean - audit["reward_mean"]) > 1e-8
                or abs(variance - audit["reward_variance"]) > 1e-8):
            raise ValueError("Ablation audit disagrees with saved rollout rewards")
    mean = lambda key: sum(a[key] for a in accepted_audits) / len(accepted_audits)
    diagnostics = {key: mean(key) for key in ("reward_mean", "reward_variance", "correctness_rate",
                                             "format_rate", "zero_variance_group_fraction")}
    return {"training": summary, "maximum_training_step": max(steps), "rollout_batches": len(accepted_audits),
            "accepted_generations_seen": len(accepted_rollouts), "raw_rollout_batches": len(audits),
            "abandoned_rollout_batches": len(audits) - len(accepted_audits),
            "accepted_reward_diagnostics": diagnostics,
            "same_training_question_sequence": same_order,
            "unique_training_questions_seen": len({r["question_id"] for r in accepted_rollouts})}


def accepted_loss_rows(rows):
    """Discard abandoned branches at a rollback; never average repeated steps."""
    accepted = {}
    count = 0
    for row in rows:
        if "loss" not in row:
            continue
        count += 1
        step = row["step"]
        for abandoned in [s for s in accepted if s >= step]:
            del accepted[abandoned]
        accepted[step] = row
    return list(accepted.values()), count - len(accepted)


def render_analysis(run):
    from reasonforge.analyze import analyze_run
    results = analyze_run(run)
    # The preserved historical analyzer plots append-only logs. Replace only
    # this new run's SFT plot with the actual successful recovery branch.
    rows, discarded = accepted_loss_rows(read_jsonl(run / "sft/train_log.jsonl"))
    target = read(run / "config.json")["sft"]["max_steps"]
    if [r["step"] for r in rows] != list(range(1, target + 1)):
        raise ValueError("Successful SFT loss history does not cover every configured step")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], linewidth=1)
    recovery = run / "stage4/recovery.json"
    if recovery.exists():
        ax.axvline(read(recovery)["checkpoint_global_step"], linestyle="--", color="gray", label="Checkpoint resume")
        ax.legend()
    ax.set(xlabel="Optimizer step", ylabel="loss", title="full / SFT (accepted training branch)")
    ax.grid(alpha=.25)
    fig.savefig(run / "figures/sft_loss.png", dpi=160)
    plt.close(fig)
    evidence = {"selected_steps": [r["step"] for r in rows], "abandoned_loss_records": discarded,
                "method": "On optimizer-step rollback, discard prior records at or after that step; preserve raw logs",
                "raw_log_sha256": file_hash(run / "sft/train_log.jsonl")}
    write_json(run / "stage6/curve_provenance.json", evidence)
    report = run / "report.md"
    write_text(report, report.read_text(encoding="utf-8") + "\nSFT 曲线仅采用成功训练分支，丢弃重复尝试记录 "
               f"{discarded} 条，原日志保留。SFT summary 的 1299.90 秒及显存仅覆盖从第 500 步恢复的调用，"
               "不代表全部尝试的总耗时或全程峰值；最初 CUDA unknown error 记录见 stage4/recovery.json。\n")
    if (run / "stage6/recovery.json").exists():
        write_text(report, report.read_text(encoding="utf-8") +
                   "\n正确性奖励消融曾在第 202 步后显存不足，从已核验的第 192 步检查点恢复。"
                   "其 summary 的资源与参数变化仅覆盖恢复调用，首次失败的完整耗时和峰值缺失。"
                   "奖励统计按成功分支排除放弃的尝试，详细依据见 stage6/recovery.json 与最终验收。\n")
    return results


def test_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    values = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    return {"passed": values["tests"] - values["failures"] - values["errors"] - values["skipped"],
            "failed": values["failures"], "errors": values["errors"], "skipped": values["skipped"],
            "seconds": sum(float(s.get("time", 0)) for s in suites)}


def run_validation(run):
    stage = run / "stage6"
    validation = stage / "validation/final"
    validation.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "pytest", "tests", "-q"]
    for artifact in (*RUNS, str(run)):
        command += ["--artifact-run", artifact]
    command += ["--basetemp", str(validation / "pytest-tmp"), "--junitxml", str(validation / "pytest.xml")]
    status(run, "running", phase="final_tests", command=command)
    with (validation / "pytest.log").open("x", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    counts = test_counts(validation / "pytest.xml")
    write_json(stage / "final_tests.json", {**counts, "command": command, "junit": "stage6/validation/final/pytest.xml"})
    if result.returncode or counts["failed"] or counts["errors"] or counts["skipped"]:
        raise RuntimeError("Final pytest did not pass without skips; inspect stage6/validation/final/pytest.log")
    return counts


def paired_counts(left, right):
    if [r["question_id"] for r in left] != [r["question_id"] for r in right]:
        raise ValueError("Paired comparison requires identical ordered questions")
    return {"both_correct": sum(a["correct"] and b["correct"] for a, b in zip(left, right)),
            "left_only_correct": sum(a["correct"] and not b["correct"] for a, b in zip(left, right)),
            "right_only_correct": sum(not a["correct"] and b["correct"] for a, b in zip(left, right)),
            "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in zip(left, right))}


def update_documents(run, results, tests, ablation):
    inputs = read(run / "stage6/provenance.json")["delivery_inputs_sha256"]
    recovery_path = run / "stage6/recovery.json"
    if recovery_path.exists():
        recovery = read(recovery_path)
        if recovery.get("delivery_inputs_after_recovery_sha256"):
            if recovery["delivery_inputs_before_recovery_sha256"] != inputs:
                raise ValueError("Recovery documents do not match launch provenance")
            inputs = recovery["delivery_inputs_after_recovery_sha256"]
    for name, expected in inputs.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Delivery document changed after launch; preserve and review: {name}")
    report_builder = module("delivery_report", "scripts/make_report.py")
    conventions = {name: b"\r\n" in (ROOT / name).read_bytes()
                   for name in ("reports/experiment_report.md", "reports/resume_bullets.md")}
    report_builder.build_reports([ROOT / p for p in RUNS] + [run], ROOT / "reports",
                                 run / "stage6/validation/final/pytest.xml")
    for name, crlf in conventions.items():
        if crlf:
            path = ROOT / name
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    predictions = {s: read_jsonl(run / f"evaluation/{s}_test/predictions.jsonl") for s in STAGES}
    pairs = {"sft_vs_grpo": paired_counts(predictions["sft"], predictions["grpo"]),
             "grpo_vs_correctness_only": paired_counts(predictions["grpo"], predictions["grpo_correctness_only"])}
    write_json(run / "stage6/paired_comparisons.json", pairs)
    delta = results["grpo"]["correct"] - results["sft"]["correct"]
    history = (
        "\n## Full 实验解释与恢复记录\n\n"
        f"SFT 与 GRPO 的正确数差为 {delta}，共同分母 1319；本次结果不支持 GRPO 提高准确率的结论。"
        "相同总正确数也不表示每题预测一致。配对统计如下（left/right 顺序见名称）：\n\n"
        + "\n".join(f"- {key}：{json.dumps(value, ensure_ascii=False)}。" for key, value in pairs.items())
        + "\n\nFull SFT 首次于 2026-11-07 14:12 出现 CUDA unknown error，原因未确定，未发现 OOM 记录；"
        "核验权重、优化器、调度器与 GPU 后，于 15:42 从第 500 步继续，最终达到第 920 步。"
        "失败与恢复证据保留在 stage4/recovery.json、failures.jsonl 和原日志。"
        "SFT summary 的 1299.90 秒和 1442.36/3402 MiB 仅覆盖成功恢复调用，"
        "首次失败调用的完整耗时和峰值缺失，不能合计或反推。"
        "SFT 曲线选择成功分支的 920 个 loss 点，排除被回滚的尝试，不对重复步骤取平均；"
        "原始日志保持不变，选择依据见 stage6/curve_provenance.json。\n\n"
        "GRPO 与消融各配置 256 个更新步，每题 4 个回答；7345 是候选训练集大小，"
        "不能声称 RL 遍历了全部 7345 题。消融实际不同训练问题数为 "
        f"{ablation['unique_training_questions_seen']}，与主 GRPO 的问题顺序一致性为 "
        f"{ablation['same_training_question_sequence']}。只去掉格式奖励，其余设置保持一致；"
        "单 seed 结果不构成稳定增益或奖励方案优劣的证明。\n\n"
        "Base/SFT 历史显卡驱动记录为 610.60，GRPO/消融阶段检测为 621.14；"
        "驱动变更并非项目操作，PyTorch CUDA wheel 12.8 与 Python 依赖未变。"
        "该环境差异为复现限制；冻结评测协议、输入、源码和权重衔接已核验。\n\n"
        "多 seed、独立 1.5B 和跨数据集泛化实验未运行。项目由 Codex 协助实现，"
        "简历使用者须能解释训练、奖励、数据隔离及负结果，不能宣称独立原创算法。\n\n"
        "实际执行：项目根目录专用 Python 运行 scripts/run_delivery_stage.py --config configs/full.yaml "
        "--run-dir outputs/runs/full-20261107T015504132102Z；该入口依次调用 ablation、"
        "分析、三个规模的 pytest 产物验收和报告生成。完整命令与原始日志见 stage6。\n")
    report = ROOT / "reports/experiment_report.md"
    report_text = report.read_text(encoding="utf-8")
    if recovery_path.exists() and ablation.get("accepted_reward_diagnostics"):
        diagnostics = ablation["accepted_reward_diagnostics"]
        rows = report_text.splitlines()
        indices = [i for i, line in enumerate(rows) if line.startswith("- grpo_correctness_only：")]
        if not indices:
            raise ValueError("Cannot locate Full ablation reward diagnostics")
        rows[indices[-1]] = (
            f"- grpo_correctness_only：成功训练分支 {ablation['rollout_batches']} 个 rollout 批次，"
            f"{ablation['accepted_generations_seen']} 个回答；批次等权 reward 均值={diagnostics['reward_mean']:.6f}、"
            f"方差均值={diagnostics['reward_variance']:.6f}；正确性={diagnostics['correctness_rate']:.2%}、"
            f"格式率={diagnostics['format_rate']:.2%}、零方差组比例={diagnostics['zero_variance_group_fraction']:.2%}。")
        report_text = "\n".join(rows) + "\n"
        history += (
            "\nFull 消融首次运行于 2026-11-07 21:57 报 CUDA out of memory，最后日志 202 步，"
            "已核验第 192 步完整检查点、优化器、调度器、冻结参考权重与当前 CUDA 运算，"
            "用新进程保持原配置恢复。具体诱因未确定，未修改系统内存设置。"
            f"原始审计共 {ablation['raw_rollout_batches']} 批，"
            f"其中 {ablation['abandoned_rollout_batches']} 批属于回滚放弃的尝试，"
            "正式奖励统计仅使用成功分支，原始文件保留。"
            "消融 summary 的耗时、峰值显存和参数变化仅覆盖从检查点恢复的调用，"
            "首次失败调用缺少完整资源记录，不冒充总耗时或全程峰值。证据见 stage6/recovery.json。\n")
    write_text(report, report_text + history)
    qa = ROOT / "reports/interview_qa.md"
    text = qa.read_text(encoding="utf-8")
    text = text.replace("比较只是小样本观察，多 seed 与更大测试集仍是后续工作。",
                        "Full 已在全部 1319 道官方测试题上比较；单 seed 与有限 RL 训练预算仍限制结论，多 seed 未运行。")
    text = text.replace("Small 提供指定预算下的观察。只有报告中实际完成的实验可写入简历，",
                        "Small 和 Full 提供各自预算下的观察。Full 三阶段和奖励消融已完成可追溯评测；只有报告中实际完成的实验可写入简历，")
    text += ("\n23. **这次 GRPO 提升了吗？** Full SFT 与 GRPO 都答对 422/1319（31.99%），"
             "因此没有测试准确率提升证据。检查权重变化、组内奖励差异、零方差比例和 KL，"
             "可区分训练发生了更新与测试效果改善；两者不能混为一谈。\n\n"
             "24. **为什么候选训练集 7345 题，不能说 GRPO 全量训练？** 候选集只是可抽取的问题池。"
             "本次只有 256 个优化步、每组 4 个回答，实际 rollout 访问数量记录在日志中；"
             "Full 指数据池及完整测试规模，不代表强化学习遍历全部训练题。\n\n"
             "25. **断点恢复后怎样画 loss 曲线？** 保存原日志，识别优化步回退，"
             "去掉已放弃分支，再连接实际保存的训练轨迹；本次保留 1–920 步，"
             "不把重复的 501–506 步取平均或计为额外训练。恢复调用的耗时也不能冒充总耗时。\n")
    write_text(qa, text)
    resume = ROOT / "reports/resume_bullets.md"
    write_text(resume, resume.read_text(encoding="utf-8") +
               "\n本项目由 Codex 协助实现；请按本人实际理解与参与程度使用表述。"
               "Full 指固定完整数据池与 1319 题测试；GRPO/消融各 256 步，未遍历全部训练题。"
               "本次 SFT 与 GRPO 均为 31.99%，不能写成强化学习提升准确率。\n")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("阶段 1 已完成运行准备")
    end = text.index("阶段 2 的记录位于", start)
    table = ["阶段 1–6 已完成本次既定 0.5B 实验与交付。正式 run："
             "`outputs/runs/full-20261107T015504132102Z/`；最终证据为 `stage6/acceptance.json`。"
             "各历史阶段的启动、失败和恢复记录保留在下文，当前状态以阶段 6 验收为准。", "",
             "| Full 阶段 | 正确数 / 1319 | 严格准确率 | 可解析率 |", "|---|---:|---:|---:|"]
    table += [f"| {s} | {results[s]['correct']} | {results[s]['accuracy']:.2%} | {results[s]['parse_rate']:.2%} |" for s in STAGES]
    table += ["", f"本轮最终 pytest：**{tests['passed']} 通过、0 失败、0 跳过**。SFT 与 GRPO 同为 422/1319，"
              "本次没有观察到 GRPO 准确率增益。Full 指完整数据池与测试规模，RL 为 256 步预算；"
              "单 seed、严格格式和驱动版本差异等限制见实验报告。", "", "历史阶段证据与资源依据：", ""]
    text = text[:start] + "\n".join(table) + text[end:]
    text = text.replace("阶段 5 GRPO 已按用户授权于 18:21 启动，消融未启动。", "阶段 5 和消融均已完成。")
    text = text.replace("当前进度（以下路径相对于正式 run）：", "阶段 5 历史进度文件（以下路径相对于正式 run，已完成）：")
    text += ("\n## 最终复现与验收\n\n"
             "阶段 6 入口（已有最终验收时只核验并保留，不重训）：\n\n```powershell\n"
             "& $python scripts/run_delivery_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z\n"
             "```\n\n阶段 6 状态见 `stage6/worker_status.json`，完整验收见 `stage6/acceptance.json`；"
             "测试 XML 在 `stage6/validation/final/pytest.xml`，曲线选择依据在 `stage6/curve_provenance.json`。"
             "实验报告、两版简历描述与面试问答均位于 reports。"
             "项目已完成的范围不包含多 seed、1.5B 或跨数据集实验。"
             "必需环境、正式产物与可手动删除的测试夹具清单见验收记录；未自动清理缓存或调整系统内存设置。\n")
    write_text(readme, text)
    environment = ROOT / "reports/environment.md"
    storage = check_storage(run)
    write_text(environment, environment.read_text(encoding="utf-8") +
               "\n## 阶段 6 完成核对\n\n"
               f"记录时间 UTC：{datetime.now(timezone.utc).isoformat()}。Full Base、SFT、GRPO 与正确性奖励消融"
               f"均完成 1319 题统一评测，最终 pytest {tests['passed']} 通过、0 失败、0 跳过。"
               "正式产物仍在同一个 Full run；本阶段新增来源快照、日志、测试和最终验收位于 stage6，"
               "消融权重与断点位于 grpo_correctness_only，预测位于 evaluation/grpo_correctness_only_test。"
               f"文档更新时 run 为 {storage['run_bytes']} 字节，输出盘剩余 {storage['disk_free_bytes']} 字节。"
               "实际环境、共享缓存和产物逐项增长见 stage6/storage_final.json 与 acceptance.json。"
               "存储和内存清理按用户要求延后，本阶段未清理缓存、删除旧实验或修改系统设置。\n")
    return pairs


def delivery_checks(run, results):
    """Check links, expected evidence and Git exclusions without staging files."""
    import re
    for name in DOCUMENTS:
        text = (ROOT / name).read_text(encoding="utf-8")
        if "\ufffd" in text:
            raise ValueError(f"Replacement character in document: {name}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            target = target.split("#", 1)[0].strip("<>")
            if not (ROOT / name).parent.joinpath(target).exists():
                raise ValueError(f"Missing local document link: {name}: {target}")
    for stage in STAGES:
        if results[stage]["n"] != 1319:
            raise ValueError("Incomplete Full test denominator")
    if len(re.findall(r"^\d+\. \*\*", (ROOT / "reports/interview_qa.md").read_text(encoding="utf-8"), re.M)) < 15:
        raise ValueError("At least 15 interview questions are required")
    ignored = subprocess.run(["git", "check-ignore", "outputs/runs/full-20261107T015504132102Z/config.json",
                              ".env", "example.safetensors"], cwd=ROOT, capture_output=True, text=True, check=True)
    if len(ignored.stdout.splitlines()) != 3:
        raise ValueError("Outputs, credentials and weights must be Git ignored")
    tracked = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    patterns = (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"hf_[A-Za-z0-9]{30,}",
                r"sk-(?:proj-)?[A-Za-z0-9_-]{30,}")
    for name in tracked:
        path = ROOT / name
        if path.is_file() and path.suffix.lower() in (".py", ".md", ".yaml", ".yml", ".txt", ".json"):
            text = path.read_text(encoding="utf-8")
            if any(re.search(pattern, text) for pattern in patterns):
                raise ValueError(f"Possible secret in source document; inspect locally: {name}")
    return {"local_document_links": True, "minimum_interview_qa": True,
            "git_ignored_outputs_credentials_weights": True, "credential_pattern_scan": "passed; not an exhaustive security audit",
            "remote_operations": "none", "source_files_checked": len(tracked)}


def finish(config_path, run):
    from reasonforge.analyze import collect_results
    from reasonforge.evaluate import evaluate
    config = verify_inputs(config_path, run)
    results = collect_results(run)
    if set(results) != set(STAGES):
        raise ValueError("All four complete evaluations are required")
    evaluate(config, run, "grpo_correctness_only")
    ablation = validate_ablation(config, run, results)
    tests = run_validation(run)
    status(run, "running", phase="analysis_and_reports")
    render_analysis(run)
    pairs = update_documents(run, results, tests, ablation)
    checks = delivery_checks(run, results)
    verify_inputs(config_path, run)
    final = storage_inventory(run.parent)
    write_json(run / "stage6/storage_final.json", final)
    before = read(run / "stage6/storage_before.json")
    deltas = [{"name": r["name"], "path": r["path"], "after_bytes": r["bytes"],
               "delta_bytes": r["bytes"] - next(v["bytes"] for v in before["directories"] if v["name"] == r["name"])}
              for r in final["directories"]]
    temporary = []
    for path in sorted((run / "stage6/validation").glob("**/pytest-tmp")):
        temporary.append({"path": str(path.resolve()), "bytes": directory_bytes(path),
                          "purpose": "Completed synthetic pytest fixtures; safe to manually remove",
                          "deletion_impact": "Keep parent, XML, formal logs, predictions and weights"})
    record = {"stage": 6, "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Qwen 0.5B single-seed Base/SFT/GRPO, correctness-only ablation and final delivery",
              "results": results, "ablation": ablation, "tests": tests, "checks": checks,
              "paired_comparisons": pairs, "storage": check_storage(run), "storage_deltas": deltas,
              "temporary_directories": temporary,
              "documents": {n: file_hash(ROOT / n) for n in DOCUMENTS},
              "figures": {p.name: file_hash(p) for p in (run / "figures").glob("*.png")},
              "limitations": ["Single seed and GSM8K only", "RL budget 256 steps; not all 7345 training questions visited",
                              "No observed Full accuracy gain from GRPO over SFT", "Driver changed between SFT and GRPO",
                              "Failed initial SFT invocation lacks complete time/peak measurements"],
              "not_run": ["multiple seeds", "independent 1.5B model", "cross-dataset generalization"],
              "resume_usable": True, "storage_cleanup": "Deferred by user; no files deleted or system settings changed"}
    write_json(run / "stage6/acceptance.json", record)
    status(run, "completed", acceptance="stage6/acceptance.json", project_scope_complete=True)
    print("DONE Stage 6; final experiment evidence, reports and tests verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUTF8="1")
    configure_runtime()
    run, config = args.run_dir.resolve(), args.config.resolve()
    if not (run / "stage6/preflight_tests.json").exists() or not (run / "stage6/storage_before.json").exists():
        raise ValueError("Stage 6 preflight tests, storage baseline and provenance are required")
    with FileLock(run / ".delivery_stage.lock", timeout=0):
        try:
            verify_inputs(config, run)
            if (run / "stage6/acceptance.json").exists():
                saved = read(run / "stage6/acceptance.json")
                if any(file_hash(ROOT / n) != h for n, h in saved["documents"].items()):
                    raise ValueError("Delivered documents changed; inspect before repeating")
                print("Stage 6 acceptance already exists; preserve it", flush=True)
                return
            run_batches(config, run, args.resume_checkpoint)
            finish(config, run)
        except BaseException as error:
            status(run, "failed", error_type=type(error).__name__,
                   message=str(error).replace(str(Path.home()), "<user-home>"),
                   recovery="Inspect stage6 logs. No automatic failure retry or cleanup; preserve existing evidence.")
            raise


if __name__ == "__main__":
    main()
