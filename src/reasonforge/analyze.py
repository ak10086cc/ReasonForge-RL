"""Generate figures and a local report exclusively from verified run artifacts."""
import json
from pathlib import Path
from .metrics import summarize, extract_answer
from .utils import read_jsonl, write_json, file_hash, digest


def collect_results(run):
    """Read and validate finished evaluations; never rewrite the source run."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    results = {}
    contracts = set()
    for stage in ("base", "sft", "grpo", "grpo_correctness_only"):
        directory = run / "evaluation" / f"{stage}_test"
        if not (directory / "metrics.json").exists():
            continue
        saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        prediction_file = directory / "predictions.jsonl"
        rows = read_jsonl(prediction_file)
        expected = read_jsonl(run / "data/test.jsonl")
        if [r["question_id"] for r in rows] != [r["question_id"] for r in expected]:
            raise ValueError("Evaluation is not the complete ordered test split")
        for row, source in zip(rows, expected):
            extracted = extract_answer(row["prediction"])
            if (row["ground_truth"] != source["ground_truth"] or row["question"] != source["question"]
                    or row["stage"] != stage or row["seed"] != config["seed"]
                    or row["split"] != "test" or row["extracted_answer"] != extracted
                    or row["parse_success"] != (extracted is not None)
                    or row["correct"] != (extracted == source["ground_truth"])):
                raise ValueError("Prediction fields disagree with source data or parser")
        recomputed = summarize(rows)
        if saved["prediction_sha256"] != file_hash(prediction_file):
            raise ValueError("Predictions changed since evaluation")
        if any(saved[key] != value for key, value in recomputed.items()):
            raise ValueError("Reported metrics disagree with predictions")
        contract = json.loads((run / "evaluation_contract.json").read_text(encoding="utf-8"))
        if saved["evaluation_contract_hash"] != digest(contract):
            raise ValueError("Saved evaluation contract hash differs")
        if contract["test_hash"] != file_hash(run / "data/test.jsonl"):
            raise ValueError("Test data differs from the frozen contract")
        if stage != "base":
            from safetensors.torch import load_file
            from .modeling import state_hash
            summary = json.loads((run / stage / "summary.json").read_text(encoding="utf-8"))
            weights = run / stage / "final/adapter_model.safetensors"
            actual = state_hash({k: v.float() for k, v in load_file(str(weights)).items()})
            if not saved["adapter_reload_verified"] or actual != summary["weight_update"]["after_hash"]:
                raise ValueError("Evaluation adapter failed verification")
            if stage.startswith("grpo"):
                parent = run / "sft/final/adapter_model.safetensors"
                sft_summary = json.loads((run / "sft/summary.json").read_text(encoding="utf-8"))
                start = summary["sft_start_hash"] if "sft_start_hash" in summary else summary["weight_update"]["before_hash"]
                if (summary["parent_adapter_sha256"] != file_hash(parent)
                        or start != sft_summary["weight_update"]["after_hash"]):
                    raise ValueError("GRPO parent differs from the recorded SFT weights")
        results[stage] = saved
        contracts.add(saved["evaluation_contract_hash"])
    if len(contracts) > 1:
        raise ValueError("Stages use different evaluation contracts")
    return results


def analyze_run(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    results = collect_results(run)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    plots = []
    if results:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        names = list(results)
        x = list(range(len(names)))
        accuracy_bars = ax.bar([v - .18 for v in x], [results[n]["accuracy"] for n in names], .36, label="Exact match")
        parse_bars = ax.bar([v + .18 for v in x], [results[n]["parse_rate"] for n in names], .36, label="Parse rate")
        ax.bar_label(accuracy_bars, fmt="%.3f", padding=3, fontsize=8)
        ax.bar_label(parse_bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(x, names, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{config['name']} / n={next(iter(results.values()))['n']} (single seed)")
        ax.legend()
        fig.savefig(figures / "stage_comparison.png", dpi=160)
        plt.close(fig)
        plots.append("stage_comparison.png")
    for stage, key, filename in (("sft", "loss", "sft_loss.png"), ("grpo", "reward", "grpo_reward.png")):
        path = run / stage / "train_log.jsonl"
        if not path.exists():
            continue
        logs = [row for row in read_jsonl(path) if key in row]
        if not logs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot([r["step"] for r in logs], [r[key] for r in logs], marker="o", markersize=3)
        ax.set(xlabel="Optimizer step", ylabel=key, title=f"{config['name']} / {stage}")
        ax.grid(alpha=.25)
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
        plots.append(filename)
    lines = [f"# {config['name']} 实际运行结果", "",
             "Smoke 仅用于流程验证。Small 为单 seed 小样本探索；两者均不能支持稳定提升结论。", "",
             "| Stage | n | Accuracy | Parse rate | Mean tokens | Eval seconds |", "|---|---:|---:|---:|---:|---:|"]
    for stage in ("base", "sft", "grpo", "grpo_correctness_only"):
        m = results.get(stage)
        lines.append(f"| {stage} | {m['n']} | {m['accuracy']:.4f} | {m['parse_rate']:.4f} | {m['mean_generated_tokens']:.1f} | {m['resources']['seconds']:.1f} |"
                     if m else f"| {stage} | 未运行 | — | — | — | — |")
    lines.extend(["", "配置、依赖、revision、源代码哈希见 config.json 和 provenance.json；数据成员与哈希见 data/manifest.json。",
                  "逐题预测和错误案例见 evaluation/<stage>_test/；权重变化、显存和训练耗时见 <stage>/summary.json。", ""])
    for stage in ("sft", "grpo", "grpo_correctness_only"):
        path = run / stage / "summary.json"
        if path.exists():
            s = json.loads(path.read_text(encoding="utf-8"))
            lines.extend([f"{stage} 训练：{s['resources']['seconds']:.2f} 秒，峰值分配显存 {s['resources']['peak_allocated_mib']} MiB；"
                          f"adapter 最大参数变化 {s['weight_update']['max_abs_delta']}。", ""])
    if (run / "grpo" / "rollout_audit.jsonl").exists():
        audits = read_jsonl(run / "grpo" / "rollout_audit.jsonl")
        zero = sum(r["zero_variance_group_fraction"] for r in audits) / len(audits)
        lines.extend([f"GRPO 零方差组占比（各 rollout 批次等权平均）：{zero:.4f}；"
                      "零方差组没有相对优势信号，需要结合实际参数变化判断。", ""])
    for filename in plots:
        lines.extend([f"![{filename}](figures/{filename})", ""])
    lines.extend(["## 错误案例", ""])
    for stage in results:
        errors = read_jsonl(run / "evaluation" / f"{stage}_test" / "errors.jsonl")
        for error in errors[:2]:
            lines.extend([f"- {stage} / {error['question_id']}：真值 {error['ground_truth']}；提取 {error['extracted_answer']}；"
                          f"parse_success={error['parse_success']}。", f"  模型原文：{error['prediction']!r}"])
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run / "verified_results.json", results)
    return results
