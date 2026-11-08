"""Compile reports from verified artifacts without modifying the source runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonforge.analyze import collect_results
from reasonforge.utils import read_jsonl

STAGES = ("base", "sft", "grpo", "grpo_correctness_only")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    return "未记录" if value is None else f"{value:.2f}"


def link(path, output):
    return Path(os.path.relpath(Path(path).resolve(), output.resolve())).as_posix()


def build_reports(runs, output, pytest_xml=None):
    collected = [(Path(r), read_json(Path(r) / "config.json"), collect_results(r)) for r in runs]
    if not collected:
        raise ValueError("At least one run is required")
    names = [c["name"] for _, c, _ in collected]
    if len(set(names)) != len(names):
        raise ValueError("Select one run per scale to keep report rows and figure names unambiguous")
    output = Path(output)
    if any(output.resolve().is_relative_to(r.resolve()) for r, _, _ in collected):
        raise ValueError("Report output must be outside the source experiment runs")
    lines = ["# ReasonForge-RL 实验报告", "", "## 结论与边界", "",
             "仅汇总下列输入运行的可核验产物；未提供不等于在其他目录从未运行。"
             "Smoke 仅验证流程；Small 是小样本探索。单次实验不能证明稳定增益。"
             "严格格式解析与生成截断都会影响准确率。", "", "## 实际结果", "",
             "| 规模 | 阶段 | n | 正确数 | Exact Match | 可解析率 | 完成率 | 平均生成 token | 评测秒数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run, config, results in collected:
        for stage in STAGES:
            m = results.get(stage)
            if m:
                lines.append(f"| {config['name']} | {stage} | {m['n']} | {m['correct']} | {m['accuracy']:.2%} | "
                             f"{m['parse_rate']:.2%} | {m['completion_rate']:.2%} | {m['mean_generated_tokens']:.2f} | {m['resources']['seconds']:.2f} |")
            else:
                directory = run / "evaluation" / f"{stage}_test"
                status = "未完成（不计指标）" if directory.exists() else "未运行"
                lines.append(f"| {config['name']} | {stage} | {status} | — | — | — | — | — | — |")
    missing = [name for name in ("smoke", "small", "full") if name not in names]
    lines.extend(["", "本报告未提供的规模：" + ("、".join(missing) if missing else "无") + "。", "",
                  "## 实验设置与资源", ""])
    for run, config, results in collected:
        provenance = read_json(run / "provenance.json")
        hardware = provenance.get("hardware", {})
        counts = {s: len(read_jsonl(run / f"data/{s}.jsonl")) for s in ("train", "validation", "test")}
        lines.extend([f"### {config['name']} / {run.name}", "",
                      f"模型 {config['model']['id']}，revision {config['model']['revision']}；"
                      f"GSM8K main revision {config['data']['revision']}；seed={config['seed']}。",
                      f"实际候选 train/validation/test={counts['train']}/{counts['validation']}/{counts['test']}；"
                      f"数据划分证据：{link(run / 'data/manifest.json', output)}。",
                      f"设备：{hardware.get('gpu_name') or '未记录 GPU / CPU'}；总显存 MiB：{number(hardware.get('gpu_total_mib'))}；"
                      f"torch CUDA runtime：{hardware.get('torch_cuda_runtime', '未记录')}。",
                      "依赖：" + "、".join(f"{k}={v}" for k, v in provenance.get("dependencies", {}).items()) + "。",
                      "实际配置：" + json.dumps({k: config[k] for k in ("lora", "sft", "grpo", "evaluation")}, ensure_ascii=False) + "。", ""])
        if all(stage in results for stage in ("sft", "grpo")):
            difference = results["grpo"]["correct"] - results["sft"]["correct"]
            lines.extend([f"GRPO 相比 SFT 正确数差为 {difference} 题，分母 {results['grpo']['n']}；"
                          "不能据此认定因果增益或稳定提升。", ""])
    lines.extend(["显存为当前进程的峰值分配/保留值，不等于整卡占用。分批评测耗时按已记录调用累计；"
                  "强制结束进程可能缺失部分耗时，具体口径见 metrics.json 的 resource_scope。", "",
                  "| 规模 | 阶段 | 进入 Trainer 样本 | 已执行步数 | 训练与验证秒数 | 峰值分配 MiB | 峰值保留 MiB | 最大参数变化 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for run, config, results in collected:
        for stage in STAGES[1:]:
            path = run / stage / "summary.json"
            if not path.exists():
                continue
            s = read_json(path)
            resource = s["resources"]
            log_file = run / stage / "train_log.jsonl"
            logs = read_jsonl(log_file) if log_file.exists() else []
            actual_steps = max((row["step"] for row in logs), default="未记录")
            lines.append(f"| {config['name']} | {stage} | {s['train_samples']} | {actual_steps} | "
                         f"{resource['seconds']:.2f} | {number(resource['peak_allocated_mib'])} | "
                         f"{number(resource['peak_reserved_mib'])} | {s['weight_update']['max_abs_delta']:.8g} |")
    lines.extend(["", "SFT 训练计时包含阶段内验证；模型加载和最终保存不在此列。"
                  "GRPO 包含 rollout 与优化。恢复训练的 summary 可能仅计当前调用，须核对 resource_scope。", "",
                  "## 奖励诊断与错误分析", ""])
    for run, config, results in collected:
        lines.extend([f"### {config['name']}", ""])
        for stage in STAGES[2:]:
            path = run / stage / "rollout_audit.jsonl"
            rows = read_jsonl(path) if path.exists() else []
            if rows:
                mean = lambda k: sum(r[k] for r in rows) / len(rows)
                lines.append(f"- {stage}：{len(rows)} 个 rollout 批次，{sum(r['n'] for r in rows)} 个生成回答；"
                             f"批次等权 reward 均值={mean('reward_mean'):.6f}、方差均值={mean('reward_variance'):.6f}；"
                             f"正确性={mean('correctness_rate'):.2%}、格式率={mean('format_rate'):.2%}、"
                             f"零方差组比例={mean('zero_variance_group_fraction'):.2%}。")
        for stage in results:
            rows = read_jsonl(run / "evaluation" / f"{stage}_test" / "predictions.jsonl")
            truncated = sum(not r["parse_success"] and not r["reached_eos"] for r in rows)
            format_failed = sum(not r["parse_success"] and r["reached_eos"] for r in rows)
            wrong = sum(r["parse_success"] and not r["correct"] for r in rows)
            lines.append(f"- {stage}：未解析且未结束 {truncated} 题；已结束但格式不合规 {format_failed} 题；数值错误 {wrong} 题。")
            for row in [r for r in rows if not r["correct"]][:2]:
                excerpt = row["prediction"].replace("\n", " ")[-220:]
                lines.extend(["", f"案例 {stage}/{row['question_id']}：真值 {row['ground_truth']}，提取 {row['extracted_answer']}。", "", f"> {excerpt}"])
        lines.append("")
    lines.extend(["零方差组没有相对奖励信号；权重变化也可能来自 KL 或数值差异，须结合 reward、KL、clip 和截断诊断。", "",
                  "## 曲线", ""])
    preferred = max(enumerate(collected), key=lambda pair: (("smoke", "small", "full").index(pair[1][1]["name"])
                    if pair[1][1]["name"] in ("smoke", "small", "full") else -1, pair[0]))[1]
    run, config, results = preferred
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("sft_loss.png", "grpo_reward.png", "stage_comparison.png"):
        source = run / "figures" / filename
        if source.exists():
            target = output / "figures" / f"{config['name']}_{filename}"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            lines.extend([f"![{config['name']} {filename}](figures/{target.name})", ""])
    lines.extend(["## 可追溯证据", ""])
    for r, c, _ in collected:
        provenance = read_json(r / "provenance.json")
        lines.extend([f"- {c['name']}：{link(r, output)}；实际命令：{json.dumps(provenance.get('command'), ensure_ascii=False)}；"
                      f"Git commit：{provenance.get('git_commit', '未记录')}。"])
    lines.extend(["", "## 测试", ""])
    statement = "最终测试数量未导入"
    if pytest_xml:
        root = ET.parse(pytest_xml).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        total = sum(int(s.get("tests", 0)) for s in suites)
        failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
        skipped = sum(int(s.get("skipped", 0)) for s in suites)
        statement = f"{total-failures-skipped} 通过、{failures} 失败、{skipped} 跳过"
        lines.append(f"导入 pytest 记录：{statement}；原始 XML：{link(pytest_xml, output)}。历史记录不等于本轮重新测试。")
    else:
        lines.append("未提供 pytest XML，不能据本报告声称最终测试通过。")
    full = next((m for _, c, m in collected if c["name"] == "full"), None)
    full_complete = full is not None and all(s in full for s in STAGES[:3])
    lines.extend(["", "## 限制与未完成项", "",
                  "Full 核心三阶段已有完整评测产物。" if full_complete else
                  "Full 核心三阶段尚无完整产物纳入本报告；不填补或推断缺失指标。",
                  "多 seed、独立 1.5B 等实验状态以显式提供的运行证据为准；本报告不推断。",
                  "GSM8K 单数据集存在泛化局限，公开模型可能存在预训练污染。"
                  "所有调参仅用训练/验证集；测试结果不得用于选择参数或挑选有利运行。", ""])
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    result_statement = "；".join(f"{s} {m['correct']}/{m['n']}（{m['accuracy']:.2%}）" for s, m in results.items()) or "无已完成评测"
    resume = ["# ReasonForge-RL 简历项目描述", "", "## 短版", "",
              "**ReasonForge-RL：Qwen 数学推理后训练与评测**", "",
              "- 基于 PyTorch、Transformers、TRL 和 PEFT，构建 GSM8K 数据隔离、LoRA/SFT、从 SFT 继续 GRPO、统一评测与奖励消融流程。",
              f"- 已核验规模：{config['name']}，seed={config['seed']}；{result_statement}。测试记录：{statement}。", "",
              "## 详版", "",
              f"**ReasonForge-RL｜{config['model']['id']} 后训练实验**", "",
              "- 固定模型与数据 revision，训练/验证/测试隔离，保存逐题预测、配置和权重证据。",
              "- 核对 SFT prompt loss mask、adapter 保存重载和训练断点恢复；GRPO 从相同 SFT 策略继续，诊断正确性/格式奖励与零方差组。",
              "- 使用冻结评测条件比较 Base、SFT 和 GRPO，核验原始预测与汇总指标。",
              f"- 当前证据：{config['name']}，{result_statement}。", "",
              "## 使用边界", "",
              "以上为运行证据，不代表本人已独立掌握所有实现；需结合代码和面试材料学习解释。"
              "Smoke 不代表正式效果，Small 不证明稳定提升；不得宣称新算法、因果增益或未核验的训练成果。",
              "完整数据规模核心评测已有证据。" if full_complete else "完整规模核心评测证据尚不齐全。",
              "具体设置、消融状态及局限见 [experiment_report.md](experiment_report.md)。", ""]
    (output / "resume_bullets.md").write_text("\n".join(resume), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--pytest-xml", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Use a separate directory to preview reports")
    args = parser.parse_args()
    build_reports(args.runs, args.output_dir, args.pytest_xml)
    print(f"Wrote {args.output_dir}/experiment_report.md from verified artifacts")


if __name__ == "__main__":
    main()
