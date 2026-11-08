# ReasonForge-RL

面向 GSM8K 小学数学题的 Qwen 后训练实验：相同的 `Qwen/Qwen2.5-0.5B-Instruct` 基座，依次执行 Base、LoRA + SFT、从 SFT adapter 继续 GRPO，再使用冻结配置统一评测。项目采用 TRL 与 PEFT，没有实现新的强化学习算法。

## 实测结果与项目范围

本地已完成真实 Smoke、Small，以及 0.5B 单种子的完整测试集评测。项目由 Codex 辅助实现；训练、预测与指标均来自本机运行。

| 方法 | 正确数 / 测试题数 | 严格准确率 | 可解析率 |
|---|---:|---:|---:|
| Base | 12 / 1319 | 0.91% | 1.90% |
| LoRA + SFT | 422 / 1319 | 31.99% | 99.17% |
| SFT + GRPO | 422 / 1319 | 31.99% | 99.17% |
| 仅正确性奖励消融 | 437 / 1319 | 33.13% | 99.39% |

![Full 单种子实验的准确率与可解析率](reports/figures/full_stage_comparison.png)

本次 GRPO 未超过 SFT；消融的单次差异不能证明稳定增益。Base 低分受严格末行答案格式影响，不应将表中差异全部解释为数学能力提升。Full 指 7345/128/1319 的训练候选池、验证集和完整官方测试集，GRPO 与消融各执行 256 步，未遍历全部训练题。多种子、1.5B、跨数据集实验未运行。

已实现数据隔离、SFT loss mask、SFT→GRPO 权重衔接、正确性/格式奖励、统一评测、断点恢复和结果审计。最终本地产物验收为 **111 通过、0 失败、0 跳过**。训练设备为 RTX 3050 Laptop 4 GB；恢复调用的耗时与显存统计范围、驱动差异及负结果见 [实验报告](reports/experiment_report.md)。

## 获取代码与公开内容

```powershell
git clone https://github.com/ak10086cc/ReasonForge-RL.git
cd ReasonForge-RL
```

公开内容包括源码、配置、测试、方案、筛选后的报告和展示图。模型、adapter、checkpoint、数据缓存、逐题预测、原始训练日志与个人简历保留在本地。文中已有 run_id、验收 JSON 与 pytest XML 路径用于定位本机历史证据，不随克隆分发；重新运行会生成自己的独立 run。首次本地实验时尚无 Git commit，历史记录中的 unavailable 状态保留，不回填本次发布提交作为当时版本。

下方环境准备与 Smoke 命令可从干净检出开始运行。首次模型下载需要网络，真实训练需要兼容 CUDA GPU；普通 pytest 无历史产物时会跳过额外的产物验收。分阶段后台脚本需要原始 run 的快照与启动前测试记录，属于该次本地实验的恢复入口；新实验请使用通用的 run_pipeline.py，并将 --run-dir 替换为自己生成的目录。

## 环境准备

本机验证环境为 Windows 11、Python 3.12.10、RTX 3050 Laptop 4 GB；PyTorch 使用官方 CUDA 12.8 wheel。完整检测结果见 [reports/environment.md](reports/environment.md)。Python 默认命令可能指向其他版本，请在专用环境中执行下列命令。

Windows PowerShell 首次安装：

```powershell
py -3.12 -m venv "$env:USERPROFILE/.cache/reasonforge/venv"
$python = "$env:USERPROFILE/.cache/reasonforge/venv/Scripts/python.exe"
& $python -m pip install -r requirements.txt
& $python -m pip check
& $python scripts/run_pipeline.py --phase env
& $python -m pytest -q
```

Linux CUDA 环境可用 `python3.12 -m venv ~/.cache/reasonforge/venv` 建环境，再使用该环境的 `bin/python`。本次仅验证 Windows，未声称验证 Linux。

首次运行从 Hugging Face 下载公开模型和 `openai/gsm8k` 的 `main` 配置，缓存位于框架默认的仓库外目录。无需付费 API 或 Token。模型与数据 revision 已在配置中固定。默认禁用上报服务与 Hub 上传；不会自动创建远程仓库。

## 运行实验

```powershell
& $python scripts/run_pipeline.py --config configs/smoke.yaml
& $python scripts/run_pipeline.py --config configs/small.yaml
```

Smoke 使用 2 条训练、2 条验证、2 道测试题，SFT/GRPO 各 1 步，仅验证流程。Small 使用 128 条候选训练数据、16 条验证、32 道测试题，SFT 32 步，GRPO 16 步，并从同一 SFT 权重独立运行正确性奖励消融。SFT 会剔除超出序列长度的完整解答并记录 ID，不截断最终答案；实际保留数量见每次运行的 `sft/summary.json`。

Full 配置使用全部训练候选、128 条验证和全部官方测试题。当前采用下文的分阶段安排，先验证资源，再启动 Full；不要直接运行默认的 `--phase all`。

每次运行创建独立的 `outputs/runs/<name>-<timestamp>/`，不会覆盖此前运行。输出的 `RUN_DIR` 是后续命令的实际参数。每个阶段都有独立入口，例如：

```powershell
& $python scripts/run_pipeline.py --config configs/small.yaml --phase prepare
# 将以下路径替换为上一条命令输出的 RUN_DIR。
$run = 'outputs/runs/<实际运行目录>'
& $python scripts/run_pipeline.py --config configs/small.yaml --phase baseline --run-dir $run
& $python scripts/run_pipeline.py --config configs/small.yaml --phase sft --run-dir $run
& $python scripts/run_pipeline.py --config configs/small.yaml --phase grpo --run-dir $run
& $python scripts/run_pipeline.py --config configs/small.yaml --phase ablation --run-dir $run
& $python scripts/run_pipeline.py --config configs/small.yaml --phase evaluate --run-dir $run
& $python scripts/run_pipeline.py --config configs/small.yaml --phase analyze --run-dir $run
```

`--resume-checkpoint` 接收 Trainer 的 checkpoint 目录，恢复权重、optimizer、scheduler 和 RNG。最终 adapter 只用于推理或开始下一阶段，不能代替含优化器状态的断点。已有完整输出的阶段会拒绝覆盖。失败时先检查 `failures.jsonl` 和最后的 checkpoint，再按报告说明恢复；不要手动覆盖已有实验。

恢复同一运行中未完成的训练阶段，例如：

```powershell
& $python scripts/run_pipeline.py --config configs/small.yaml --phase sft --run-dir $run --resume-checkpoint "$run/sft/checkpoint-16"
& $python scripts/run_pipeline.py --config configs/small.yaml --phase grpo --run-dir $run --resume-checkpoint "$run/grpo/checkpoint-8"
```

这些是训练故障恢复示例，已有最终模型的训练会拒绝恢复；checkpoint 必须保留原运行配置，包含优化器/RNG 状态，且早于配置的最后一步。不能通过恢复覆盖更晚的 checkpoint。SFT 连续训练一致性和 GRPO 根目录 adapter/ref 加载分支有实际回归测试。若训练已经完成但评测尚未完成，重复该阶段命令并省略 `--resume-checkpoint`，会复用最终模型续接评测，不重新训练。

## 分阶段执行与停止位置

总技术依据始终为 [项目实施方案](docs/项目实施方案.md)，操作规则为 [AGENTS.md](AGENTS.md)。下表仅安排执行顺序，不替代方案或授权下一阶段。分次运行不改变同一实验的模型、数据划分、seed、训练预算或冻结的评测条件；不得根据 test 结果调参、挑选模型或决定是否保留结果。

| 阶段 | 输入与执行范围 | 验收条件 | 停止位置 |
|---|---|---|---|
| 1．运行准备 | 当前代码和已有 Smoke/Small；修复恢复与报告、存储检查、测试 | 续接与一次完成结果一致；拒绝变更输入；旧产物不变；报告按真实状态生成 | 交付测试和占用变化后停止，不运行 Full |
| 2．资源验证 | 现有 0.5B 环境；用 Full 长度处理少量训练/验证样本 | 记录真实显存、耗时、磁盘增长和 Full 估算；不使用 test 调参 | 保存探测结果后停止；探测不包装为 Full |
| 3．Base | 经资源验证确认的配置，正式 run 的数据和划分清单 | 完成验证预检、冻结评测契约、1319 题完整基线 | 保存预测、指标与契约后停止 |
| 4．SFT | 同一 run、配置、基座与训练/验证划分 | 完成训练及统一测试；loss mask、参数变化、保存重载可核验 | 保存最终 adapter、日志、断点与评测后停止 |
| 5．GRPO | 阶段 4 保存的同一 SFT adapter | 初始策略/参考策略与 SFT 一致；完成训练与相同条件评测 | 保存模型、奖励审计和指标后停止 |
| 6．消融与交付 | 同一 SFT 起点、可用资源和前三阶段证据 | 资源允许时独立运行正确性奖励消融；随后报告、曲线、简历、完整测试及验收；未运行项明确标注 | 消融可单独一次运行；最终交付后停止 |

阶段 1–6 已完成本次既定 0.5B 实验与交付。正式 run：`outputs/runs/full-20261107T015504132102Z/`；最终证据为 `stage6/acceptance.json`。各历史阶段的启动、失败和恢复记录保留在下文，当前状态以阶段 6 验收为准。

| Full 阶段 | 正确数 / 1319 | 严格准确率 | 可解析率 |
|---|---:|---:|---:|
| base | 12 | 0.91% | 1.90% |
| sft | 422 | 31.99% | 99.17% |
| grpo | 422 | 31.99% | 99.17% |
| grpo_correctness_only | 437 | 33.13% | 99.39% |

本轮最终 pytest：**111 通过、0 失败、0 跳过**。SFT 与 GRPO 同为 422/1319，本次没有观察到 GRPO 准确率增益。Full 指完整数据池与测试规模，RL 为 256 步预算；单 seed、严格格式和驱动版本差异等限制见实验报告。

历史阶段证据与资源依据：
阶段 2 的记录位于 `outputs/runs/stage2-resource-probe-20261107T012552232627Z/`：`probe_summary.json` 保存常规探测耗时，`length_stress/resources.json` 保存长度边界测试，`resource_estimate.json` 保存估算依据，`validation/acceptance.json` 保存验收与占用变化，`validation/pytest.xml` 保存测试结果。RTX 3050 Laptop 4 GiB 上，SFT/GRPO 常规训练峰值分配显存为 1491.72/1563.63 MiB；最长训练 prompt（248 token）配合 4 个 512-token 回答的 GRPO 单步压力测试峰值分配/保留为 1776.96/1914 MiB，未出现 OOM。SFT 实际最长样本为 481 token，上限仍为 768；没有用填充样本冒充实际训练长度。

资源探测只从官方训练集取 8 条训练、4 条验证样本，沿用既有划分哈希；SFT/GRPO 仅运行 4/8 步，三组各做 4 条验证推理，不读取测试题。压力测试额外强制最小生成长度为 512，仅用于资源诊断，奖励为零，不计入质量结果。短跑外推的核心 Full 时间约 **17.3 小时**，规划预留约 **9–35 小时**，多数用于逐题推理；这是显式采用 0.5–2 倍系数的粗估，不是统计区间或最长耗时承诺。真实 Full 模型生成长度、设备温度、其他程序和故障恢复都会影响耗时。新增正式 run 暂按 0.5–1 GiB 预留，消融另计，原有环境与共享缓存不重复计算。

资源探测可在项目根目录复现（每次创建独立探测目录，不恢复或覆盖旧探测）：

```powershell
& $python scripts/resource_probe.py --config configs/full.yaml --split-run outputs/runs/small-20261106T032826964295Z
$probe = 'outputs/runs/<上一步实际输出的RUN_DIR名称>'
& $python scripts/resource_probe.py --stress-run $probe
```

阶段 3 使用独立本地进程：先完成 128 条验证题，再冻结契约并评测 1319 条测试题。运行器只调用 `--phase baseline`，每小时的计划暂停会自动续接；错误或没有进展则停止，不自动重试失败，不启动 SFT。电脑需保持接电、不休眠；关机、手动休眠或结束进程会中断执行。逐题结果持续保存，未完成前不生成最终测试指标。完成后自动核验完整有序预测、重算指标、契约、源码和数据哈希，保存 `stage3/acceptance.json` 后退出。

阶段 3 已结束，下面是其复现入口；已完成验收时只读核验并保留原记录，不重新推理。不要在阶段 4 开始后再次启动 Base 运行器：

```powershell
& $python scripts/run_baseline_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z
```

该运行器需要原 run 的数据、源码快照和 `stage3/preflight_tests.json`。启动前测试 XML 位于 `stage3/validation/pytest.xml`，同目录的 `pytest-tmp/` 是已结束测试的可删除夹具；数据、正式预测、日志、评测契约和验收记录应保留。运行期间不修改推理代码、依赖、配置和数据。

阶段 4 于 2026-11-07 13:44 首次启动、15:42 从断点恢复、17:52 完成验收，沿用该正式 run 和冻结契约，SFT 使用 7345 条训练、128 条验证数据，最多 920 步，之后评测全部 1319 道测试题。启动前测试为 **89 通过、0 失败、0 跳过**（19.19 秒），XML 位于 `stage4/validation/pytest.xml`。独立运行器只调用 `--phase sft`；训练每 100 步保存断点，最多保留两个中间断点和最终 adapter。评测每小时的计划暂停自动续接，复用已完成的最终模型；错误或无进展则停止，不自动重试失败，不进入 GRPO。启动前测试和新增入口源码快照位于 `stage4/`，原 run 的源码快照、provenance 和评测契约保留不变。

```powershell
& $python scripts/run_sft_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z
```

仅在进程已停止并查明原因后重复运行。若训练中断，先核实最后完整 checkpoint，再传 `--resume-checkpoint`；评测中断且已有最终模型时省略该参数。评测时间预算不限制训练时间，电脑需保持接电、不休眠。进度查看：

- `stage4/worker_status.json`：整个阶段的运行、失败或完成状态，以及当前日志位置。
- `sft/train_log.jsonl`：已完成训练步、训练与验证 loss（920 步为本阶段预算）。
- `evaluation/sft_test/progress.json`：测试进度，全部完成后才产生 `metrics.json`。
- `stage4/acceptance.json`：完整训练、loss mask、参数变化、保存重载、全部预测及指标通过核验后才生成，状态应为 `complete`。

根目录 `status.json` 在长训练期间可能仍显示上一阶段完成或上次失败，判断当前运行请查看 `stage4/worker_status.json` 和训练日志。`stage4/validation/pytest-tmp/` 仅保存测试夹具，测试结束后可手动删除；保留同目录的 XML、正式日志、模型与验收记录。阶段 4 已完成；其入口仅供追溯，不在 GRPO 启动后重新执行。当前进度请看下方阶段 5 记录。

本次正式 SFT 首次运行在 2026-11-07 14:12 报 `CUDA error: unknown error`，最后训练日志为第 506 步，未进入测试评测；具体原因未确定，未发现 OOM 记录。核对 GPU 运算、检查点模型参数、优化器和调度器后，已于 15:42 按用户授权从 `sft/checkpoint-500` 恢复。模型、数据、训练预算和冻结评测协议未改变。原错误保留在 `failures.jsonl` 与 `stage4/sft-0001.log`，恢复证据和实际命令保存在 `stage4/recovery.json`，本次执行日志为 `stage4/sft-0002.log`；后续计划暂停会使用递增日志编号。训练总日志追加保留原记录，501–506 步可能出现重跑记录，不应当作额外训练预算或将重复步骤直接平均。最终 `sft/summary.json` 的资源计量只覆盖成功恢复的那次训练调用，不包含首次失败的完整耗时或峰值；不得将其写成全部尝试的资源总量。

阶段 5 于 2026-11-07 18:21 启动、20:58 完成验收：从同一 `sft/final` 继续训练 256 步，每题 4 个回答，生成上限 512，正确性/格式权重 2.0/0.2。启动前本轮 pytest **97 通过、0 失败、0 跳过**（19.70 秒）。源码、依赖、数据、SFT 权重与冻结评测协议已核验；原训练和推理实现保持不变。GPU 驱动本轮检测为 621.14，与 Base/SFT 历史记录的 610.60 不同；并非本轮修改，PyTorch CUDA wheel 仍为 12.8，GPU 运算检查通过，该环境差异保留为复现限制。

```powershell
& $python scripts/run_grpo_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z
```

该入口独立后台执行，只调用 GRPO；训练每 64 步保存断点，最多两个中间断点和最终 adapter。训练完成后评测相同 1319 题，评测每小时的计划暂停自动续接；错误或无进展则停止。完成后核对 SFT 衔接、实际参数变化、训练集 rollout 奖励、逐题预测和汇总指标，写入 `stage5/acceptance.json` 后退出，不运行消融或后续交付阶段。电脑需接电、不休眠；聊天结束后本地进程可继续，关机或休眠会中断。只有完整评测完成后才有正式 GRPO 指标。

阶段 5 历史进度（已完成，以下路径相对于正式 run）：

- `stage5/worker_status.json`：当前阶段状态、日志位置；优先查看此文件，根 `status.json` 可能仍显示 SFT 完成。
- `grpo/train_log.jsonl`：已完成的训练步，目标 256 步。
- `grpo/rollout_audit.jsonl`：实际奖励、正确性、格式率和零方差组诊断。
- `evaluation/grpo_test/progress.json`：训练完成后的测试进度，目标 1319 题。
- `stage5/acceptance.json`：仅在整个阶段验证通过后生成，不能把旧阶段验收当作当前阶段完成。

阶段 5 日志、检查点和预测分别存于本 run 的 `stage5/`、`grpo/`、`evaluation/grpo_test/`，预计新增约 0.1–0.3 GiB，不含系统分页文件，仍受 5 GiB/剩余 50 GiB 保护。复用现有环境与仓库外模型/数据缓存。`stage5/validation/pytest-tmp/` 为测试结束后可手动删除的夹具（912735 字节，约 0.87 MiB）；保留父目录 XML、正式模型和日志。实际目录清单及增量见 `stage5/storage_before.json`、`storage_launch.json` 与最终验收记录。

阶段 6 已于 2026-11-07 获授权：从相同 SFT final 训练正确性单项奖励消融，256 步、每题 4 个回答，其余配置与冻结评测协议不变；评测相同 1319 题。之后自动核验奖励、数据和权重衔接，生成真实曲线、报告、两版简历描述与至少 15 组面试问答，并运行 Smoke/Small/Full 产物的最终 pytest。仅分析和绘图处理 SFT 恢复后的重复日志，训练/推理源码、依赖与配置保持不变。

```powershell
& $python scripts/run_delivery_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z
```

启动前全套 pytest 为 **107 通过、0 失败、0 跳过**（20.06 秒），XML 位于 `stage6/validation/release/pytest.xml`。另一次补充测试因未创建临时父目录产生 2 通过、8 项准备错误，原记录保留在 `stage6/validation/delivery-check/pytest.xml`；修正目录准备后重跑全部测试，未改变训练实现。当前阶段状态只看 `stage6/worker_status.json`；训练日志为 `grpo_correctness_only/train_log.jsonl`，奖励记录为同目录 rollout_audit.jsonl，测试进度为 `evaluation/grpo_correctness_only_test/progress.json`。只有完成全部交付和验收后才生成 `stage6/acceptance.json`。旧 stage3/4/5 的 complete 不代表阶段 6 完成。

该入口在本地后台继续，计划暂停自动续接，失败或无进展停止，不无限重试。训练完成后的分析、最终测试和报告生成已获本轮授权，会自动执行；期间请勿手动修改源码、配置、数据或交付文档，运行器检测到变化会保护文件并停止。保持电脑接电、不休眠。产物仍在同一 run，预计新增约 0.1–0.3 GiB，保留 5 GiB/剩余 50 GiB 保护；按用户要求，内存和磁盘清理留待项目完成后处理。

每次续接先核对：本次授权阶段 → 当前规范与方案 → run 的 `config.json`、`data/manifest.json`、`evaluation_contract.json` → 实际模型、日志与预测 → 剩余空间及预算。CLI 在训练前核验前置评测：SFT 需要 Base 完成，GRPO 需要 Base/SFT 完成，消融需要核心三阶段完成；缺失或证据不一致则拒绝启动。每次结束汇报完成内容、测试结果、保存位置、新增占用、未完成项和下一阶段；技术路线或关键验收条件变化先说明并确认。

### 评测分批运行

在阶段 2 完成且阶段 3 获授权后，从项目根目录创建正式 run 并保存其 `RUN_DIR`：

```powershell
& $python scripts/run_pipeline.py --config configs/full.yaml --phase prepare
$run = 'outputs/runs/<上一步实际输出的RUN_DIR名称>'
& $python scripts/run_pipeline.py --config configs/full.yaml --phase baseline --run-dir $run --eval-max-seconds 3600
# 下次重复上一条 baseline 命令，自动跳过已核验完成的题目与验证结果。
```

`--eval-max-seconds` 是本次命令的评测时间预算（秒），不改变生成参数；达到预算后完成当前题目、保存并退出。核验、模型加载及单题生成不能在任意时刻截断，因此这是软时间上限。它不限制训练时长。也可用 `--eval-max-samples 50` 限制每个评测目标本次新增题数，完整评测分母不变。SFT、GRPO、消融阶段同样可加评测预算。

暂停时 `status.json` 为 `paused`，保留 `predictions.jsonl`、`progress.json` 与 `resume_manifest.json`，不会生成最终 `metrics.json`；全部题目完成后才汇总。新评测续接校验数据顺序、配置、模型 adapter、评测代码与关键依赖，防止重复计数或混入不同实验。新实验的评测契约还固定跨阶段的推理代码、设备类型、精度和关键依赖；旧版完整产物保留原契约，不回写新条件。已有完成结果只读核验，不覆盖。

训练仍从最后一个包含 optimizer/scheduler/RNG 的完整 checkpoint 恢复，检查点之后未保存的步骤可能需要重跑。计划暂停应等本次程序退出后再关机；运行时保持接电、不休眠。强制结束或断电造成的残缺 JSONL 会报错并原样保留，不擅自截断修复；旧版本未完成且没有续接清单的评测也不会自动续接。CLI 对单个 run 加进程锁，同时启动第二个阶段会拒绝执行；正常退出或进程结束后操作系统释放锁。

### 存储位置与预算

实际绝对路径和逻辑占用可用以下只读命令检查，不创建实验、不下载内容：

```powershell
& $python scripts/run_pipeline.py --config configs/full.yaml --phase storage
```

| 内容 | 本机默认位置 | 保留方式 |
|---|---|---|
| 必需 Python 环境 | 用户目录下 `.cache/reasonforge/venv` | 复用，不作为临时缓存删除 |
| 模型和数据缓存 | 用户目录下 `.cache/huggingface/`；实际值以框架环境变量为准 | 共享复用，不能全部归因于本项目 |
| pip 下载缓存 | `pip cache dir` 返回的位置 | 共享缓存，不自动清理 |
| 安装等系统临时内容 | `TEMP/TMP` 指向的目录 | 不扫描整个临时目录，不推断项目占用 |
| 训练日志、预测、最终模型与断点 | 本项目 `outputs/runs/<run_id>/` | 每次实验独立目录，禁止覆盖其他运行 |

新命令默认每个 run 上限 **5 GiB**、输出盘至少剩余 **50 GiB**，记录在该 run 的 `storage_policy.json`。如需不同预算，在首次运行与后续命令中一致传入 `--max-run-gib`、`--min-free-gib`；不会静默提高限制。阶段开始、每个评测题、训练优化步骤及 checkpoint 保存后检查占用；达到保护线报错停止，保留证据与已有断点，不自动清理。检查是阶段性保护，单次模型保存或题目仍可能跨过上限，不能视作操作系统硬配额。

每阶段最多保留两个中间训练 checkpoint，最终 adapter 另存。预算按单个 run 计算，旧实验及共享缓存额外占空间；开始、结束仍须核对总体增量。统计为逻辑文件大小，不等于文件系统实际分配块；重叠目录不重复汇总。不会修改系统缓存设置或擅自迁移文件。

公开模型、数据下载完成后，可在当前 PowerShell 设置以下变量复用本地缓存：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
```

## 数据、训练与评测约定

- 仅从官方 train 按 seed=42 拆出 128 条验证数据；官方 test 不参与训练、验证损失、奖励或超参数选择。规范化问题后去重，跨集合重复题从训练侧剔除，保留审计记录。
- 三阶段共用原生 chat template 和 system prompt。SFT 使用 prompt-completion，`completion_only_loss=True`，并检查 TRL 实际生成的标签和 collator，确认 prompt 标签全部为 `-100`。
- GRPO prompt 仅含问题与格式要求，标准答案作为独立奖励字段。加载已训练 SFT adapter，核对参数哈希，再交给 `GRPOTrainer`，不重复传入 LoRA 配置。KL 参考 adapter 是 SFT 策略的冻结副本。
- LoRA 初始 `r=16, alpha=32, dropout=0.05`，目标为注意力的 q/k/v/o 投影。低显存配置使用单条 microbatch、梯度累积、bf16 和非重入 gradient checkpointing，不启用 vLLM。
- GRPO 每题生成 4 个回答，采用 `loss_type='grpo'`、组内奖励标准化、正确性/格式权重 2.0/0.2、KL 系数 0.04。消融只去掉格式奖励；基座、SFT 起点、seed、预算、生成与评测参数保持一致。
- 最后一行必须为 `Final Answer: <number>`。不从推理中的中间数字猜答案。支持负数、小数和正确的千位分隔符；百分号按 GSM8K 标量标签的“百分点”解释，例如 `25%` 与标签 `25` 对齐，不把它转换成 `0.25`。不支持分数或带单位的末行。
- Base 先运行固定验证题进行预检，随后冻结评测契约；test 使用 greedy decoding。三阶段必须具有相同的模型 revision、tokenizer、prompt、parser、生成长度和测试文件哈希。解析失败按错误计入总题数。

## 运行记录与结果

实际实验结论见 [reports/experiment_report.md](reports/experiment_report.md)。Smoke 结果不能代表模型能力；Small 单 seed 小样本结果不能证明稳定提升。

每个运行目录保留：

| 文件 | 用途 |
|---|---|
| `config.json`, `provenance.json` | 配置、命令、依赖、Git 状态、源文件哈希 |
| `data/manifest.json` | 数据 revision、划分成员、样本数量和哈希 |
| `evaluation_contract.json` | 冻结后的统一评测条件 |
| `evaluation/<stage>_test/` | 逐题预测、汇总指标、错误案例 |
| `evaluation/<stage>_<split>/progress.json`, `resume_manifest.json` | 分批进度、调用资源记录与不可变续接条件 |
| `storage_policy.json` | 本 run 的产物上限和输出盘剩余空间保护线 |
| `<stage>/train_log.jsonl`, `summary.json` | 真实训练日志、耗时、峰值显存和参数变化 |
| `grpo/rollout_audit.jsonl` | 奖励均值/方差、正确性、格式率、零方差组比例 |
| `<stage>/checkpoint-*`, `final/` | 断点及最终 adapter/tokenizer |
| `report.md`, `figures/`, `verified_results.json` | 从预测重算后的结果、曲线与对比图 |
| `failures.jsonl`, `status.json` | 错误类型、失败命令、最后成功阶段与恢复提示 |

`outputs/`、模型权重、数据缓存、虚拟环境和凭据均被 Git 忽略。不要上传逐题数据或模型，也不要将 `.env` 加入版本控制。

完成实验后，可将实际运行目录传入验收与报告命令：

```powershell
& $python -m pytest tests -q --artifact-run $run --junitxml "$run/validation/pytest.xml"
& $python scripts/make_report.py --runs $run --pytest-xml "$run/validation/pytest.xml"
```

报告生成只读输入实验；需要更新曲线时先执行获授权 run 的 `--phase analyze`。预览报告可传 `--output-dir 'outputs/runs/<本次验收目录>/report-preview'`，避免覆盖 `reports/` 的现有材料。报告按实际提供的规模、完成状态、配置与资源记录生成，未完成评测不填指标；没有提供某规模的证据不等于断言该规模在别处未运行。

`--artifact-run` 可重复传入，以同时核对 Smoke 和 Small。它额外检查数据哈希、相同测试题、指标分母、checkpoint 权重与 SFT→GRPO 衔接。没有实际运行产物时，普通单测会跳过这一项产物验收，不伪造训练结果。

## 局限

GSM8K 是单一英文数学数据集，不能代表跨领域推理能力；公开模型可能在预训练时接触过相关数据。严格格式解析会将语义正确但格式不合规的输出判错；生成截断也会降低可解析率。报告同时给出可解析率、完成率、生成长度和零方差奖励诊断，便于区分这些因素。

简历用语见 [reports/resume_bullets.md](reports/resume_bullets.md)，面试问答见 [reports/interview_qa.md](reports/interview_qa.md)。只使用报告中已验证的事实，不写未经验证的准确率提升。

API 依据：[TRL SFT](https://huggingface.co/docs/trl/sft_trainer)、[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)、[PEFT 集成](https://huggingface.co/docs/trl/peft_integration)、[GSM8K 数据集](https://huggingface.co/datasets/openai/gsm8k)。项目代码按已安装版本的实际接口验证。

阶段 6 首次消融于 2026-11-07 21:57 报 CUDA out of memory，最后训练日志 202/256 步，第 192 步检查点完整。失败进程已退出，当前显存已释放；已核验检查点权重有限、冻结参考与 SFT 一致、优化器及调度器均为 192，GPU 运算通过。具体诱因未确定，采用新进程从 checkpoint-192 控制恢复，模型、数据、训练预算、超参数和冻结评测协议不变。原失败记录和日志保留，恢复证据见正式 run 的 stage6/recovery.json。成功分支应为前 192 个回答组加恢复后的 64 个组；失败尝试末尾 11 个组保留但不混入正式奖励统计。恢复后 summary 仅计当前调用的资源，不冒充全部训练耗时或全程峰值。

## 最终复现与验收

阶段 6 历史入口（需原始本地 run 及原始交付文档；新检出使用前文通用 CLI）：

```powershell
& $python scripts/run_delivery_stage.py --config configs/full.yaml --run-dir outputs/runs/full-20261107T015504132102Z
```

发布整理已更新 README 和环境报告，原始交付文档保存在本机该 run 的 `stage6/prepublication_documents.zip`，变更哈希见 `stage6/publication.json`。原验收保持不变，直接对更新后的文档重跑上述一次性交付入口会因文档哈希保护而拒绝；不应为此覆盖原验收或重新训练。

阶段 6 状态见 `stage6/worker_status.json`，完整验收见 `stage6/acceptance.json`；测试 XML 在 `stage6/validation/final/pytest.xml`，曲线选择依据在 `stage6/curve_provenance.json`。实验报告、两版简历描述与面试问答均位于 reports。项目已完成的范围不包含多 seed、1.5B 或跨数据集实验。必需环境、正式产物与可手动删除的测试夹具清单见验收记录；未自动清理缓存或调整系统内存设置。
