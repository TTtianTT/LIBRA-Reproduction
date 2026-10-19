# LIBRA：仅 Simulation 复现

复现 **LIBRA: An Economical Hybrid Approach for Cloud Applications with Strict SLAs**（IC2E 2021）的 Section IV，覆盖图 3–6 的仿真实验。本项目不部署云资源，不需要 AWS 账号、凭证或 boto3。

已完成并实际运行：LIBRA / FaaS / MAX / SPOCK / AUTO 五种策略、官方 WITS 12 小时 trace、phi=-1…4 敏感性实验、Berkeley 和两种官方合成 trace。保留作者源代码，并提供独立的历史实现重放。

本实现属于**基于论文公式、官方参数及 trace 的独立重建**。官方代码不完整，且论文未公开所有实验细节，因此不能声称逐数值复现原图。[方法与差异说明](docs/methodology.md)记录了全部关键选择。

## 直接看结果

- [PPO 固定预算实验：RL-1–4](outputs/rl_study/REPORT.md)：85 次真实训练、5 种子、843 条测试与泛化/消融；当前结果不支持 PPO 稳定优于强分位数规则。
- [RL-0：DP 成本下限与 Gymnasium 环境](outputs/rl0/REPORT.md)：v1 精确 DP、60 秒启动状态、一致性校验与 v2 排队敏感性；尚未训练 PPO。
- [Azure Functions 2021 两周独立验证](outputs/ml_azure2021/REPORT.md)：前 7 天训练、后 7 天测试，原始/64 倍负载对照；本次 GBDT 未超过强规则。
- [LIBRA-ML 改进报告](outputs/ml/REPORT.md)：用学到的 τ=0.797 需求分位数替代 avg+φ·std，跨 trace 与时间切分共 6 折评估。
- [补丁后 E1–E3 重跑报告](outputs/review/REPORT.md)：完整基线扫描、phi/rho 联合扫描和路由/EWMA 变体，共 1,263 次仿真。
- [P0–P2 验证报告](outputs/validation/REPORT.md)：MAX 区间/价格审计、EWMA 单变量消融、AUTO 三项修复的全部八种组合。
- [WITS 主实验报告](outputs/wits_12h/REPORT.md)：完整表格和图 3–6。
- [结构化结果](outputs/wits_12h/results.json) / [汇总 CSV](outputs/wits_12h/summary.csv)。
- [作者代码对照审计](docs/upstream_audit.md)。
- [Berkeley](outputs/berkeley_12h/REPORT.md)、[正弦 trace](outputs/synthetic_sin/REPORT.md)、[方波 trace](outputs/synthetic_square/REPORT.md)。

主实验：WITS 前 43,200 秒，6,217,439 个请求，默认 phi=1。

| 策略 | 总成本 USD | SLA 违约 % |
|---|---:|---:|
| LIBRA | 22.2951 | 0 |
| FaaS | 53.0658 | 0 |
| MAX | 87.6000 | 0 |
| SPOCK | 25.8673 | 1.4560 |
| AUTO | 26.0483 | 4.1377 |

LIBRA 相比 AUTO 节省约 14.41%，相比 SPOCK 节省约 13.81%。此次 phi 扫描的最优值是 **2**，与论文报告的 **1** 不同。Berkeley 和合成 trace 上 LIBRA 并非总比其他策略便宜，详见对应实际输出；没有为匹配论文结论调整参数。

后续验证发现：仅切换 EWMA deviation 更新顺序，两组整数扫描最优值都仍为 2。AUTO 的 $33.924→约 $26 成本差异主要来自利用率窗口修正，同时 SLA 违约会增加；缩容与计费两项修复在主 trace 上无影响。MAX 的论文图差异仍未完全解释，保留实际峰值对应的 73 台，不调整实例数。

## 安装与运行

需要 Python 3.10+。模拟器核心只依赖 Python 标准库；画图需要 NumPy 和 Matplotlib。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python scripts/reproduce.py
.venv/bin/python -m pytest -q
```

当前工作区已配置 `.venv`，可直接执行后两条命令。主实验一次运行包括五种策略和六个 phi 值；仿真时间是虚拟时间，不需要等待真实 12 小时。

已验证环境为 Python 3.12.7、NumPy 1.26.4、Matplotlib 3.9.2、pytest 7.4.4，精确依赖版本见 `requirements-reproduce.txt`。也可先执行 `.venv/bin/python -m pip install -r requirements-reproduce.txt` 固定这些版本。

```bash
# 快速运行；仍使用真实 trace
.venv/bin/python scripts/reproduce.py --seconds 1800 --output outputs/smoke

# 作者默认是 45,000 秒；论文描述为 12 小时，默认采用 43,200 秒
.venv/bin/python scripts/reproduce.py --seconds 45000 --output outputs/wits_45000

# Berkeley：12 小时
.venv/bin/python scripts/reproduce.py --trace data/traces/berkeley_logs.txt \
  --output outputs/berkeley_12h --no-plots

# 合成 trace 保持作者给出的长度，不填充、不循环
.venv/bin/python scripts/reproduce.py --trace data/traces/sin_logs.txt \
  --seconds 10000 --output outputs/synthetic_sin --no-plots
.venv/bin/python scripts/reproduce.py --trace data/traces/square_load.txt \
  --seconds 1112 --output outputs/synthetic_square --no-plots

# 校验 trace 哈希、请求守恒、成本及已生成的主实验图表
.venv/bin/python scripts/verify_artifacts.py
```

自定义配置：复制 [configs/paper.json](configs/paper.json)，修改后传入 `--config path.json`。自定义 trace 每行是该秒请求数，必须是非负整数；支持 `--offset` 选取片段。读取越界会直接报错。当前实现刻意限定论文的 1 秒 epoch 和 1 秒服务时间。

## 输出内容

每个实验目录包含 `results.json`、`summary.csv`、`phi_sweep.csv`、五种策略逐秒 `*_timeline.csv`、缩放决策 `*_decisions.json` 和 `REPORT.md`。绘图开启时产生图 3–6 的 PNG、SVG。

`results.json` 记录 trace SHA-256、实际配置、仿真核心源码哈希、Python 版本、运行命令及响应时间分布。运行确定性，无随机采样；运行时长和生成时间等元数据随运行变化。

## ML 扩展（改进部分）

`libra_ml/` 是可选扩展，模拟器核心不依赖它。`simulate(..., estimator=f)` 在每次扩缩容决策时用 `f(t, counts[:t+1])` 替代 `avg + φ·std`；默认 `estimator=None` 时结果与原复现逐位一致。由价格可推出最优配置目标是下一窗口每秒需求的 0.797 分位数，因此 ML 部分是一个分位数回归问题（线性 QR、GBDT QR）。方法、结果和局限见 [outputs/ml/REPORT.md](outputs/ml/REPORT.md)。

```bash
.venv/bin/python -m pip install -e '.[test,ml]'   # 精确版本见 requirements-ml.txt
.venv/bin/python scripts/ml_experiments.py        # 约 2–3 分钟
.venv/bin/python scripts/plot_ml.py
```

两周 Azure 2021 数据已在本地下载。`data/azure2021/aggregate_counts.npz` 是未缩放的逐秒聚合，`manifest.json` 记录官方来源、授权归属和校验值。原始 RAR/CSV 位于 `data/azure2021/raw/`，由 Git 忽略。专页时间单位为秒，到达时间使用 `end_timestamp-duration`；不是将毫秒直接当秒处理。

```bash
.venv/bin/python scripts/prepare_azure_2021.py --download  # 下载需 curl，解压需 unrar
.venv/bin/python scripts/azure_ml_experiments.py
.venv/bin/python scripts/summarize_azure_ml.py
```

Azure 对照保持固定服务时间，不重放实际 duration；64 倍场景是明确标注的调用量复制。日周期特征收益、原始负载低于 CIP 的影响及负结果都保留在报告中。`outputs/review/run_metadata.json` 保持旧 E1–E3 运行的历史指纹，本次没有仅重写该文件或用旧 CSV 冒充新代码实验。

## PPO 扩展（RL-1–4）

第 1 步的 v1/v2 环境与 DP 已完成，并继续运行了 PPO 主实验、基线调参、泛化和消融。主实验采用明确的 Azure 16 倍调用量复制；1–9 天训练、10–11 天验证、12–14 天冻结测试，另外评估原始 Azure、WITS、Berkeley 和 Sin。每个配置 5 个随机种子，所有实际样本数、训练耗时、checkpoint 选择和 95% 区间均保留。

这是预先固定本地预算的一轮实验，不能声称已收敛或具有严格等计算调参预算。0.5 秒服务下 PPO 满足 SLA，但没有可靠超过强规则；0.7 秒下所选 PPO 的测试均值违约率超过 0.1%。完整方法、AUTO/SPOCK 适配边界与历史测试集暴露见 [运行说明](docs/rl_study.md)和[实际报告](outputs/rl_study/REPORT.md)。

```bash
.venv/bin/python -m pip install -r requirements-rl.txt
.venv/bin/python -m pip install -e '.[test,rl]'
.venv/bin/python scripts/rl_study.py --stage all
.venv/bin/python scripts/summarize_rl_study.py
.venv/bin/python scripts/verify_rl_study.py --require-models
```

本地可用 `scripts/rl_study_parallel.py --workers 3` 并行运行独立种子。模型权重由 Git 忽略；新 clone 应传 `--output outputs/rl_study_replay` 使用新目录重训，汇总和校验同样传该目录，避免覆盖已发布的历史记录。

## RL 诊断：动作范围、物理模型与训练预算

在既有 PPO 研究之后，新增同动作域 DP、全 FaaS 动作、教师规则先验、独立连续时间排队校准及两档训练预算诊断。新实验只使用训练和验证分区，原测试结果不重新当作盲测。运行协议见 [诊断说明](docs/rl_diagnostics.md)，实际测量见 [诊断报告](outputs/rl_diagnostics/REPORT.md)。

```bash
.venv/bin/python scripts/rl_diagnostics.py --stage all --workers 4
.venv/bin/python scripts/summarize_rl_diagnostics.py
.venv/bin/python scripts/verify_rl_diagnostics.py --require-models
```

## 作者代码重放

官方仓库在论文参考文献 [27] 中给出。固定 commit：`53a804412ca24c6eca7fa226fcccc0220b17fb0f`。

```bash
# 仅在缺少 upstream/ 时需要联网；下载研究代码，不执行其中 aws/ 目录
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python scripts/replay_upstream.py

# P0/P1/P2 全部控制实验（需要上述固定 upstream checkout）
.venv/bin/python scripts/validate_reproduction.py
```

此脚本只做 Python 2→3 语法适配，保留原有模型和 bug。公开代码未实现 SPOCK，因此历史重放只报告其实际存在的四种策略，也不自动补加 Gateway 成本。细节和历史结果见[审计说明](docs/upstream_audit.md)。

控制实验通过 `libra_validation/` 在原作者 AUTO engine 上独立开关利用率窗口、缩容清理和计费修复；不会修改 `upstream/` 原文件。`--part p0/p1/p2` 可单独运行一项。论文柱形读数和 PDF 哈希保存在 `references/paper_evidence.json`；重新提取读数的可选脚本是 `scripts/extract_paper_evidence.py`，需安装 PyMuPDF。常规仿真与验证无需此依赖。

## 来源

- [论文 arXiv:2104.05491v1](https://arxiv.org/abs/2104.05491)，本复现核对版本；本地 PDF 在 `references/libra-arxiv.pdf`。
- [作者官方代码与 trace](https://github.com/aliraza0337/LIBRA)。
- [trace 校验清单](data/manifest.json)。trace 是作者仓库的原始副本，保留其来源；不对上游代码、数据或论文另行声明许可。
