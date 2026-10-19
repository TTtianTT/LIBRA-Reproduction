# 本地 PPO 研究协议与运行方法

第 1 步的环境和原始成本下限见 [RL-0](rl0.md)。第 2–3 步的固定协议在 [configs/rl_study.json](../configs/rl_study.json)，训练入口是 [scripts/rl_study.py](../scripts/rl_study.py)。实际结果、统计、学习曲线、Pareto 图和限制由 [实验报告](../outputs/rl_study/REPORT.md)给出。

## 本次固定范围

| 项目 | 设置 |
|---|---|
| 数据切分 | Azure 1–9 天训练、10–11 天验证、12–14 天最终测试 |
| 主要需求 | 原始聚合调用量复制 16 倍，非实际观测到达率 |
| 泛化 | Azure 原始 1 倍，以及 WITS、Berkeley、Sin |
| 回合 | 随机训练 6 小时；评估不重叠的至多 6 小时回合，各自空资源/空历史 |
| 主决策间隔 | 60 秒，冷启动 100 秒 |
| 主延迟 | Erlang-C 等待尾 + 固定执行时间，服务 0.5/0.7 秒 |
| 敏感性 | 完整 M/M/16 响应尾，冻结策略后重放，不重新调参 |
| 算法 | stable-baselines3 PPO，CPU，64×64 MLP，γ=0.99 |
| v1 | 8,192 决策样本 × 5 种子，仅 λ=0 |
| v2 主实验 | 16,384 决策样本 × 5 种子 × 3 λ × 2 服务时间 |
| 消融 | 无增强、隐藏预测、原始增减动作、120/300 秒；两服务时间，各 5 种子 |
| λ | 0 / 0.0001 / 0.01 USD 每违约请求 |
| checkpoint | 25/50/75/100% 样本处，更新后在完整验证分区评估 |
| 基线 | 各族 12 训练候选；LIBRA、分位数、GBDT、AUTO-adapted、SPOCK-adapted |

全部 85 次 PPO 拟合，共 1,351,680 个训练决策样本。预算固定，没有以测试结果为理由延长训练或修改 λ。预算是否足以收敛应由后续独立研究验证，当前不预设结论。

## 安装及分阶段运行

```bash
.venv/bin/python -m pip install -r requirements-rl.txt
.venv/bin/python -m pip install -e '.[test,rl]'

# 全流程，已有完成记录可按指纹恢复；最终测试批次不会重复覆盖
.venv/bin/python scripts/rl_study.py --stage all
.venv/bin/python scripts/summarize_rl_study.py
.venv/bin/python scripts/verify_rl_study.py --require-models
```

也可分别运行 `--stage train`、`--stage baselines`、`--stage final`。训练/基线扫描可独立进行；final 必须在 85 个训练记录和基线选择完成后执行。`--stage pilot` 用独立试跑目录，不参与正式统计。

本地也可运行 `scripts/rl_study_parallel.py --workers 3`，让独立种子用三个进程调用同一训练函数；每进程 CPU 单线程，样本预算和选择顺序不变。该入口另存源码指纹及实际并发墙钟时间，最后仍由原入口完成冻结和评估。

Git 不保存本地 `models/` 内的 SB3 ZIP 和 GBDT joblib。新 clone 请指定新结果目录，以免覆盖历史记录：

```bash
.venv/bin/python scripts/rl_study.py --output outputs/rl_study_replay --stage all
.venv/bin/python scripts/summarize_rl_study.py --output outputs/rl_study_replay
.venv/bin/python scripts/verify_rl_study.py --output outputs/rl_study_replay --require-models
```

ZIP 保存包含运行信息，跨平台/重新训练的字节哈希不要求相同。已发布的结构化结果可以在无权重时校验；缺少本地权重时不会假装能够恢复旧训练，必须使用新目录。正式实验之前曾修复 NumPy 整数写出问题，旧尝试保留于本地忽略目录 `outputs/rl_study_attempt1`，未混入正式结果。

## 防止调参和统计混淆

GBDT 的特征、标签和固定参数只使用 1–9 天；训练增强只在训练分区内定义新负载。PPO 使用固定的有符号 log 观测变换，没有用测试统计量归一化。每个种子独立初始化网络、环境 RNG 和训练增强。

每个 λ 的 checkpoint 只按完整验证分区的 `cost + λ*expected_violations` 选择。消融所用的 λ 根据主实验五种子的验证均值选取：有可行点时选最便宜者，否则选违约最低者。`test_freeze.json` 在读取最终测试分区之前保存所有权重指纹、λ 和基线参数。恢复被中断的最终批次时只补未完成的冻结项。

基线的参数按训练分区目标选取，测试时固定。AUTO/SPOCK 的 12 组阈值、组大小、窗口和最小实例来自 E1 网格的预先确定抽样。每族候选上限为 12；PPO 每种子也有 3×4 个 λ/checkpoint 候选。它们并不具有相同训练计算量或验证查询次数，报告据实列账，不将其写成严格等计算预算。

GBDT 继承 100 秒 lead、300 秒预测窗口，固定超参数，三个服务设定各按相应报童分位数拟合；未另外调预测窗口匹配 60 秒时钟，也未进行端到端决策损失训练。策略/预测器的计算开销不计入 VM/FaaS 模型成本，训练时间另报。

五种子区间使用 Student-t，反映固定数据上 PPO 训练随机性，不代表所有未来 trace 的不确定性。确定性基线无虚构种子区间。区间跨越差异时不写“RL 更好”；不满足 epsilon 的低成本点不被当作严格 SLA 下的胜出。

## 原论文及 RL-0 的边界

原 `libra_sim`、WITS 成本模型及 P0–P2 不修改。论文 LIBRA 基线仍使用 300 秒时钟与 estimate 路由，其他主策略使用 60 秒，因此图表需要同时披露时钟差异。AUTO/SPOCK 在本研究中是目标控制器适配，使用共用 FaaS 溢出路由和解析 SLA，没有原有 FIFO 积压，不能声称是原策略的逐数值复现。

v2 的每秒准稳态近似不产生实际队列积压；主要跨时间效应仍是启动及取消。完整指数服务模型与固定执行假设的尾部差异可能彻底改变 SLA 可行性。全 FaaS 是零违约参照，但残差 ±3 动作未必能把很高的规则目标直接清零。

本研究评估分成空资源的 6 小时回合，DP 也按相同回合求和，不能拿 RL-0 连续 12 小时数字直接当这些回合的下限。`fast_dp.py` 用前缀/区间最小值将 K<D 时钟的更新从 O(M³) 降为 O(M²)，测试对照原始轨迹 DP 及穷举。v1 有全局成本下限；v2 只在明确的 VM/rho 域上优化拉格朗日目标。v2 的成本字段并非 epsilon 约束最优成本，连续路由或域外策略也不受该有限域界约束。

Azure 12–14 天在先前 ML 报告中已被查看。这里冻结的是新的 PPO 评估协议，不能重新宣称它是项目历史上从未暴露的测试集。长期外推仍需要新的独立 trace。
