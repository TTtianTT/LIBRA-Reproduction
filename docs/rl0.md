# RL-0：先验证上限与环境

RL-0 提供可回放、可校验的环境和已知未来的成本下限。本次没有训练 PPO，也没有证据表明 RL 胜过规则。实际数字见 [运行报告](../outputs/rl0/REPORT.md)，原论文复现仍使用 `libra_sim`，没有改写其计费或历史结果。

```bash
.venv/bin/python -m pip install -r requirements-rl.txt
.venv/bin/python -m pip install -e '.[test,rl]'
.venv/bin/python scripts/rl0_experiments.py --with-forecast
.venv/bin/python scripts/verify_rl0.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_artifacts.py
```

未准备 Azure 时可以先运行 `scripts/prepare_azure_2021.py --download`；也可传 `--skip-azure`，其结果会明确标记跳过。去掉 `--with-forecast` 可只运行规则、随机策略和 DP，状态中的预测值及可用性标记均为零。

## 环境及计费契约

`libra_rl.env.LibraEnv` 使用 [Gymnasium Env 接口](https://gymnasium.farama.org/introduction/create_custom_env/)，测试通过官方 [check_env](https://gymnasium.farama.org/api/utils/)。默认 60 秒决策、随机截取最多 6 小时、空初始资源；默认种子通过 `reset(seed=...)` 控制。有限数据窗口结束时返回 `truncated=True`，随后必须 reset。验证/测试调用者必须关闭增强，并显式选定起止点。

```python
import numpy as np
from libra_rl.env import LibraEnv
from libra_rl.model import CostModel

env = LibraEnv(counts, model=CostModel(version="v2", service_seconds=0.7))
observation, info = env.reset(seed=0)
# 残差 a=0，路由 rho=0.8。这里只演示接口，不是训练结果。
observation, reward, terminated, truncated, info = env.step(np.array([3, 2]))
```

每个决策先观察当前秒请求量，再选择目标 VM 数，与原仿真顺序一致。创建即计费，100 秒后就绪；缩容删除最新创建的 VM，保留最老的 VM。终点按实际窗口截断计费，不为终点后未发生的运行收费。新环境逐秒向量化请求路由与延迟计算；没有真实请求或 AWS 调用。

原论文 LIBRA 的 `rho × estimate` 路由会在扩容启动结束时更新。`step_target(..., routing="estimate", estimate=...)` 专门回放这一行为，要求决策间隔大于启动时间。默认 RL 接口则填充就绪容量，允许多个启动批次跨决策重叠。

v1 使用原有 1 秒固定服务、每 VM 16 req/s、按秒 VM 计费、FaaS 每请求 $0.000008535、Gateway $0.0116/h。固定论文 LIBRA 与 300 秒窗口分位数规则在 WITS/Berkeley、120/300 秒决策上逐项核对 VM/FaaS/Gateway/总成本，绝对容差 1e-9。原 `Config` 仍拒绝小于等于冷启动的论文决策间隔；新环境将自己的决策间隔独立配置，没有放宽原模型或造成旧结果哈希失效。

## 状态、动作、奖励

状态按 `env.observation_names` 解释：

- `libra_ml` 的 21 个因果特征；每回合最多保留起点前 1,800 秒的同一数据分区历史，EWMA 在该历史起点初始化。
- 一个 GBDT 归一化分位数预测及 `has_forecast`。默认无模型，二者为零；传入 `forecast_model` 后批量预测。模型接口要求与 `feature_matrix` 的均值尺度目标一致。
- 就绪、启动中、总分配容量（除以历史需求尺度），上一决策段利用率与违约率，距上次扩容时间、回合剩余比例、当前 λ。
- 100 个启动剩余秒数桶，每桶实例数转换成相同的需求尺度。启动总数本身不足以判断下一段有多少 VM 就绪，这组桶保留资源延迟状态。

这些特征保证观测只读取当前及过去需求。资源状态完整，并不意味着真实工作负载在压缩后的 21 个特征上满足马尔可夫性；仍应将其视为部分可观测的负载过程。

v1 动作 `Discrete(7)`，索引 0–6 对应 a=-3…3，目标为 `clip(规则目标+a, 0, max_vms)`。v2 动作 `MultiDiscrete([7,5])`，第二维为 rho=0.6/0.7/0.8/0.9/1.0。`action_mode="raw"` 改为在当前已分配 VM 总数上直接加 -8/-4/-2/-1/0/1/2/4/8；不预热实例。

默认规则使用最近 300 秒需求的报童分位数，τ=1−VM每请求位秒成本/FaaS每请求成本，CIP 保留论文的 4 req/s。v1 参考 rho=1；v2 参考 rho=0.8，目标为 `ceil(q/(throughput*rule_rho))`。这是未调参起点，不声称为含排队损失时的最优分位数。路由动作的 rho 不会隐式改变这个参考目标。`rule_uses_forecast=True` 可切换成 GBDT 预测目标；需要传入已拟合模型。

奖励精确定义为：

```
reward = -(VM成本 + FaaS成本 + λ*(预期违约数 - epsilon*请求数)) / cost_normalizer
```

λ 的单位是 USD/违约请求，默认 0；`cost_normalizer` 默认固定为 1 USD，不用测试数据计算归一化常数。Gateway 是与动作无关的常数，仅记录于总成本。λ 是可观测值，通过 `env.set_penalty(value)` 更新；RL-0 没有实现 PPO 优化器或自动对偶更新。后续可按验证协议固定扫描 λ，或在训练回合统计上作投影更新 `λ←max(0,λ+η*(violation_rate-epsilon))`，不可用测试集调节 λ。

`starting_vm_seconds` 是全部启动开销；`cancelled_startup_vm_seconds` 只计取消尚未就绪实例已经支付的秒数；`unfinished_startup_vm_seconds` 单独报告终点截断的启动开销。不能把所有启动时间直接写成反复扩缩造成的浪费。

## v2 延迟模型：明确近似及敏感性

每 VM 有 c=16 个请求位，平均服务时间 s=0.5 或 0.7 秒。就绪吞吐为 c/s；每秒均匀分配 VM 请求，实际利用率 u=VM请求数/(就绪VM数×c/s)。剩余请求走 FaaS。FaaS 按相同 s 收取 GB-second 费用，调用费用不变；其执行时间固定，无队列或冷启动，s≤SLA 时无违约。此处同时改变吞吐与价格；不能把 v1/v2 的成本差直接归因于 RL。

[M/M/c 等待模型及 Erlang-C](https://web.mit.edu/urban_or_book/www/book/chapter4/4.6.1.html)给出等待概率 C(u,c)；条件等待时间为指数分布，速率 k=c(1−u)/s。本项目由这些分布推导两种响应时间假设：

1. `wait_budget`（默认）：执行时间固定为 s，等待时间使用 M/M/c 近似。若 S≥s，违约概率为 `C*exp(-k*(S-s))`。这是提议中概率表使用的公式，是混合近似，不是 M/M/c 的完整响应分布，也不是精确 M/D/c。
2. `mmc_response`：服务时间为独立 Exp(μ)，μ=1/s。完整响应尾概率为 `(1-C)*exp(-μS)+C*(k*exp(-μS)-μ*exp(-kS))/(k-μ)`。当 k=μ 时使用连续极限 `exp(-μS)*(1+μS)`。零负载时尾概率仍趋近 `exp(-S/s)`；不能在同一严格 SLA 假设下忽略它。

u≥1 时稳态不存在，保守地设违约概率为 1。空 VM 没有请求，因此不会凭空产生违约。所有违约数为预期值，可为小数；它们不是离散事件仿真的实测违约数。

两个模式都按每秒需求使用准稳态近似，每台 VM 的队列独立且负载均衡，没有实现等待积压、缩容时队列迁移、瞬态排队或服务时间随机样本。队列状态没有跨秒反馈，真实的序贯作用仍主要来自启动延迟与资源保留。若这仍使决策接近单步优化，应先测新的 DP 差距，不能仅凭加入延迟公式就断言 RL 会有优势。

## 已知未来 DP 的状态和边界

入口 `libra_rl.dp.solve_dp`；返回目标序列、rho 序列、成本、预期违约与目标函数值。默认从零实例开始，给定整个未来需求，保留最老 VM。它优化该模型、该决策时钟与路由网格中的全时域目标，不是只看一个窗口的 oracle。

令决策间隔 K、启动时间 D=100，进入决策时共有 n 台旧 VM，动作目标 m。

- K≥D：所有旧 VM 已就绪。前 D 秒就绪数 `min(n,m)`，之后 m；下一状态 m。
- D/2≤K<D：状态 `(n,r)`，r 台旧 VM 已就绪，其余旧 VM 在 D−K 秒后就绪。前 D−K 秒就绪数 `min(r,m)`，之后 `min(n,m)`；新启动实例本段不会就绪。下一状态为 `(m,min(n,m))`。

每段按实际剩余长度截断，对所有动作计算 `m*VM秒价*段长度 + FaaS成本 + λ*预期违约 + 下一状态价值`，反向取最小值。最终重建动作序列，再用独立逐秒启动时钟回放检查目标值。小型四决策问题穷举所有 VM/rho 序列，与 DP 在 60/120/300 秒时钟上相符。

目前精确 DP 要求 `K≥ceil(D/2)`，否则明确拒绝；不能用于任意更短时钟。v1 填满容量时自动使用 `ceil(peak/throughput)` 作为足够的有限上界，因此其结果是本模型下的全局成本下限。固定参考规则可回收比例为 `(规则成本-DP成本)/规则成本`；`(规则成本/DP成本)-1` 描述规则贵多少，分母不同。

v2 必须显式指定 `max_vms` 及有限 rho 网格。结果只在该域内精确；不一定是更大 VM 域、连续 rho 或不同服务/路由模型下的下限。λ>0 优化的是成本+λ×违约，不能把这个成本字段当成满足 epsilon 的最低成本。RL-0 只在小型独立合成负载上演示 v2 的 λ 扫描，完整主实验的 v2 DP 规模、epsilon 可行性和网格误差尚未评估。

无约束 v1 DP 允许直接跳到任意目标；残差 ±3 的 RL 动作域更小。DP 是乐观下限，不能保证某种神经网络、有限数据或残差动作一定能达到它。若规则本身可回收的差距很小，优先预测/报童优化仍比扩大 RL 训练合理。

## 数据及 RL-1–4 仍需的工作

[configs/rl0.json](../configs/rl0.json) 固定后续切分：Azure 1–9 天训练、10–11 天验证、12–14 天测试。`load_azure_partition` 仅返回请求的数据段；增强在传入分区内部完成，幅度 0.5–2、时间伸缩 0.8–1.25、Poisson 重采样，边界不读相邻验证/测试段。增强得到的是新合成序列，相对这条序列的观测仍是因果的；不应称为实际 Azure 到达样本。

本次只使用训练段的一次增强回合/首步，以及验证段前 6 小时固定策略；若启用 GBDT，训练标签全部来自 1–9 天，模型参数固定，没选 checkpoint。报告明确区分 GBDT 拟合与尚未发生的 PPO 训练。此前 Azure ML 研究使用了后 7 天，因此 12–14 天有历史暴露；后续 RL 可以冻结独立协议，但不能声称那是项目从未接触的测试集。

后续要做 PPO/拉格朗日或 λ 扫描；同预算调参 LIBRA、分位数、GBDT+规则、v2 SPOCK/AUTO；至少两种服务时间；5 个独立训练种子与学习曲线；只用验证集选择 checkpoint；报告样本数、训练时间、成本、预期违约率、取消启动开销及与适用 DP 下限的差距。RL-0 的五个随机动作种子只用于参照，不能替代训练统计。当前不宣称已经完成 RL-1–4 或三周实验。
