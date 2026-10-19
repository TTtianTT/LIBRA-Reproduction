# 仿真方法与复现边界

## 依据与范围

核对 [arXiv v1](https://arxiv.org/pdf/2104.05491) 的 Section II–IV、Algorithm 1、Figures 3–6，以及[官方模拟器](https://github.com/aliraza0337/LIBRA/tree/53a804412ca24c6eca7fa226fcccc0220b17fb0f)。Section V 的 AWS 实验不属于本项目范围。所有美元金额是按论文时期价格计算的**模拟成本**。

主结果使用 `data/traces/wits_logs.txt` 的前 43,200 行。作者提供的是每秒请求计数，而非原始 HTTP/TCP 时间戳，因此每秒内请求统一在 epoch 起点到达。并发容量按内存确定；不额外虚构网络、CPU 争用或 FaaS 冷启动。

## 参数来源

| 参数 | 值 | 依据 |
|---|---:|---|
| epoch、请求服务时间、SLA | 1 s | IV-A/B；服务时间以外的等待导致违约 |
| 请求内存 | 512 MB | IV-A.3 |
| VM 内存、每小时价格 | 8192 MB、$0.1 | m4.large，IV-A.3 |
| VM 冷启动 | 100 s | IV-A.3 |
| alpha、beta | 0.2、0.2 | IV-A.3；代码 `alpha`、`mu` |
| 缩放间隔 K | 300 s | IV-A.3 |
| rho、phi | 0.8、1 | IV-A.3 |
| CIP | 4 req/s | II-C 和上游 `break_even_point`；上游未使用此字段 |
| FaaS 计算费 | $0.00001667 / GB-s | 上游 `serverless_cloud.exec_model` |
| FaaS 调用费 | $0.0000002 / request | 同上；不另计 API Gateway、流量、免费额度 |
| LG Gateway | $0.0116 / h | 上游 t2.micro 价格，V-C 明确使用此机型；IV 只说明计入 LG，未指明机型，属于明确假设 |
| AUTO/SPOCK 最小/最大 VM | 10 / 292 | 上游默认配置 |
| AUTO/SPOCK 伸缩组 | 10 | 上游默认配置 |
| AUTO/SPOCK 高/低阈值 | 0.7 / 0.3 | 上游默认配置 |
| AUTO/SPOCK 观察窗口 | 120 s | 上游 `auto_scale_grace_period`；修正历史平均值计算错误 |

容量 `r_max = floor(8192 / 512) / 1 = 16 req/s`，对应式 (5)。FaaS 单请求成本为 `$8.535e-6`。

固定服务时间与上述价格推导出的单台 VM CIP 实际为 `0.1 / 3600 / 8.535e-6 ≈ 3.25457 req/s`。论文背景示例的 CIP=4 使用其执行模型；主实验为保留公开设置使用 4，不宣称两者严格相等。`results.json` 同时记录固定 CIP 与推导值；可在配置中显式改为推导值。

## 状态更新与 LIBRA

1. 时间从 0 开始；第 t 行请求在 `[t,t+1)` 执行。每个 epoch 起点完成上一 epoch 的服务，冷启动完成的 VM 在恰好 `created+100` 时可用。
2. Traffic Monitor 以首个 epoch 的请求数初始化 avg，deviation=0；随后按式 (6) 更新 avg，再按式 (7) 更新 deviation：`avg=(1-alpha)*avg+alpha*requests`，`deviation=(1-beta)*deviation+beta*abs(requests-avg)`。
3. 在 `t=0,300,600,...` 做缩放，采用该 epoch 已收到的批次计数；不读取未来 epoch。此顺序与作者的“先统计本 epoch 再缩放”约定一致。它是批次时间模型，并不模拟 epoch 内逐个请求到达。
4. 若 `avg<CIP`，VM 数和路由估计均设 0。否则 `vm_reqs=max(0,avg+phi*deviation)`，分配 `ceil(vm_reqs/16)` 台 VM。负 phi 导致负估计时截断到 0，不沿用上游的绝对值。
5. 扩容时，新路由估计延迟到新 VM 冷启动完成后生效；等待期间保留旧路由预算。缩容在 epoch 边界生效，原请求均已完成。LIBRA 无额外最小实例数。
6. 每秒优先将 `min(demand, floor(rho*vm_reqs), ready_vms*16)` 个请求发往 VM，余量发往 FaaS。预算既受估计也受实际 ready 容量限制，因而请求不会被发给尚未就绪的 VM。

式 (7) 的 `std` 是指数平滑的**绝对偏差**，不是平方差的标准差。实现使用 `deviation` 名称以避免混淆。上游先更新 deviation 再更新 avg，使用旧均值，独立重建选择论文公式顺序；这是 phi 扫描结果可能不同的一项原因，不构成已验证的因果分解。

## 基线

- **FaaS**：全部请求交给无限并发、无冷启动的 FaaS。这与论文仿真假设一致，默认 SLA 下无违约。
- **MAX**：扫描所选 trace 峰值，一开始预热 `ceil(peak/16)` 台 VM，整段保持。论文说 MAX 应无违约，因此主实现采用预热；预热费用在测量窗口外，不计入窗口成本。WITS 峰值 1168，故 MAX 为 73 台。原始代码默认是 292 台且从冷启动开始，直接运行不能得到论文描述的 MAX。
- **AUTO**：初始最小 10 台 VM，从冷启动开始。以每个 epoch 实际执行数除以所有已分配 VM 的总容量计算利用率，冷启动 VM 也在分母中，与上游口径一致。收集最近 120 秒利用率；均值高于 0.7 或低于 0.3 时，在下一 epoch 边界增加/减少 10 台，严格限制在 [10,292]。实际发生伸缩后清空观察窗口；无伸缩则继续滚动窗口。
- **SPOCK**：使用同一 AUTO 控制器；仅当存在启动中的 VM 时，把当前 ready 容量之外的新请求交给 FaaS。已有排队请求仍按 FIFO 在 VM 上完成。非启动期的突发仍会排队。初始 VM 启动也属于启动窗口。这是 IV-C.3 的 Spock-like 基线，**不是重新实现 Spock 原论文的全部系统**；LIBRA 作者公开代码也没有此策略。

VM 只用 1 秒统一服务时间，因此共享 FIFO 队列按就绪 VM 总槽位派发，与同容量 VM 上的工作守恒调度在完成时间和成本上等价。仿真不追踪物理请求 ID，而按到达 epoch 批量记录请求，保留每批的等待时间和数量。

## 排队、SLA 与成本

AUTO/SPOCK 超出 VM 容量且未发往 FaaS 的请求保留在 FIFO 队列。响应时间 = 开始执行时间 - 到达时间 + 1 秒；严格大于 SLA 视为违约。没有丢弃、超时删除或对延迟请求重复收费。

trace 结束后不再输入请求或触发控制决策，继续运行到 FIFO 清空，最后一批执行完整一秒后停止计费。所有到达请求必须恰好完成一次。存活 VM 从申请时到释放时连续按秒计费，包含实际发生的冷启动和排队清空时间；不人为给已提前取消的 VM 补收完整冷启动。LIBRA 的 Gateway 贯穿 trace 测量窗口，其成本单列。所有策略的负载分布和 VM 时间线只覆盖输入 trace，末尾清空时间另在 `drain_seconds` 报告。

## 图表定义与结论限制

- 图 3：原始请求率与逐秒 EWMA。
- 图 4a：成本除以本次 LIBRA 成本；右轴为 SLA 违约百分比。
- 图 4b：LIBRA 每次缩放的目标 VM 数；对照是**离线下一 300 秒窗口均值/16 向上取整**。论文没有公开其 “Optimal” 曲线生成算法，因此图中明确写 reference，不将此均值参考曲线称为严格成本最优解。
- 图 4c：每 300 秒实际到达 VM/FaaS 的平均请求率；末尾短窗口按实际长度平均。
- 图 5：总 VM 秒和包括 Gateway 的成本分解，均以 LIBRA 对应指标归一化。
- 图 6：保持其他参数不变扫描 phi=-1,0,1,2,3,4；总成本含同样的 Gateway。

本次 WITS 主实验重现了“LIBRA 零违约且成本低于四个基线”的主要趋势，但 MAX 相对成本、AUTO/SPOCK 违约率以及 phi 最优点不与原图逐数值一致。不能将论文摘要中真实 AWS 实验的 “85%” 当成本地仿真必须匹配的目标；也不能将单条 WITS trace 的趋势推广到所有输入。

已运行 Berkeley 和两种合成 trace，部分场景中 LIBRA 更贵。所有这些结果均保留在输出中，作为该模型和默认参数的适用边界。没有对基线单独增加 Gateway 成本或为目标结论优化阈值。

## 验证

单元测试覆盖 EWMA 顺序、CIP、负 phi、冷启动精确边界、FIFO 等待、结束后清空、缩容费用、旧路由等待新 VM、SPOCK 启动期路由、MAX 预热、零负载以及五种策略的请求守恒。`scripts/verify_artifacts.py` 检查实际结果与逐秒数据、成本公式、响应时间分布的一致性及官方 trace SHA-256。
