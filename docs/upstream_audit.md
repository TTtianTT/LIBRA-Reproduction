# 作者公开模拟器审计与历史重放

上游仓库：[aliraza0337/LIBRA](https://github.com/aliraza0337/LIBRA)，固定 commit `53a804412ca24c6eca7fa226fcccc0220b17fb0f`。本地 checkout 在 `upstream/`，保持原文件。主实验的核心实现位于 `libra_sim/`，不 import 上游代码。

## 影响复现的差异

| 位置 | 上游行为 | 本项目主实验 |
|---|---|---|
| Python 版本 | Python 2 `print`、iterator `.next` | Python 3.10+ |
| trace 长度 | 默认 45,000 秒 | 论文所述 12 小时，43,200 秒；允许显式 45,000 |
| `manager.handle_requests` | 先用旧 avg 更新 std，再更新 avg | 按式 (6)→(7) 使用新 avg |
| `manager.handle_requests` | `int(abs(mean+1*std))`；phi 写死为 1 | phi 可配置、非负截断，缩放时不提前取整估计 |
| `break_even_point` | 配置为 4，但完全没有使用 | 按 Algorithm 1 实现 avg<CIP 分支 |
| `vm_cloud.__init__` | hybrid 先初始化 10 台，再由 manager 缩放 | LIBRA 初始无 VM |
| `vm_cloud_max` | 默认 292 台且初始需冷启动 | 按当前 trace 的真实峰值定规模，预热 |
| Spock | 没有实现 | 根据 IV-C.3 补全 Spock-like 基线 |
| AUTO 平均利用率 | `sum(hist[-300:])/120`；窗口增长后分母不匹配，甚至可超过 1 | 120 秒窗口，除以实际样本数 |
| AUTO 范围 | 组伸缩可能越过上限，下限未严格应用 | 严格 clamp 到 [min,max] |
| 释放 VM | `vms_to_kill` 赋值覆盖；部分调用路径不持续清理 | 在无在途请求的 epoch 边界立即释放、准确累计费用 |
| 冷启动计费 | `max(0,end-ready)+cold_start`，提前关闭仍收整段 cold-start | 实际 `end-created` |
| 结束和统计 | 处理日志中无统一请求数守恒断言 | 统一清空并对所有请求做守恒检查 |
| Gateway | 没有计入 | 按论文说明显式单列成本 |
| 日志/结果 | 每个 VM 请求写一条 INFO，无图表生成 | 按 epoch 批量建模，输出 CSV/JSON/PNG/SVG |

## 重放方式

`scripts/replay_upstream.py` 将六个上游仿真源文件复制到临时目录，只将 `print epoch,...` 改成 `print(epoch,...)`、`.next` 改为 `.__next__`。算法和计费 bug 保留。不会运行 `aws/` 下的文件。

实验覆盖项只有：trace 的绝对路径、结束时间、策略及是否开启 autoscaling；MAX 根据论文 IV-C.1 配置 `ceil(trace_peak/16)=73`，否则会沿用默认 292 台。MAX 的初始冷启动行为不改，以便展示上游与论文描述不一致之处。

由于上游每个请求都构造对象和发出 INFO，该重放明显慢于批量实现。终端只显示汇总，原始 print 信息写入同名 `.stdout.log`。结果与六个上游源文件的 SHA-256 一并记录在 [upstream_replay.json](../outputs/upstream_replay.json)。

## WITS 前 43,200 秒实际结果

| 原始策略名 | 成本 USD，不含 Gateway | SLA 违约 % | VM 秒 |
|---|---:|---:|---:|
| hybrid | 21.859056 | 0 | 545,400 |
| serverless | 53.065842 | 0 | 0 |
| vm_cloud_max（73 台） | 87.600000 | 0.405682 | 3,153,600 |
| vm_cloud（AUTO） | 33.924167 | 2.458536 | 1,221,270 |

四种策略都完成了输入的 6,217,439 个请求。原始 hybrid 加上相同 12 小时 t2.micro Gateway 成本 $0.1392 后，为 $21.998256；主实验独立实现为 $22.295063。原始 MAX 的 SLA 违约来自初始冷启动；其费用与预热 MAX 相同是历史计费公式及仿真窗口边界共同产生的结果，不表示两种初始条件相同。

这些结果说明：直接把上游改成 Python 3 并运行，也不能自动复现论文图 4–6。主实现与历史重放分开报告，避免把修复后的指标冒充作者原始输出。
