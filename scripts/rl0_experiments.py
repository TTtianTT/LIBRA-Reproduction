#!/usr/bin/env python3
"""Run RL-0 accounting, exact bounds, and untrained environment diagnostics.

No PPO fitting or Azure test-day evaluation happens in this script.
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim import Config, simulate
from libra_sim.traces import load_trace
from libra_ml.features import training_set
from libra_ml.models import GBDT_PARAMS, fit_gbdt
from libra_ml.policies import window_quantile_estimator
from libra_rl.data import AZURE_SPLITS, load_azure_partition
from libra_rl.dp import solve_dp
from libra_rl.env import LibraEnv, RHO_GRID
from libra_rl.model import CostModel
from libra_rl.queueing import violation_probability


def play(env, *, seed=0, random=False, rho=1.0):
    env.reset(seed=seed, options={"start": 0})
    env.action_space.seed(seed)
    done = False
    while not done:
        if random:
            result = env.step(env.action_space.sample())
        else:
            result = env.step_target(env.rule_target(), rho=rho)
        done = result[3]
    return env.summary()


def parity(counts, config, policy):
    estimator = None if policy == "paper" else window_quantile_estimator(CostModel(config).tau)
    routing = "estimate" if policy == "paper" else "capacity"
    reference = simulate(counts, config=config, estimator=estimator, routing=routing)
    env = LibraEnv(counts, model=CostModel(config),
                   decision_seconds=config.scaling_interval_seconds, episode_seconds=None)
    env.reset(seed=0)
    for decision in reference["decisions"]:
        env.step_target(decision["target_vms"], rho=config.rho,
                        estimate=decision["estimate"], routing=routing)
    result = env.summary()
    differences = {k: abs(result[k] - reference["summary"][k]) for k in
                   ("vm_cost_usd", "faas_cost_usd", "gateway_cost_usd", "total_cost_usd")}
    if max(differences.values()) > 1e-9:
        raise AssertionError(f"accounting parity failed: {differences}")
    return {"policy": policy, "decision_seconds": config.scaling_interval_seconds,
            "cost_usd": result["total_cost_usd"], "absolute_errors_usd": differences}


def random_summary(values):
    # Two-sided 95% Student t interval, df=4, for these five random-policy runs.
    mean = float(np.mean(values))
    half_width = 2.7764451051977987 * float(np.std(values, ddof=1)) / math.sqrt(5)
    return {"costs_usd": values, "mean_usd": mean, "ci95_usd": [mean - half_width, mean + half_width],
            "meaning": "random residual actions, not five trained PPO seeds"}


def azure_smoke(with_forecast):
    if not (ROOT / "data/azure2021/aggregate_counts.npz").exists():
        return {"status": "dataset unavailable; skipped; no synthetic training substituted"}
    train = load_azure_partition("train", root=ROOT / "data/azure2021")
    validation = load_azure_partition("validation", root=ROOT / "data/azure2021")
    forecast, fit_metadata = None, {"enabled": False}
    if with_forecast:
        t0 = time.perf_counter()
        features, y = training_set(train, stride=300, target_step=10)
        forecast = fit_gbdt(features, y, CostModel().tau, seed=0)
        fit_metadata = {"enabled": True, "training_rows": len(y), "feature_count": features.shape[1],
                        "tau": CostModel().tau, "parameters": GBDT_PARAMS,
                        "fit_seconds": time.perf_counter() - t0,
                        "target_domain": "raw Azure train days 1-9 only; v1 quantile"}
    rows = []
    for version, service in (("v1", 1.0), ("v2", 0.5), ("v2", 0.7)):
        model = CostModel(version=version, service_seconds=service)
        rho = 1.0 if version == "v1" else 0.8
        # 64x is a declared replication, not the observed Azure invocation rate.
        env = LibraEnv(train * 64, model=model, augment=True, forecast_model=forecast)
        obs, metadata = env.reset(seed=0)
        _, _, _, _, info = env.step(3 if version == "v1" else np.array([3, 2]))
        rows.append({"split": "train", "version": version, "service_seconds": service,
                     "replication": 64, "seed": 0, "episode": metadata,
                     "observation_size": len(obs), "first_step": info})
        modes = (False, True) if forecast is not None else (False,)
        for learned_rule in modes:
            env = LibraEnv(validation * 64, model=model, forecast_model=forecast,
                           rule_rho=rho, rule_uses_forecast=learned_rule)
            result = play(env, rho=rho)
            rows.append({"split": "validation", "version": version, "service_seconds": service,
                         "replication": 64, "range_seconds_in_partition": [0, 21600],
                         "policy": "GBDT rule (untuned)" if learned_rule else "window rule (untuned)",
                         "summary": result})
    return {"status": "train/validation smoke only", "splits": AZURE_SPLITS,
            "forecast": fit_metadata, "rows": rows,
            "test_evaluated": False,
            "caveat": "Days 12-14 were previously examined in the ML project; not an untouched global holdout."}


def generate_report(data, output):
    lines = ["# RL-0：成本下限与环境一致性", "",
             "本次仅完成 RL-0，没有训练 PPO，没有得到 RL 优于规则的实验结论。", "",
             "## 已知未来的 v1 成本下限", "",
             "同一段 43,200 秒 trace，零初始 VM，100 秒启动，按实际 VM 秒计费，含 Gateway。", "",
             "| Trace | 决策秒数 | 300 秒分位数规则 USD | DP USD | 规则比 DP 贵 | 规则最多可省 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in data["bounds"]:
        lines.append(f"| {row['trace']} | {row['decision_seconds']} | {row['rule_cost_usd']:.6f} | "
                     f"{row['dp_cost_usd']:.6f} | {100 * row['over_optimum_fraction']:.2f}% | "
                     f"{100 * row['recoverable_fraction']:.2f}% |")
    lines += ["", "“规则比 DP 贵”以 DP 为分母，“最多可省”以规则为分母。DP 允许任意目标 VM 数；"
              "残差动作 ±3 的 RL 未必能达到这个下限。规则是固定 300 秒历史窗口，CIP=4，填满就绪容量。",
              "", "## 一致性", "",
              f"论文 LIBRA 与分位数规则共 {len(data['parity'])} 项完整 trace 回放，"
              f"VM/FaaS/Gateway/总成本最大绝对差 {max(max(r['absolute_errors_usd'].values()) for r in data['parity']):.3g} USD，"
              "阈值 1e-9。60 秒环境由独立时钟回放、重叠启动/取消测试及小规模穷举校验。", "",
              "## 随机策略参照（60 秒）", "",
              "下列置信区间只描述五次随机残差策略运行，不是 PPO 的训练种子统计。", "",
              "| Trace | 随机策略均值 USD | 95% t 区间 | 永远 a=0 USD |", "|---|---:|---:|---:|"]
    for row in data["random_reference"]:
        lo, hi = row["ci95_usd"]
        lines.append(f"| {row['trace']} | {row['mean_usd']:.6f} | [{lo:.6f}, {hi:.6f}] | {row['rule_cost_usd']:.6f} |")
    lines += ["", "## v2 的模型假设", "",
              "`wait_budget` 用 Erlang-C 等待概率乘等待尾分布，并从 SLA 中减去固定服务时间；"
              "它复现提议中的概率表，但不是完整 M/M/16 响应时间。`mmc_response` 使用指数服务的完整响应尾分布。"
              "两者均为每秒准稳态、每 VM 均衡分流近似，不保留实际排队积压。ρ≥1 时保守地按 100% 违约处理。", "",
              "服务时间 0.5/0.7 秒同时改变 VM 吞吐（16/service）和 FaaS GB-second 成本。"
              "FaaS 无排队/冷启动，执行时间固定为 service，调用费照收。默认 CIP 保留论文值 4，未调参。"
              "v2 分位数规则目标是 ceil(q/(throughput×参考ρ))，参考ρ默认 0.8；动作里的ρ控制实际路由。", "",
              "`v2_diagnostics` 是同一模型内未调参规则的敏感性诊断，不能视为 RL-2 的公平调参比较。"
              "`v2_bounded_dp` 在独立小型合成负载、最多 8 台 VM、有限ρ网格上精确求解成本+λ×预期违约，"
              "并不是完整 trace 上带 SLA 约束的全局下限。", "",
              "## 数据与后续边界", "",
              "Azure 切分已固定为训练 1–9 天、验证 10–11 天、测试 12–14 天。"
              "本脚本仅跑训练/验证 smoke；12–14 天此前已被 ML 实验使用，不能称为从未查看过的测试集。"
              "64 倍负载明确是数据复制。GBDT 如启用，仅在训练 1–9 天拟合 v1 分位数，用于状态接口校验，"
              "不是已调优的 v2 预测器。", "",
              "完整 RL-1–4 尚未执行：PPO/对偶更新、同预算基线网格（含 SPOCK/AUTO 的 v2 适配）、"
              "验证集 checkpoint 选择、五训练种子和学习曲线、泛化/消融仍需后续实验。", "",
              "运行方式、DP 状态推导、奖励单位和队列模型来源见 [docs/rl0.md](../../docs/rl0.md)。", "",
              "![DP bound](bounds.png)", "", "![Latency assumptions](latency.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def plots(data, output):
    from libra_sim.plotting import plt
    labels = [f"{r['trace']}\nK={r['decision_seconds']}" for r in data["bounds"]]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(positions - 0.18, [r["rule_cost_usd"] for r in data["bounds"]], width=0.36, label="Window quantile")
    ax.bar(positions + 0.18, [r["dp_cost_usd"] for r in data["bounds"]], width=0.36, label="Clairvoyant exact DP")
    ax.set(xticks=positions, xticklabels=labels, ylabel="Total cost (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "bounds.png", dpi=160)
    fig.savefig(output / "bounds.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    utilization = np.linspace(0, 0.999, 300)
    for ax, mode in zip(axes, ("wait_budget", "mmc_response")):
        for service in (0.5, 0.7, 1.0):
            ax.plot(utilization, 100 * violation_probability(utilization, service, mode=mode), label=f"service={service}s")
        ax.set(title=mode, xlabel="VM utilization", ylabel="SLA violation probability (%)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "latency.png", dpi=160)
    fig.savefig(output / "latency.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rl0")
    parser.add_argument("--with-forecast", action="store_true")
    parser.add_argument("--skip-azure", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_files = sorted(set([Path(__file__), ROOT / "configs/rl0.json", ROOT / "pyproject.toml"]
                             + list((ROOT / "libra_rl").glob("*.py"))
                             + list((ROOT / "libra_sim").glob("*.py"))
                             + list((ROOT / "libra_ml").glob("*.py"))))
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "command": [".venv/bin/python", "scripts/rl0_experiments.py"]
            + sys.argv[1:], "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in source_files},
            "protocol": json.loads((ROOT / "configs/rl0.json").read_text()),
            "base_config": asdict(Config()), "traces": {}, "bounds": [], "parity": [],
            "random_reference": [], "v2_diagnostics": [], "v2_bounded_dp": [],
            "queue_table": {mode: {str(s): violation_probability([0.7, 0.8, 0.9], s, mode=mode).tolist()
                                   for s in (1.0, 0.7, 0.5)}
                            for mode in ("wait_budget", "mmc_response")}}
    for name, filename in (("WITS", "wits_logs.txt"), ("Berkeley", "berkeley_logs.txt")):
        x, trace = load_trace(ROOT / "data/traces" / filename, seconds=43200)
        data["traces"][name] = trace
        for k in (300, 120, 60):
            print(f"{name}, K={k}: exact DP", flush=True)
            dp = solve_dp(x, decision_seconds=k)
            env = LibraEnv(x, decision_seconds=k, episode_seconds=None)
            rule = play(env)["total_cost_usd"]
            data["bounds"].append({"trace": name, "decision_seconds": k, "rule_cost_usd": rule,
                                   "dp_cost_usd": dp.total_cost_usd,
                                   "over_optimum_fraction": rule / dp.total_cost_usd - 1,
                                   "recoverable_fraction": 1 - dp.total_cost_usd / rule,
                                   "dp": asdict(dp)})
            if k > Config().vm_cold_start_seconds:
                for policy in ("paper", "quantile"):
                    cfg = replace(Config(), scaling_interval_seconds=k, rho=0.8 if policy == "paper" else 1.0)
                    data["parity"].append({"trace": name, **parity(x, cfg, policy)})
            if k == 60:
                values = [play(LibraEnv(x, episode_seconds=None), random=True, seed=s)["total_cost_usd"]
                          for s in range(5)]
                data["random_reference"].append({"trace": name, "rule_cost_usd": rule, **random_summary(values)})
        for mode in ("wait_budget", "mmc_response"):
            for service in (0.5, 0.7):
                model = CostModel(version="v2", service_seconds=service, latency_mode=mode)
                for rho in RHO_GRID:
                    result = play(LibraEnv(x, model=model, episode_seconds=None, rule_rho=rho), rho=rho)
                    data["v2_diagnostics"].append({"trace": name, "latency_mode": mode,
                                                   "service_seconds": service, "rho": rho,
                                                   "policy": "untuned window rule", "summary": result})
    synthetic = np.maximum(0, np.rint(80 + 60 * np.sin(np.arange(1800) * 2 * np.pi / 600))).astype(int)
    for service in (0.5, 0.7):
        model = CostModel(version="v2", service_seconds=service)
        for penalty in (0.0, 0.00001, 0.001):
            dp = solve_dp(synthetic, model=model, decision_seconds=60, max_vms=8,
                          rho_grid=RHO_GRID, penalty=penalty)
            data["v2_bounded_dp"].append({"trace": "synthetic: 80+60*sin(2πt/600), 1800s",
                                          "service_seconds": service, **asdict(dp)})
    print("Azure train/validation environment smoke", flush=True)
    data["azure"] = {"status": "skipped explicitly"} if args.skip_azure else azure_smoke(args.with_forecast)
    import gymnasium
    import scipy
    import sklearn
    data["versions"] = {"numpy": np.__version__, "gymnasium": gymnasium.__version__,
                        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    data["elapsed_seconds"] = time.perf_counter() - started
    (output / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    columns = ["trace", "decision_seconds", "rule_cost_usd", "dp_cost_usd",
               "over_optimum_fraction", "recoverable_fraction"]
    with (output / "bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["bounds"])
    plots(data, output)
    generate_report(data, output)
    print(f"RL-0 finished in {data['elapsed_seconds']:.1f}s; {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
