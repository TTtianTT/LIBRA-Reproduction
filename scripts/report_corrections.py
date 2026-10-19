#!/usr/bin/env python3
"""Attach short, reproducible methodological corrections to historical reports."""
from pathlib import Path

START = "<!-- research-correction:start -->"
END = "<!-- research-correction:end -->"


def corrected_report(text,kind):
    if START in text:
        before,rest = text.split(START,1)
        _,after = rest.split(END,1)
        text = before+after.lstrip("\n")
    title,_,body = text.partition("\n")
    if kind=="study":
        finding = "**本报告改列早期训练探索记录：85 次运行各只有 8k–16k 步，不能据此评价 PPO 的收敛性能或 RL-1–4 的有效性。**"
    else:
        finding = "**主要发现更正：PPO 的 15/15 个高预算随机初始化选中结果，都可由固定残差动作解释；没有展示状态相关的额外控制收益。教师对照漏扫了 δ=−1。**"
    notice = [START,finding,"",
        "[固定动作复核](../constant_action_audit/REPORT.md)给出了全部训练网格、动作直方图和逐种子匹配。",
        "旧 v2 使用的混合等待尾不构成有效的 M/D/16 SLA 模型；下文违约数仅保留为旧近似的历史数值，撤回物理 SLA 解读。",
        "查表与跨秒事件队列修复、以及进一步 RL 门槛另见 [物理模型核查](../physical_headroom/REPORT.md)。",END,""]
    return title+"\n\n"+"\n".join(notice)+"\n"+body.lstrip("\n")


def main():
    root = Path(__file__).resolve().parents[1]
    for kind,directory in (("study","rl_study"),("diagnostics","rl_diagnostics")):
        path = root/"outputs"/directory/"REPORT.md"
        path.write_text(corrected_report(path.read_text(),kind))


if __name__=="__main__":
    main()
