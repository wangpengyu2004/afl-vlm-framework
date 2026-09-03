"""V1 到达结构性分析：arrivals.jsonl 的任务序列是不是"按任务成块"。

统计量（全部手写实现，无 scipy 依赖）：
    - 同任务游程（runs）置换检验：把任务标签序列做 1000 次随机置换（保持各任务
      计数），得到"无任务结构"零分布；观测游程数显著更少 = 到达按任务聚集
    - 块长分布（最大/平均连续同任务长度）
    - 任务序列滞后-1 自相关
    - 分任务能力曲线（evals.jsonl）的稳态震荡幅度 A_k（去前 20% 热身取 std）
    - 标量遮蔽比 M = Var(mean_k cap_k) / mean_k Var(cap_k)：M≪1 = 平均指标对
      分任务震荡失明

用法:
    python -m scripts.analyze_structure --dir runs/v1/natural --report out.json
    python -m scripts.analyze_structure --dir runs/v1/natural --plot
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from afl_vlm.evaluation.forgetting import capability_series  # noqa: E402
from afl_vlm.logging_utils import read_jsonl                 # noqa: E402


# -- 到达序列结构 -------------------------------------------------------------

def arrival_task_seq(run_dir: str) -> list[str]:
    """按提交顺序（arrivals.jsonl 行序）排列的任务标签序列。"""
    return [r["task"] for r in read_jsonl(os.path.join(run_dir, "arrivals.jsonl"))]


def count_runs(seq: list[str]) -> int:
    if not seq:
        return 0
    return 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def runs_test(seq: list[str], n_perm: int = 1000, seed: int = 42) -> dict:
    """游程置换检验。游程越少越聚集；p = P(零分布游程 ≤ 观测)（单侧）。"""
    obs = count_runs(seq)
    labels = list(set(seq))
    rng = random.Random(seed)
    pool = list(seq)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if count_runs(pool) <= obs:
            ge += 1
    return {"observed_runs": obs, "n": len(seq),
            "perm_p_left": (ge + 1) / (n_perm + 1)}   # +1 修正避免 p=0


def block_stats(seq: list[str]) -> dict:
    if not seq:
        return {"mean_block": 0.0, "max_block": 0, "blocks": 0}
    sizes, cur = [], 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
        else:
            sizes.append(cur)
            cur = 1
    sizes.append(cur)
    return {"mean_block": sum(sizes) / len(sizes), "max_block": max(sizes),
            "blocks": len(sizes)}


def lag1_autocorr(seq: list[str]) -> float:
    """任务序列滞后-1 自相关（0=无结构，1=完全同任务相连）。"""
    if len(seq) < 2:
        return 0.0
    same = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    return same / (len(seq) - 1)


# -- 能力曲线与震荡 -----------------------------------------------------------

def steady_std(vals: list[float], warmup_frac: float = 0.2) -> float:
    """去掉前 warmup 比例的热身点后取 std。"""
    tail = vals[int(len(vals) * warmup_frac):]
    if len(tail) < 2:
        return 0.0
    m = sum(tail) / len(tail)
    return (sum((x - m) ** 2 for x in tail) / (len(tail) - 1)) ** 0.5


def masking_ratio(curves: dict[str, list[float]]) -> float:
    """M = Var(mean_k c_k) / mean_k Var(c_k)（曲线长度不一取最短对齐）。"""
    if len(curves) < 2:
        return float("nan")
    n = min(len(v) for v in curves.values())
    if n < 3:
        return float("nan")
    cols = [[curves[t][i] for t in curves] for i in range(n)]

    def var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    means = [sum(c) / len(c) for c in cols]
    per_task = [var([c[i] for c in cols]) for i in range(len(cols[0]))]
    denom = sum(per_task) / len(per_task)
    return var(means) / denom if denom > 0 else float("nan")


def capability_curves(run_dir: str) -> dict[str, list[float]]:
    rows = read_jsonl(os.path.join(run_dir, "evals.jsonl"))
    return {t: [p["capability"] for p in pts]
            for t, pts in capability_series(rows).items()}


def analyze_run(run_dir: str, n_perm: int = 1000) -> dict:
    seq = arrival_task_seq(run_dir)
    out = {
        "run_dir": run_dir,
        "n_arrivals": len(seq),
        "runs_test": runs_test(seq, n_perm=n_perm) if seq else {},
        "blocks": block_stats(seq),
        "lag1_autocorr": lag1_autocorr(seq),
    }
    curves = capability_curves(run_dir)
    out["oscillation_amp"] = {t: steady_std(v) for t, v in curves.items()}
    out["masking_ratio"] = masking_ratio(curves)
    return out


def compare_v1(natural_dirs: list[str], shuffled_dirs: list[str],
               n_perm: int = 1000) -> dict:
    """V1 go/no-go：natural 的聚集度/震荡是否显著高于 shuffled（臂 B）。"""
    nat = [analyze_run(d, n_perm) for d in natural_dirs]
    shf = [analyze_run(d, n_perm) for d in shuffled_dirs]

    def mean(xs):
        xs = [x for x in xs if x == x]      # 过滤 NaN
        return sum(xs) / len(xs) if xs else float("nan")

    verdict = {
        "natural_runs": [a["runs_test"].get("observed_runs") for a in nat],
        "shuffled_runs": [a["runs_test"].get("observed_runs") for a in shf],
        "natural_perm_p": [a["runs_test"].get("perm_p_left") for a in nat],
        "natural_mean_block": mean([a["blocks"]["mean_block"] for a in nat]),
        "shuffled_mean_block": mean([a["blocks"]["mean_block"] for a in shf]),
        "natural_amp_mean": mean([v for a in nat for v in a["oscillation_amp"].values()]),
        "shuffled_amp_mean": mean([v for a in shf for v in a["oscillation_amp"].values()]),
        "natural_masking_ratio": mean([a["masking_ratio"] for a in nat]),
    }
    verdict["block_ratio"] = (
        verdict["natural_mean_block"] / verdict["shuffled_mean_block"]
        if verdict["shuffled_mean_block"] else float("nan"))
    verdict["amp_ratio"] = (
        verdict["natural_amp_mean"] / verdict["shuffled_amp_mean"]
        if verdict["shuffled_amp_mean"] else float("nan"))
    # go/no-go 判据（V1 预注册）
    verdict["pass_clustered"] = bool(
        verdict["natural_perm_p"] and max(verdict["natural_perm_p"]) < 0.05
        and verdict["block_ratio"] >= 2.0)
    verdict["pass_oscillation"] = bool(
        verdict["amp_ratio"] == verdict["amp_ratio"] and verdict["amp_ratio"] >= 2.0)
    return verdict


def maybe_plot(run_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("（未安装 matplotlib，跳过绘图）")
        return
    curves = capability_curves(run_dir)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for t, vals in curves.items():
        ax.plot(vals, label=t)
    ax.set_xlabel("eval point (per aggregation batch)")
    ax.set_ylabel("capability (match_rate / -loss)")
    ax.set_title(f"per-task capability — {os.path.basename(run_dir)}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(run_dir, "capability_curves.png")
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run 输出目录")
    ap.add_argument("--report", default=None, help="把分析写成 JSON")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    result = analyze_run(args.dir, args.n_perm)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    if args.plot:
        maybe_plot(args.dir)


if __name__ == "__main__":
    main()
