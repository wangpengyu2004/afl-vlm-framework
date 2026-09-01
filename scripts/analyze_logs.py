"""日志分析：python -m scripts.analyze_logs --dir runs/xxx

输出：
    1. 到达时间线（客户端/任务/版本/staleness）
    2. 每个聚合批次的施加顺序
    3. 各任务能力曲线 + 峰值-终值遗忘摘要
可选: --plot 生成到达时序图与能力曲线 PNG（需要 matplotlib）。
"""

from __future__ import annotations

import argparse
import os

from afl_vlm.evaluation.forgetting import capability_series, forgetting_summary
from afl_vlm.logging_utils import read_jsonl


def show_arrivals(run_dir: str, limit: int) -> None:
    rows = read_jsonl(os.path.join(run_dir, "arrivals.jsonl"))
    print(f"\n--- 到达时间线（{len(rows)} 条，显示前 {min(limit, len(rows))} 条）---")
    print(f"{'client':<8}{'task':<12}{'round':<6}{'ver@train':<10}{'train_s':<10}{'net_delay':<10}")
    for r in rows[:limit]:
        print(f"{r['client']:<8}{r['task']:<12}{r['round']:<6}{r['version_trained_on']:<10}"
              f"{r['train_seconds']:<10.2f}{r['net_delay']:<10.3f}")


def show_aggregations(run_dir: str, limit: int) -> None:
    rows = read_jsonl(os.path.join(run_dir, "aggregations.jsonl"))
    print(f"\n--- 聚合施加顺序（{len(rows)} 次施加）---")
    by_batch: dict[int, list[dict]] = {}
    for r in rows:
        by_batch.setdefault(r["batch"], []).append(r)
    for b in sorted(by_batch)[:limit]:
        seq = by_batch[b]
        chain = " → ".join(f"{r['client']}({r['task']},st{r['staleness']})" for r in seq)
        print(f"batch {b}: {chain}")


def show_evals(run_dir: str) -> None:
    rows = read_jsonl(os.path.join(run_dir, "evals.jsonl"))
    if not rows:
        print("\n（无评估记录）")
        return
    print(f"\n--- 能力曲线 ---")
    for task, pts in capability_series(rows).items():
        curve = " ".join(f"{p['capability']:.3f}" for p in pts)
        print(f"{task:<14} {curve}")
    print(f"\n--- 遗忘摘要（drop = 峰值 − 终值，>0 即遗忘）---")
    for task, s in forgetting_summary(rows).items():
        print(f"{task:<14} peak={s['peak']:.4f} final={s['final']:.4f} "
              f"drop={s['drop']:.4f} ({s['drop_ratio']*100:.1f}%)")


def maybe_plot(run_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("（未安装 matplotlib，跳过绘图）")
        return
    arrivals = read_jsonl(os.path.join(run_dir, "arrivals.jsonl"))
    if arrivals:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, r in enumerate(arrivals):
            color = {"c%d" % j: f"C{j%10}" for j in range(100)}.get(r["client"], None)
            ax.plot([r["t_train_start"], r["t_arrival"]], [i, i], lw=4,
                    color=color or "C0")
            ax.scatter([r["t_arrival"]], [i], color="red", zorder=3, s=12)
        ax.set_yticks(range(len(arrivals)))
        ax.set_yticklabels([f"{r['client']}:{r['task']}" for r in arrivals], fontsize=7)
        ax.set_xlabel("wall-clock time"); ax.set_title("arrival timeline (red = arrival)")
        fig.tight_layout(); fig.savefig(os.path.join(run_dir, "timeline.png"), dpi=150)
        print(f"图已保存: {os.path.join(run_dir, 'timeline.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run 输出目录")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    show_arrivals(args.dir, args.limit)
    show_aggregations(args.dir, args.limit)
    show_evals(args.dir)
    if args.plot:
        maybe_plot(args.dir)


if __name__ == "__main__":
    main()
