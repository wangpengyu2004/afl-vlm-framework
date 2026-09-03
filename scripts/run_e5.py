"""E5 baseline 矩阵一键驱动：9 种聚合策略 × 多种子，汇总对比表。

Baseline 矩阵（全部跑在同一 E1 设定上，只换聚合/顺序）：
    e5_immediate_fifo   FedAsync 原始形态：到达即施加、等权（顺序=FIFO）
    e5_fedasync         FedAsync staleness 降权 λ=0.3
    e5_fedasync_l05     λ=0.5（更激进的降权）
    e5_fedbuff_k4       FedBuff 缓冲 K=4（并发异步，权重均匀）
    e5_fedcompass       FedCompass 时间变化 staleness 阈值（策略层近似）
    e5_sync_fedavg      同步屏障锚（攒齐全部 9 客户端才聚合=FedAvg）
    e5_sync_fedprox     同步 + 客户端 FedProx μ=0.01
    e5_random_order     FIFO 但批内随机序（fixed_order/random，顺序效应对照组）
    e5_fullmix          K=9 全混合（ACE 的任务间梯度混合设定）

用法（服务器上，code/ 目录；E5 建议在数据/标定完成后跑）:
    python -m scripts.run_e5 --seeds 42,43,44
    python -m scripts.run_e5 --configs e5_fedbuff_k4,e5_fullmix --seeds 42
产出：runs/e5/<name>_s<seed>/ 各 run 目录 + runs/e5_summary.csv + 终端表。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from afl_vlm.evaluation.forgetting import capability_metric   # noqa: E402
from afl_vlm.logging_utils import read_jsonl                  # noqa: E402
from scripts.analyze_structure import steady_std              # noqa: E402

ALL_CONFIGS = [
    "e5_immediate_fifo", "e5_fedasync", "e5_fedasync_l05", "e5_fedbuff_k4",
    "e5_fedcompass", "e5_sync_fedavg", "e5_sync_fedprox", "e5_random_order",
    "e5_fullmix",
]


def write_seed_config(name: str, seed: int, out_dir: str) -> str:
    path_src = os.path.join(REPO_ROOT, "configs", f"{name}.yaml")
    with open(path_src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", {})["seed"] = seed
    cfg["experiment"]["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def run_train(config_path: str, out_dir: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    print(f"\n>>> python -m scripts.run_train --config {config_path} --out {out_dir}")
    subprocess.run([sys.executable, "-m", "scripts.run_train",
                    "--config", config_path, "--out", out_dir],
                   cwd=REPO_ROOT, env=env, check=True)
    print(f"<<< 完成（{time.time() - t0:.0f}s）")


def collect_run(run_dir: str, warmup_frac: float = 0.2) -> dict:
    """从 run 目录提取汇总指标：final 能力 / 稳态震荡 / 慢任务权重占比 / 平均 staleness。"""
    rows = read_jsonl(os.path.join(run_dir, "evals.jsonl"))
    finals = {r["task"]: capability_metric(r) for r in rows if r.get("tag") == "final"}
    curves: dict[str, list[float]] = {}
    for r in rows:
        if r.get("tag") == "final":
            continue
        curves.setdefault(r["task"], []).append(capability_metric(r))
    amp = {t: steady_std(v, warmup_frac) for t, v in curves.items()}

    aggs = read_jsonl(os.path.join(run_dir, "aggregations.jsonl"))
    w_total = sum(a["weight"] for a in aggs)
    w_by_task: dict[str, float] = {}
    for a in aggs:
        w_by_task[a["task"]] = w_by_task.get(a["task"], 0.0) + a["weight"]
    out = {
        "final": finals,
        "amp": amp,
        "weight_share": {t: (w / w_total if w_total else float("nan"))
                         for t, w in w_by_task.items()},
        "mean_staleness": (sum(a["staleness"] for a in aggs) / len(aggs)) if aggs else float("nan"),
        "n_aggs": len(aggs),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 baseline 矩阵一键驱动")
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    e5_root = os.path.join(REPO_ROOT, "runs", "e5")
    os.makedirs(e5_root, exist_ok=True)

    rows: list[dict] = []
    for name in configs:
        for seed in seeds:
            run_dir = os.path.join(e5_root, f"{name}_s{seed}")
            if os.path.exists(os.path.join(run_dir, "evals.jsonl")):
                print(f"\n[e5] {name}_s{seed} 已有产出，跳过训练（删目录可重跑）")
            else:
                run_train(write_seed_config(name, seed, run_dir), run_dir)
            m = collect_run(run_dir)
            rows.append({"config": name, "seed": seed, **m})
            print(f"[e5] {name}_s{seed}: final={m['final']} "
                  f"mean_staleness={m['mean_staleness']:.2f} n_aggs={m['n_aggs']}")

    # 汇总 CSV
    tasks = sorted({t for r in rows for t in r["final"]} |
                   {t for r in rows for t in r["amp"]})
    csv_path = os.path.join(REPO_ROOT, "runs", "e5_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed"]
                   + [f"final_{t}" for t in tasks]
                   + [f"amp_{t}" for t in tasks]
                   + [f"wshare_{t}" for t in tasks]
                   + ["mean_staleness", "n_aggs"])
        for r in rows:
            w.writerow([r["config"], r["seed"]]
                       + [f"{r['final'].get(t, float('nan')):.4f}" for t in tasks]
                       + [f"{r['amp'].get(t, float('nan')):.4f}" for t in tasks]
                       + [f"{r['weight_share'].get(t, float('nan')):.4f}" for t in tasks]
                       + [f"{r['mean_staleness']:.3f}", r["n_aggs"]])

    # 终端汇总表（跨种子取均值）
    print(f"\n{'=' * 100}\n[e5] 汇总（各策略跨种子均值；final 越高越好、amp 越低越稳）\n{'=' * 100}")
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    print(f"{'config':<22}{'mean_staleness':>15}{'wshare_ocr':>12}"
          + "".join(f"{f'final_{t}':>12}" for t in tasks)
          + "".join(f"{f'amp_{t}':>10}" for t in tasks))
    for name, rs in by_cfg.items():

        def mean(getter):
            vals = [v for r in rs for v in [getter(r)]
                    if isinstance(v, float) and v == v]
            return sum(vals) / len(vals) if vals else float("nan")

        line = f"{name:<22}{mean(lambda r: r['mean_staleness']):>15.2f}" \
               f"{mean(lambda r: r['weight_share'].get('ocr', float('nan'))):>12.3f}"
        line += "".join(f"{mean(lambda r, t=t: r['final'].get(t, float('nan'))):>12.4f}"
                        for t in tasks)
        line += "".join(f"{mean(lambda r, t=t: r['amp'].get(t, float('nan'))):>10.4f}"
                        for t in tasks)
        print(line)
    print(f"\n已写入: {csv_path}")
    print("下一步: 分析见 V_EXPERIMENTS.md §E5 —— 比较各策略的 final/amp/wshare_ocr，"
          "验证『标量策略（降权/缓冲/同步）都压不住任务方向性错位』这一论文核心论点")


if __name__ == "__main__":
    main()
