"""Killer experiment 驱动：**同一批更新、不同聚合顺序** → 各任务能力对比。

设计要点（保证跨顺序可比）：
    - 所有客户端只跑 1 轮（num_rounds=1），全部基于同一个初始全局模型 W0 训练
      → 产生的 delta 集合在各 order 配置间完全一致（同种子同数据同初始化）；
    - 唯一自由度 = fixed_order 的 order.mode；
    - 每个 order 单独起一个 run_train 子进程，互不污染。

用法（仓库根目录）:
    python -m scripts.run_killer --config configs/killer/base.yaml \
        --orders random,clustered,alternating,blocked --out runs/killer
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys

import yaml

from afl_vlm.evaluation.forgetting import capability_metric
from afl_vlm.logging_utils import read_jsonl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_order_config(base_cfg: dict, order: str, out_dir: str) -> str:
    """在 base 配置上叠加 fixed_order + 单轮设定，写成子进程用的 yaml。"""
    cfg = copy.deepcopy(base_cfg)
    n_clients = sum(cfg["clients"]["task_mix"].values())

    cfg.setdefault("experiment", {})["output_dir"] = out_dir
    agg = cfg.setdefault("server", {}).setdefault("aggregation", {})
    agg["name"] = "fixed_order"
    agg["batch_size"] = n_clients          # 一批攒齐全部更新，一次性按序施加
    agg["order"] = {"mode": order, "seed": cfg.get("experiment", {}).get("seed", 42)}
    cfg["server"]["eval_every_batches"] = 1
    cfg["clients"]["num_rounds"] = 1       # 关键：全员只训一轮 → delta 集合跨顺序一致
    # 延迟固定 wallclock：staleness 的 τ 持留在 wallclock 实时模式下会让到达集合
    # 依赖版本推进，破坏"唯一自由度 = 顺序"的可比性（timing: virtual 时 τ 在计划里
    # 模拟，语义等价，无需此顾虑）。killer 里延迟只负责产生到达时刻
    cfg["clients"]["delay_mode"] = "wallclock"
    cfg["clients"].pop("staleness_tau", None)
    cfg["clients"].pop("download_lag", None)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def run_one(config_path: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--config", config_path],
        cwd=REPO_ROOT, env=env, check=True,
    )


def collect_final(out_dir: str) -> dict[str, float]:
    rows = [r for r in read_jsonl(os.path.join(out_dir, "evals.jsonl"))
            if r.get("tag") == "final"]
    return {r["task"]: capability_metric(r) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="killer experiment：同集合×不同顺序")
    ap.add_argument("--config", required=True, help="base 配置（建议 killer/base.yaml）")
    ap.add_argument("--orders", default="random,clustered,alternating,blocked")
    ap.add_argument("--out", default="runs/killer")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    results: dict[str, dict[str, float]] = {}
    for order in orders:
        out_dir = os.path.join(args.out, order)
        print(f"\n{'='*70}\n[killer] order = {order} → {out_dir}\n{'='*70}")
        cfg_path = build_order_config(base_cfg, order, out_dir)
        run_one(cfg_path)
        results[order] = collect_final(out_dir)

    # 汇总表
    tasks = sorted({t for r in results.values() for t in r})
    csv_path = os.path.join(args.out, "killer_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order"] + tasks)
        for order, res in results.items():
            w.writerow([order] + [f"{res.get(t, float('nan')):.4f}" for t in tasks])

    print(f"\n{'='*70}\n[killer] 结果汇总（能力值越高越好）\n{'='*70}")
    header = f"{'order':<14}" + "".join(f"{t:<16}" for t in tasks)
    print(header)
    for order, res in results.items():
        print(f"{order:<14}" + "".join(f"{res.get(t, float('nan')):<16.4f}" for t in tasks))
    print(f"\n已写入: {csv_path}")


if __name__ == "__main__":
    main()
