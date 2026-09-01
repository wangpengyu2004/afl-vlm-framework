"""主入口：python -m scripts.run_train --config configs/smoke_cpu.yaml"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from afl_vlm.config import load_config, dump_config, required_history
from afl_vlm.models import build_model_adapter, SharedModelManager
from afl_vlm.device_pool import resolve_devices
from afl_vlm.data.tasks import get_task, build_collator
from afl_vlm.aggregation import build_policy
from afl_vlm.planner import build_plan, print_plan_schedule
from afl_vlm.server import AsyncServer
from afl_vlm.client import ClientThread


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _delay_desc(d) -> str:
    if d.dist == "fixed":
        return f"fixed({d.value:g})"
    if d.dist in ("uniform", "uniform_int"):
        return f"{d.dist}({d.low:g}..{d.high:g})"
    if d.dist == "lognormal":
        return f"lognormal(mu={d.mu:g},sigma={d.sigma:g})"
    if d.dist == "exponential":
        return f"exponential(rate={d.rate:g})"
    return d.dist


def print_plan(cfg) -> None:
    print("=" * 70)
    print(f"实验: {cfg.name} | seed={cfg.seed} | 模型={cfg.model.name} | "
          f"聚合={cfg.server.aggregation.name} | 计时={cfg.timing}")
    print(f"任务: " + ", ".join(f"{n}({t.source.type})" for n, t in cfg.tasks.items()))
    if cfg.clients.task_profiles:
        print("任务相关延迟: " + "; ".join(
            f"{t}: {prof}" for t, prof in cfg.clients.task_profiles.items()))
    print(f"{'客户端':<8}{'任务':<12}{'轮数':<6}{'速度':<8}{'启动':<6}{'模式':<11}延迟/τ")
    for c in cfg.clients.expand():
        if c["delay_mode"] == "staleness":
            desc = _delay_desc(c["staleness_tau"])
        else:
            desc = _delay_desc(c["net_delay"])
        lag = c["download_lag"]
        if (lag.dist == "fixed" and lag.value > 0) or \
           (lag.dist in ("uniform", "uniform_int") and lag.high > 0):
            desc += f" + dl:{_delay_desc(lag)}"
        if cfg.timing == "virtual":
            desc += f" | t={c['train_seconds']:g}s"
        print(f"{c['id']:<8}{c['task']:<12}{c['num_rounds']:<6}{c['speed_factor']:<8.2f}"
              f"{c['start_offset']:<6.1f}{c['delay_mode']:<11}{desc}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="AFL × VLM 异步联邦指令微调")
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--out", default=None, help="覆盖 experiment.output_dir")
    ap.add_argument("--dry-run", action="store_true", help="只打印实验计划，不训练")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg.output_dir = args.out

    if args.dry_run:
        print_plan(cfg)
        if cfg.timing == "virtual":
            # 阶段1 秒级跑完：不训练也能看到完整到达/施加调度表
            print_plan_schedule(build_plan(cfg))
        return

    set_seed(cfg.seed)
    print_plan(cfg)

    # 虚拟时钟模式：先离线生成完整调度表（阶段1），训练只是按表回放（阶段2）
    plan = build_plan(cfg) if cfg.timing == "virtual" else None
    if plan is not None:
        print(f"[setup] 虚拟时钟计划已生成: {len(plan.rounds)} 轮次 / "
              f"{len(plan.applies)} 次施加 / {len(plan.arrival_order)} 次到达")

    pool = resolve_devices(cfg.model.device)
    print(f"[setup] 设备池: {pool.devices}")

    adapter = build_model_adapter(cfg.model)
    manager = SharedModelManager(adapter, pool, history_size=required_history(cfg))
    print(f"[setup] 加载模型 {cfg.model.name} ...")
    manager.startup()
    if plan is not None:
        consumers = plan.sync_consumers()
        manager.set_plan_retention(consumers)
        print(f"[setup] 按计划保留版本快照: {sorted(consumers)}")

    collator = build_collator(adapter, cfg.model.name)
    tasks = {name: get_task(tc) for name, tc in cfg.tasks.items()}
    for t in tasks.values():
        print(f"[setup] {t}")

    policy = build_policy(cfg.server.aggregation)
    server = AsyncServer(cfg, manager, policy, tasks, collator, plan=plan)
    dump_config(cfg, f"{cfg.output_dir}/config.used.yaml")
    if plan is not None:
        import json
        with open(f"{cfg.output_dir}/plan.json", "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, ensure_ascii=False, indent=1)
        print(f"[setup] 调度表已写入 {cfg.output_dir}/plan.json")

    threads = [
        ClientThread(cc, tasks[cc["task"]], manager, server, cfg,
                     plan_rounds=(plan.rounds_of(cc["id"]) if plan else None))
        for cc in cfg.clients.expand()
    ]
    for t in threads:
        t.start()
    print(f"[setup] {len(threads)} 个客户端线程已启动")

    server.run()


if __name__ == "__main__":
    main()
