"""V 系列前置实验一键驱动：V0 校准冒烟 → V1 到达结构性 → V2 矢量 staleness → V3 killer。

用法（服务器上，code/ 目录）:
    python -m scripts.run_vseries --stage all               # 全流程
    python -m scripts.run_vseries --stage v0                # 只跑校准+冒烟
    python -m scripts.run_vseries --stage v1 --seeds 42,43  # 多种子
    python -m scripts.run_vseries --stage all --skip-calibration   # 无 GPU 先跑机制链
    python -m scripts.run_vseries --stage v3 --killer-config configs/killer/base_pilot.yaml

各阶段：
    v0a smoke        tiny_mock 虚拟时钟冒烟 ×2 → plan.json 逐字节一致（确定性）
    v0b calibration  7B wallclock 每任务 3 客户端 × 2 轮 → 标定 task_profiles
    v1               9 客户端 3/3/3 virtual：natural vs shuffled-duration（臂 B）
                     → 游程置换检验 + 震荡幅度比 → go/no-go
    v2               复用 v1 natural 的 delta_probes → τ 脱钩 / 同 τ 不同命 / 分组检验
    v3               killer（同 delta 集合、只变施加顺序）→ 中途态离散 vs 终点重合

产出：runs/v0|v1|v3/ 各 run 目录 + runs/v*/v*_report.json + 终端 go/no-go 汇总表。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from afl_vlm.logging_utils import read_jsonl            # noqa: E402
from scripts.analyze_structure import analyze_run, compare_v1, maybe_plot  # noqa: E402
from scripts.analyze_vector import analyze as analyze_vector            # noqa: E402

TASKS = ("vqa", "ocr", "caption")


def data_ready() -> bool:
    return all(os.path.exists(os.path.join(REPO_ROOT, "data", "pilot", f"{t}.json"))
               for t in TASKS)


def run_train(config_path: str, out_dir: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    print(f"\n>>> python -m scripts.run_train --config {config_path} --out {out_dir}")
    subprocess.run([sys.executable, "-m", "scripts.run_train",
                    "--config", config_path, "--out", out_dir],
                   cwd=REPO_ROOT, env=env, check=True)
    print(f"<<< 完成（{time.time() - t0:.0f}s）")


def write_seed_config(base_config: str, seed: int, out_dir: str) -> str:
    """复制配置并覆写 seed/output_dir（config+seed 是调度计划的纯函数）。"""
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", {})["seed"] = seed
    cfg["experiment"]["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


# -- V0 -----------------------------------------------------------------------

def stage_v0(args) -> dict:
    report: dict = {"smoke_determinism": None, "calibration": None}

    # v0a：tiny 冒烟 ×2 → plan.json 确定性
    outs = [os.path.join(REPO_ROOT, "runs", "v0", "smoke_a"),
            os.path.join(REPO_ROOT, "runs", "v0", "smoke_b")]
    cfg_path = os.path.join(REPO_ROOT, "configs", "v0_smoke.yaml")
    for out in outs:
        run_train(cfg_path, out)
    plans = []
    for out in outs:
        with open(os.path.join(out, "plan.json"), "rb") as f:
            plans.append(f.read())
    report["smoke_determinism"] = (plans[0] == plans[1])
    print(f"\n[v0a] 调度表确定性（同 config 同 seed 两次 plan.json 逐字节一致）: "
          f"{'PASS' if report['smoke_determinism'] else 'FAIL'}")

    # v0b：7B wallclock 标定（需要数据 + GPU）
    if args.skip_calibration:
        print("[v0b] 跳过标定（--skip-calibration）")
        return report
    if not data_ready():
        print("[v0b] 缺少 data/pilot/ 数据——先跑 "
              "`python -m scripts.prepare_datasets`，或用 --skip-calibration")
        return report
    out = os.path.join(REPO_ROOT, "runs", "v0", "calibration")
    run_train(os.path.join(REPO_ROOT, "configs", "v0_calibration.yaml"), out)
    rows = read_jsonl(os.path.join(out, "arrivals.jsonl"))
    per_task: dict[str, list[float]] = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r["train_seconds"])
    med = {t: sorted(v)[len(v) // 2] for t, v in per_task.items() if v}
    ratio = (max(med.values()) / min(med.values())) if len(med) == 3 else float("nan")
    report["calibration"] = {
        "median_train_seconds": med,
        "max_min_ratio": ratio,
        "recommended_task_profiles": {t: round(s) for t, s in med.items()},
        "pass_ratio_ge_1p5": bool(ratio == ratio and ratio >= 1.5),
    }
    print("\n[v0b] 标定结果（单轮真实训练秒数中位数）:")
    for t, s in med.items():
        print(f"    {t}: {s:.0f}s")
    print(f"    最快/最慢比 = {ratio:.2f}（判据 ≥1.5："
          f"{'PASS' if report['calibration']['pass_ratio_ge_1p5'] else 'FAIL — 拉不开时长差，换数据/改 max_len'}）")
    print("    → 建议把 configs/v1_natural.yaml 的 task_profiles 改为: "
          + json.dumps(report["calibration"]["recommended_task_profiles"]))
    return report


# -- V1 -----------------------------------------------------------------------

def stage_v1(args) -> dict:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    natural, shuffled = [], []
    for seed in seeds:
        n_dir = os.path.join(REPO_ROOT, "runs", "v1", f"natural_s{seed}")
        s_dir = os.path.join(REPO_ROOT, "runs", "v1", f"shuffled_s{seed}")
        run_train(write_seed_config(os.path.join(REPO_ROOT, "configs", "v1_natural.yaml"),
                                    seed, n_dir), n_dir)
        run_train(write_seed_config(os.path.join(REPO_ROOT, "configs", "v1_shuffled.yaml"),
                                    seed, s_dir), s_dir)
        natural.append(n_dir)
        shuffled.append(s_dir)
        if args.noise:
            nz = os.path.join(REPO_ROOT, "runs", "v1", f"noise_s{seed}")
            run_train(write_seed_config(os.path.join(REPO_ROOT, "configs", "v1_noise.yaml"),
                                        seed, nz), nz)

    report = compare_v1(natural, shuffled)
    report["natural_dirs"] = natural
    report["shuffled_dirs"] = shuffled
    report["pass"] = bool(report["pass_clustered"] and report["pass_oscillation"])
    if args.plot:
        for d in natural + shuffled:
            maybe_plot(d)
    print("\n[v1] 到达结构性判定:")
    print(f"    natural 游程/置换p: {report['natural_runs']} / {report['natural_perm_p']}")
    print(f"    平均块长比 natural/shuffled = {report['block_ratio']:.2f}（判据 ≥2）")
    print(f"    震荡幅度比 natural/shuffled = {report['amp_ratio']:.2f}（判据 ≥2）")
    print(f"    遮蔽比 M = {report['natural_masking_ratio']:.3f}（≪1 = 平均指标失明）")
    print(f"    pass = {report['pass']}  "
          f"(clustered={report['pass_clustered']}, oscillation={report['pass_oscillation']})")
    return report


# -- V2 -----------------------------------------------------------------------

def stage_v2(args) -> dict:
    v1_root = os.path.join(REPO_ROOT, "runs", "v1")
    dirs = sorted(d for d in os.listdir(v1_root) if d.startswith("natural_s")) \
        if os.path.isdir(v1_root) else []
    if not dirs:
        print("[v2] 找不到 runs/v1/natural_* —— 先跑 --stage v1")
        return {"error": "no v1 runs"}
    reports = {}
    for d in dirs:
        run_dir = os.path.join(v1_root, d)
        if not os.path.exists(os.path.join(run_dir, "delta_probes.jsonl")):
            print(f"[v2] {run_dir} 无 delta_probes.jsonl（probe_on_apply 未开？）跳过")
            continue
        reports[d] = analyze_vector(run_dir, n_boot=args.n_boot)
        print(f"\n[v2] {d}:")
        print(f"    事件数 = {reports[d]['n_events']}")
        print(f"    Spearman(τ,|CI|) = {reports[d]['tau_ci_spearman']:.3f} "
              f"(判据 |ρ|<0.15)")
        print(f"    Spearman(‖d‖,|CI|) = {reports[d]['drift_ci_spearman']:.3f} "
              f"boot95 {reports[d]['drift_ci_boot95']} (判据 >0)")
        print(f"    同τ反方向事件对 = {reports[d]['same_tau_pairs']} "
              f"CI差均值 {reports[d]['same_tau_ci_diff_mean']:.4f} "
              f"boot95 {reports[d]['same_tau_ci_diff_boot95']}")
        print(f"    离开期内容分组 p = {reports[d]['mw_other_vs_own']['p']:.4f}")
        print(f"    pass = decoupling:{reports[d]['pass_decoupling']} "
              f"same_tau:{reports[d]['pass_same_tau']} group:{reports[d]['pass_group']}")
    reports["pass"] = bool(reports and all(
        v.get("pass_decoupling") and v.get("pass_same_tau")
        for k, v in reports.items() if k != "pass"))
    return reports


# -- V3 -----------------------------------------------------------------------

def analyze_killer(out_root: str, n_mid: tuple = (2, 3, 4, 5, 6)) -> dict:
    """跨顺序条件比较：中途态离散度 vs 终点重合度（数据 = delta_probes.jsonl）。"""
    orders = {}
    for name in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, name)
        path = os.path.join(d, "delta_probes.jsonl")
        if os.path.isdir(d) and os.path.exists(path):
            rows = read_jsonl(path)
            # state_k = 施加完 k 条后的状态（order_index=k-1 的 loss_after）
            states: dict[int, dict[str, float]] = {}
            for r in rows:
                states[r["order_index"] + 1] = r["loss_after"]
            orders[name] = states
    if len(orders) < 2:
        return {"error": f"顺序条件不足（{list(orders)}）——killer 未跑全？"}
    n_steps = max(max(s) for s in orders.values())
    tasks = list(next(iter(orders.values())).values())[0].keys() if orders else []

    def spread(step: int) -> float:
        """同一步数跨顺序条件的逐任务极差均值。"""
        ranges = []
        for t in tasks:
            vals = [orders[o][step][t] for o in orders
                    if step in orders[o] and t in orders[o][step]]
            if len(vals) >= 2:
                ranges.append(max(vals) - min(vals))
        return sum(ranges) / len(ranges) if ranges else 0.0

    mid = sum(spread(k) for k in n_mid if k <= n_steps) / len([k for k in n_mid if k <= n_steps])
    endpoint = spread(n_steps)
    return {
        "orders": sorted(orders),
        "n_steps": n_steps,
        "mid_spread_mean": mid,
        "endpoint_spread": endpoint,
        "ratio": mid / max(endpoint, 1e-9),
        # 判据：中途态离散显著大于终点重合残差（纯加法理论：终点应逐位重合）
        "pass": bool(mid > 10 * max(endpoint, 1e-9) and mid > 1e-4),
    }


def stage_v3(args) -> dict:
    out_root = os.path.join(REPO_ROOT, "runs", "v3", "killer")
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "scripts.run_killer",
           "--config", args.killer_config,
           "--orders", args.killer_orders,
           "--out", out_root]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    report = analyze_killer(out_root)
    print("\n[v3] killer 判定（冻结 delta、唯一自由度=顺序）:")
    if "error" not in report:
        print(f"    顺序条件 = {report['orders']}")
        print(f"    中途态平均离散 = {report['mid_spread_mean']:.5f}")
        print(f"    终点重合残差   = {report['endpoint_spread']:.2e}")
        print(f"    比值 = {report['ratio']:.0f}（判据 ≥10）")
        print(f"    pass = {report['pass']}")
    else:
        print(f"    {report['error']}")
    return report


# -- 主流程 --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V 系列前置实验一键驱动")
    ap.add_argument("--stage", default="all",
                    choices=["all", "v0", "v1", "v2", "v3", "summary"])
    ap.add_argument("--seeds", default="42", help="逗号分隔种子（v1 用）")
    ap.add_argument("--noise", action="store_true", help="v1 附加硬件噪声臂")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--killer-config", default="configs/killer/base_tiny.yaml")
    ap.add_argument("--killer-orders", default="clustered,alternating,random")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO_ROOT, "runs"), exist_ok=True)
    stages = ["v0", "v1", "v2", "v3"] if args.stage == "all" else [args.stage]
    all_reports: dict[str, dict] = {}
    for s in stages:
        print(f"\n{'=' * 70}\n========== 阶段 {s} ==========\n{'=' * 70}")
        fn = {"v0": stage_v0, "v1": stage_v1, "v2": stage_v2, "v3": stage_v3}[s]
        try:
            all_reports[s] = fn(args)
        except subprocess.CalledProcessError as e:
            print(f"[{s}] 子进程失败（exit={e.returncode}），中止后续阶段")
            all_reports[s] = {"error": f"exit {e.returncode}"}
            break
        out = os.path.join(REPO_ROOT, "runs", f"{s}_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_reports[s], f, ensure_ascii=False, indent=1, default=str)
        print(f"[{s}] 报告已写入 {out}")

    # go/no-go 汇总
    print(f"\n{'=' * 70}\n========== V 系列 go/no-go 汇总 ==========\n{'=' * 70}")
    rows = [
        ("V0a 确定性", "smoke_determinism" in all_reports.get("v0", {}),
         all_reports.get("v0", {}).get("smoke_determinism")),
        ("V0b 标定比≥1.5", "calibration" in all_reports.get("v0", {}),
         (all_reports.get("v0", {}).get("calibration") or {}).get("pass_ratio_ge_1p5")),
        ("V1 到达结构性", "pass" in all_reports.get("v1", {}) or "error" not in all_reports.get("v1", {}),
         all_reports.get("v1", {}).get("pass")),
        ("V2 τ 脱钩+同τ不同命", "v2" in all_reports,
         all_reports.get("v2", {}).get("pass")),
        ("V3 killer 直接层", "v3" in all_reports,
         all_reports.get("v3", {}).get("pass")),
    ]
    print(f"{'阶段':<24}{'执行':<8}{'判定':<8}")
    for name, executed, ok in rows:
        print(f"{name:<24}{str(executed):<8}"
              f"{'PASS' if ok else 'FAIL' if ok is not None else '--':<8}")
    print("\n全 PASS → 数据/时间线/机制链验证完成，可以上 E1 master run（configs/e1_natural.yaml）")


if __name__ == "__main__":
    main()
