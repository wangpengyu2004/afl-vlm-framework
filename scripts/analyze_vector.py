"""V2 矢量 staleness 分析：delta 级"伤害方向"与 τ 的脱钩检验。

数据源（probe_on_apply: true 的 run）：
    delta_probes.jsonl  每条 delta 施加前后的三任务探针 loss
    aggregations.jsonl  施加日志（用于重建客户端"离开期"全局模型被谁写过）
    arrivals.jsonl      到达日志（τ 的另一来源）

每个 delta 事件计算：
    improvement_t = loss_before[t] − loss_after[t]   （>0 = 该任务被改善）
    CI（纠偏指数）= improvement_own − mean(improvement_others)  （>0 = 纠偏型）
    τ = staleness（版本数）
    离开期构成 = (version_trained_on, version_before] 窗口内各任务被施加的占比
        → 漂移方向 d = argmax(count_k/n − 1/3)，幅度 ‖d‖ = 各任务偏差绝对值之和

检验（对应 experiment-design E2 的预注册判据）：
    1. 脱钩：Spearman(τ, |CI|) ≈ 0（|ρ|<0.15）而 Spearman(‖d‖, |CI|) 显著为正
       （bootstrap 95% CI 不含 0）
    2. 同 τ 不同命：同 staleness、离开期方向相反的事件对 → CI 的置换检验
    3. 离开期内容分组：他任务占比 ≥0.7 vs 同任务占比 ≥0.7 → CI 的 Mann-Whitney
       （预测：离开期被异任务写满 → delta 纠偏 → CI 更高）

统计全部手写（无 scipy）：Spearman=秩相关、MW U 正态近似（并列取平均秩）、
bootstrap=固定种子重采样。

用法:
    python -m scripts.analyze_vector --dir runs/v1/natural --report out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from afl_vlm.logging_utils import read_jsonl  # noqa: E402


# -- 基础统计（手写实现） ------------------------------------------------------

def _rank(xs: list[float]) -> list[float]:
    """平均秩（并列取平均）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 5:
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


def bootstrap_rho_ci(xs: list[float], ys: list[float], n_boot: int = 1000,
                     seed: int = 42) -> tuple[float, float, float]:
    """(ρ, 95%CI 下界, 95%CI 上界)。"""
    rho = spearman(xs, ys)
    rng = random.Random(seed)
    n = len(xs)
    rhos = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if r == r:
            rhos.append(r)
    if not rhos:
        return rho, float("nan"), float("nan")
    rhos.sort()
    lo = rhos[int(0.025 * (len(rhos) - 1))]
    hi = rhos[int(0.975 * (len(rhos) - 1))]
    return rho, lo, hi


def mannwhitney_u(a: list[float], b: list[float]) -> dict:
    """MW U 检验（正态近似，含并列修正）。返回 U、z、双侧 p。"""
    a, b = [x for x in a if x == x], [x for x in b if x == x]
    if not a or not b:
        return {"U": float("nan"), "z": float("nan"), "p": float("nan")}
    combined = a + b
    ranks = _rank(combined)
    ra = sum(ranks[:len(a)])
    n1, n2 = len(a), len(b)
    u1 = ra - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2
    # 并列修正
    from collections import Counter
    ties = [c for c in Counter(combined).values() if c > 1]
    sigma_sq = (n1 * n2 / 12) * ((n1 + n2 + 1)
                                 - sum(c ** 3 - c for c in ties) / (n1 + n2) / (n1 + n2 - 1))
    if sigma_sq <= 0:
        return {"U": u, "z": 0.0, "p": 1.0}
    z = (u - mu) / math.sqrt(sigma_sq)
    p = math.erfc(abs(z) / math.sqrt(2))       # 双侧
    return {"U": u, "z": z, "p": p}


def perm_test_mean_diff(a: list[float], b: list[float], n_perm: int = 5000,
                        seed: int = 42) -> dict:
    """置换检验：H0 两组均值同。返回观测差与双侧 p。"""
    obs = (sum(a) / len(a)) - (sum(b) / len(b)) if a and b else float("nan")
    pool = list(a) + list(b)
    na = len(a)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = sum(pool[:na]) / na - sum(pool[na:]) / max(1, len(pool) - na)
        if abs(d) >= abs(obs):
            ge += 1
    return {"diff": obs, "perm_p": (ge + 1) / (n_perm + 1)}


# -- 事件构建 ------------------------------------------------------------------

def build_events(run_dir: str) -> list[dict]:
    probes = read_jsonl(os.path.join(run_dir, "delta_probes.jsonl"))
    aggs = read_jsonl(os.path.join(run_dir, "aggregations.jsonl"))
    # 施加日志按全局版本索引：版本 v 是谁写的
    writer_of_version = {r["global_version_after"]: r["task"] for r in aggs}
    events = []
    for row in probes:
        before, after = row["loss_before"], row["loss_after"]
        own = row["task"]
        others = [t for t in before if t != own]
        if not others:
            continue
        improvement = {t: before[t] - after[t] for t in before}
        ci = improvement[own] - sum(improvement[t] for t in others) / len(others)
        # 离开期构成：(version_trained_on, version_before] 内的施加
        counts = {t: 0 for t in before}
        for v in range(row["version_trained_on"] + 1, row["version_before"] + 1):
            t = writer_of_version.get(v)
            if t is not None and t in counts:
                counts[t] += 1
        n_win = sum(counts.values())
        shares = {t: (counts[t] / n_win if n_win else 0.0) for t in counts}
        dev = {t: shares[t] - 1 / len(shares) for t in shares}
        direction = max(dev, key=dev.get) if n_win else None
        events.append({
            "client": row["client"], "task": own, "round": row["round"],
            "tau": row["staleness"], "weight": row["weight"],
            "ci": ci, "abs_ci": abs(ci),
            "improvement_own": improvement[own],
            "n_absence": n_win,
            "absence_shares": shares,
            "drift_l1": sum(abs(d) for d in dev.values()),
            "drift_direction": direction,
            "other_share": n_win and 1 - shares[own],
        })
    return events


def analyze(run_dir: str, n_boot: int = 1000) -> dict:
    events = build_events(run_dir)
    if len(events) < 10:
        return {"run_dir": run_dir, "n_events": len(events),
                "error": "事件太少（probe_on_apply 未开启或 run 太短）"}

    taus = [e["tau"] for e in events]
    abs_cis = [e["abs_ci"] for e in events]
    drifts = [e["drift_l1"] for e in events]
    cis = [e["ci"] for e in events]

    r_tau, lo_tau, hi_tau = bootstrap_rho_ci(taus, abs_cis, n_boot)
    r_drift, lo_drift, hi_drift = bootstrap_rho_ci(drifts, abs_cis, n_boot)

    # 2. 同 τ 不同命：同 staleness 内方向相反的事件对
    by_tau: dict[int, list[dict]] = {}
    for e in events:
        by_tau.setdefault(e["tau"], []).append(e)
    pair_cis_diff = []
    n_pairs = 0
    for tau, group in by_tau.items():
        own_dir = [e for e in group if e["drift_direction"] == e["task"]]
        other_dir = [e for e in group if e["drift_direction"] not in (None, e["task"])]
        for a in own_dir:
            for b in other_dir:
                pair_cis_diff.append(a["ci"] - b["ci"])
                n_pairs += 1
    pair_mean = sum(pair_cis_diff) / len(pair_cis_diff) if pair_cis_diff else float("nan")
    rng = random.Random(7)
    if pair_cis_diff:
        boot = []
        for _ in range(n_boot):
            s = [pair_cis_diff[rng.randrange(len(pair_cis_diff))]
                 for _ in range(len(pair_cis_diff))]
            boot.append(sum(s) / len(s))
        boot.sort()
        pair_ci = (boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))])
    else:
        pair_ci = (float("nan"), float("nan"))

    # 3. 离开期内容分组
    g_other = [e["ci"] for e in events
               if e["n_absence"] >= 3 and e["absence_shares"][e["task"]] <= 0.3]
    g_own = [e["ci"] for e in events
             if e["n_absence"] >= 3 and e["absence_shares"][e["task"]] >= 0.7]
    mw = mannwhitney_u(g_other, g_own)

    return {
        "run_dir": run_dir,
        "n_events": len(events),
        "tau_ci_spearman": r_tau, "tau_ci_boot95": [lo_tau, hi_tau],
        "drift_ci_spearman": r_drift, "drift_ci_boot95": [lo_drift, hi_drift],
        "same_tau_pairs": n_pairs,
        "same_tau_ci_diff_mean": pair_mean,
        "same_tau_ci_diff_boot95": list(pair_ci),
        "group_absence_other": {"n": len(g_other), "ci_mean": sum(g_other) / len(g_other) if g_other else float("nan")},
        "group_absence_own": {"n": len(g_own), "ci_mean": sum(g_own) / len(g_own) if g_own else float("nan")},
        "mw_other_vs_own": mw,
        # 预注册判据
        "pass_decoupling": bool(abs(r_tau) < 0.15 and r_drift > 0 and lo_drift > 0),
        "pass_same_tau": bool(n_pairs >= 3 and pair_ci[0] > 0),
        "pass_group": bool(mw["p"] == mw["p"] and mw["p"] < 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run 输出目录（需 delta_probes.jsonl）")
    ap.add_argument("--report", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    result = analyze(args.dir, args.n_boot)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
