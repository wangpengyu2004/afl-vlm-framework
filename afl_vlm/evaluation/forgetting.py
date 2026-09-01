"""遗忘分析：从 evals.jsonl 提取各任务能力曲线与"峰值 vs 终值"遗忘摘要。

能力指标优先 match_rate（生成匹配，越高越好）；没有时用 eval_loss 取负
（统一成"越高越好"），便于跨任务、跨顺序配置比较。
"""

from __future__ import annotations

from collections import defaultdict

from ..logging_utils import read_jsonl


def capability_metric(row: dict) -> float:
    if "match_rate" in row:
        return float(row["match_rate"])
    return -float(row["eval_loss"])


def capability_series(rows: list[dict]) -> dict[str, list[dict]]:
    """{task: [{tag, global_version, capability}...]} 按全局版本排序。"""
    series: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        series[r["task"]].append({
            "tag": r.get("tag"),
            "global_version": r.get("global_version"),
            "capability": capability_metric(r),
        })
    for t in series:
        series[t].sort(key=lambda x: (x["global_version"] is None, x["global_version"] or 0))
    return dict(series)


def forgetting_summary(rows: list[dict]) -> dict[str, dict]:
    """{task: {peak, final, drop, drop_ratio}}。drop = 峰值 − 终值（>0 即发生遗忘）。"""
    series = capability_series(rows)
    out = {}
    for task, pts in series.items():
        caps = [p["capability"] for p in pts]
        peak, final = max(caps), caps[-1]
        out[task] = {
            "peak": peak,
            "final": final,
            "drop": peak - final,
            "drop_ratio": (peak - final) / abs(peak) if peak else 0.0,
        }
    return out
