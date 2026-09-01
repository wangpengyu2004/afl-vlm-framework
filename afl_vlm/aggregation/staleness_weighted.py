"""staleness_weighted：陈旧度降权重（re-weighting 家族代表，FedASMU/AsyncFedED 风格）。

weight = max(min_weight, exp(-λ · staleness))，staleness = 当前全局版本 − 更新训练时版本。

意义：这是论文论证的"re-weighting 天花板"基线——它只能对更新降权，
永远改变不了施加顺序。与 fixed_order 对比即可分离两个维度。
"""

from __future__ import annotations

import math

from .base import AggregationPolicy, UpdateRecord, ServerState


class StalenessWeightedPolicy(AggregationPolicy):
    name = "staleness_weighted"

    def __init__(self, K: int = 4, staleness_lambda: float = 0.3, min_weight: float = 0.1):
        self.K = K
        self.lam = staleness_lambda
        self.min_weight = min_weight

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        if len(buffer) < self.K:
            return []
        ready = sorted(buffer, key=lambda x: x.seq_no)[: self.K]
        out = []
        for u in ready:
            staleness = max(0, state.global_version - u.version_trained_on)
            w = max(self.min_weight, math.exp(-self.lam * staleness))
            out.append((u, w))
        return out
