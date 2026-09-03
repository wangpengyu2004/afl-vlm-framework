"""fedasync：FedAsync（Xie et al., 2019, arXiv:1903.03934）标准形态。

每个到达**立即**施加（不缓冲），权重 = staleness 函数 exp(-λ·τ)（带下限截断）——
与 immediate_fifo（等权立即施加）的区别只在权重，与 staleness_weighted（先攒 K 个
再降权）的区别在"立即 vs 缓冲"。三条异步经典基线在框架内的对应：

    FedAsync   → fedasync（立即 + 降权）/ immediate_fifo（立即 + 等权）
    FedBuff    → fedbuff（缓冲 K 个整批）
    FedCompass → fedcompass（时间变化阈值准入）
"""

from __future__ import annotations

import math

from .base import AggregationPolicy, ServerState, UpdateRecord


class FedAsyncPolicy(AggregationPolicy):
    name = "fedasync"

    def __init__(self, staleness_lambda: float = 0.3, min_weight: float = 0.1):
        self.lam = staleness_lambda
        self.min_weight = min_weight

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        out = []
        for u in sorted(buffer, key=lambda x: x.seq_no):
            staleness = max(0, state.global_version - u.version_trained_on)
            w = max(self.min_weight, math.exp(-self.lam * staleness))
            out.append((u, w))
        return out
