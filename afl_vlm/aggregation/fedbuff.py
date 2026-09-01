"""fedbuff：缓冲 K 个更新后一次性 flush（FedBuff, AISTATS 2022）。

flush 顺序 = 到达顺序（seq_no）。注意：缓冲构成本身就是"顺序第一层"
（composition），本策略不对其做任何控制——这正是原论文没分析的部分。
"""

from __future__ import annotations

from .base import AggregationPolicy, UpdateRecord, ServerState


class FedBuffPolicy(AggregationPolicy):
    name = "fedbuff"

    def __init__(self, K: int = 4):
        self.K = K

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        if len(buffer) < self.K:
            return []
        ready = sorted(buffer, key=lambda x: x.seq_no)[: self.K]
        return [(u, 1.0) for u in ready]
