"""immediate_fifo：经典 FedAsync——每个到达立即按到达序聚合，权重 1。"""

from __future__ import annotations

from .base import AggregationPolicy, UpdateRecord, ServerState


class ImmediateFifoPolicy(AggregationPolicy):
    name = "immediate_fifo"

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        ready = sorted(buffer, key=lambda x: x.seq_no)
        return [(u, 1.0) for u in ready]
