"""sync_avg：同步 FedAvg 锚——等齐全部客户端的更新后一次性平均。

异步框架里的"同步上界/下界参照"：攒齐 num_clients_total 个（每客户端各至少
一条）才整批施加，权重恒 1.0，施加顺序 = 到达序。配 prox_mu > 0 即 FedProx
（近端项在客户端侧，见 client._train_one_round）。

实现说明：planner 传入的 ServerState.clients_done 恒为 0（计划阶段无人完成），
故只按"buffer 覆盖了全部客户端"判断；结束阶段的残余由 on_finish 兜底。
"""

from __future__ import annotations

from .base import AggregationPolicy, ServerState, UpdateRecord


class SyncAvgPolicy(AggregationPolicy):
    name = "sync_avg"

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        contributors = {u.client_id for u in buffer}
        if len(contributors) < state.num_clients_total:
            return []                      # 没等齐 → 继续攒（同步屏障）
        return [(u, 1.0) for u in sorted(buffer, key=lambda x: x.seq_no)]
