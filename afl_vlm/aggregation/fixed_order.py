"""fixed_order：killer experiment 执行器。

攒够 batch_size 个更新 → 取最早的 batch_size 个 → 按 order.mode **重排后**顺序施加。
所有权重恒为 1.0：保证批次内唯一自由度是**顺序**，与 staleness_weighted 正交。

order.mode:
    random      带种子随机排列
    clustered   同任务连续（任务组按最早到达排序，组内按到达序）——论文里最坏情形
    alternating 各任务轮转交错（最大化交替）
    blocked     按任务名排序成块（确定性分块，类 CL blocked 训练）
    explicit    按 order.explicit 里的 client_id 顺序，未列出的按到达序垫后
"""

from __future__ import annotations

from collections import defaultdict

from ..config import stable_seed
from .base import AggregationPolicy, UpdateRecord, ServerState


class FixedOrderPolicy(AggregationPolicy):
    name = "fixed_order"

    def __init__(self, batch_size: int = 8, order: dict | None = None):
        self.batch_size = batch_size
        order = order or {}
        self.mode = order.get("mode", "random")
        self.seed = order.get("seed", 0)
        self.explicit = order.get("explicit", [])
        if self.mode not in {"random", "clustered", "alternating", "blocked", "explicit"}:
            raise ValueError(f"未知 order.mode: {self.mode}")

    # -- AggregationPolicy ---------------------------------------------------

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        if len(buffer) < self.batch_size:
            return []
        chosen = sorted(buffer, key=lambda x: x.seq_no)[: self.batch_size]
        return [(u, 1.0) for u in self._reorder(chosen)]

    def on_finish(self, buffer: list[UpdateRecord], state: ServerState
                  ) -> list[tuple[UpdateRecord, float]]:
        rest = sorted(buffer, key=lambda x: x.seq_no)
        return [(u, 1.0) for u in self._reorder(rest)]

    # -- 顺序生成器（也供未来的 order-aware 调度方法复用） ---------------------

    def _reorder(self, updates: list[UpdateRecord]) -> list[UpdateRecord]:
        if self.mode == "random":
            import random as _random
            rng = _random.Random(stable_seed("fixed_order", self.seed, len(updates)))
            return rng.sample(updates, k=len(updates))

        if self.mode in {"clustered", "blocked"}:
            groups: dict[str, list[UpdateRecord]] = defaultdict(list)
            for u in updates:
                groups[u.task].append(u)
            for g in groups.values():
                g.sort(key=lambda x: x.seq_no)
            if self.mode == "clustered":
                # 任务组按"最早到达"排序：先来的任务先整块写入
                task_order = sorted(groups.keys(), key=lambda t: min(u.seq_no for u in groups[t]))
            else:  # blocked：按任务名确定性排序
                task_order = sorted(groups.keys())
            return [u for t in task_order for u in groups[t]]

        if self.mode == "alternating":
            groups: dict[str, list[UpdateRecord]] = defaultdict(list)
            for u in updates:
                groups[u.task].append(u)
            for g in groups.values():
                g.sort(key=lambda x: x.seq_no)
            out, task_cycle = [], list(groups.keys())
            while any(groups[t] for t in task_cycle):
                for t in task_cycle:
                    if groups[t]:
                        out.append(groups[t].pop(0))
            return out

        # explicit：按 client_id 顺序
        rank = {cid: i for i, cid in enumerate(self.explicit)}
        return sorted(updates, key=lambda u: (rank.get(u.client_id, len(rank)), u.seq_no))
