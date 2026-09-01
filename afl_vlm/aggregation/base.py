"""聚合策略基类与数据结构。

研究主钩子：策略唯一职责是回答两个问题——
    1. 现在聚合哪些更新？
    2. 以什么**顺序**施加？（返回列表的顺序即施加顺序）

服务器保证按返回列表逐个施加（w ← w + lr·weight·Δ），因此：
    - re-weighting 类策略改 weight；
    - 顺序研究类策略改**列表顺序**；
    - 两者正交，这正是论文要拆分的两个维度。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class UpdateRecord:
    """一次客户端上传。delta 只含可训练（LoRA）参数，CPU float32。"""
    seq_no: int                     # 全局递增序号（buffer 内稳定排序用）
    client_id: str
    task: str
    round: int                      # 客户端本地第几轮训练
    version_trained_on: int         # 训练时的全局版本
    t_train_start: float
    t_train_end: float
    t_arrival: float
    net_delay: float
    train_seconds: float            # 本地训练真实耗时（含慢设备补差）
    tau: int = 0                    # staleness 模式：以聚合步数计的延迟（0=wallclock 模式）
    hold_until_version: int = 0     # staleness 模式：版本推进到该值才进入 buffer
    download_lag: int = 0           # 训练所用版本落后最新版的步数（下载侧延迟）
    delta: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ServerState:
    """传给策略的只读快照。"""
    global_version: int
    sim_time: float
    num_clients_total: int
    clients_done: int


class AggregationPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        """到达事件后调用。返回本次要聚合的 (更新, 权重) 列表，**列表顺序=施加顺序**；
        返回空列表 = 攒着不聚合。被返回的更新会从 buffer 移除。"""

    def on_finish(self, buffer: list[UpdateRecord], state: ServerState
                  ) -> list[tuple[UpdateRecord, float]]:
        """所有客户端完成后的最终 flush；默认聚合剩余全部（按到达序）。"""
        return [(u, 1.0) for u in sorted(buffer, key=lambda x: x.seq_no)]
