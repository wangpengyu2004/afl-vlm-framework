"""fedcompass：FedCompass（arXiv:2309.14675）的策略层实现——时间变化的 staleness 阈值。

原文机制（cross-silo）：服务器维护一个随时间增长的 staleness 阈值 θ(t)，
只有 staleness ≤ θ 的更新获准进入聚合；等待时长被 θ 的增长上界封顶。
客户端侧的"超量供给 ρ"属于执行调度（让闲时算力提前跑后续轮次），与聚合规则
正交，不在策略层模拟——本策略忠实实现的是**阈值准入 + 整批施加**这一半：

    θ(t) = min(theta_max, theta0 + theta_growth × (t − t_last_agg))
    合格集 = {u ∈ buffer : staleness(u) ≤ θ(t)}
    |合格集| ≥ min_count 时按到达序整批施加（权重 1.0）

t 在 wallclock 模式是真实秒、virtual 模式是虚拟秒（planner 传入的 sim_time），
theta_growth 的单位是"版本数/秒"。θ 封顶 theta_max 保证慢更新最终都能进
（staleness 超过 theta_max 的更新等到 θ 增长到位，或 run 结束时 on_finish 兜底）。
"""

from __future__ import annotations

from .base import AggregationPolicy, ServerState, UpdateRecord


class FedCompassPolicy(AggregationPolicy):
    name = "fedcompass"

    def __init__(self, theta0: float = 1.0, theta_growth: float = 0.0025,
                 theta_max: float = 6.0, min_count: int = 3):
        if theta0 <= 0 or theta_growth < 0 or theta_max < theta0 or min_count <= 0:
            raise ValueError("fedcompass 参数非法：需 theta0>0, growth≥0, theta_max≥theta0, min_count>0")
        self.theta0 = theta0
        self.growth = theta_growth
        self.theta_max = theta_max
        self.min_count = min_count
        self._t_last = None       # 上次聚合的时刻（None = 尚未聚合过）

    def _threshold(self, state: ServerState) -> float:
        if self._t_last is None:
            return self.theta0
        elapsed = max(0.0, state.sim_time - self._t_last)
        return min(self.theta_max, self.theta0 + self.growth * elapsed)

    def on_arrivals(self, buffer: list[UpdateRecord], state: ServerState
                    ) -> list[tuple[UpdateRecord, float]]:
        theta = self._threshold(state)
        eligible = [
            u for u in sorted(buffer, key=lambda x: x.seq_no)
            if max(0, state.global_version - u.version_trained_on) <= theta
        ]
        if len(eligible) < self.min_count:
            return []
        self._t_last = state.sim_time
        return [(u, 1.0) for u in eligible]
