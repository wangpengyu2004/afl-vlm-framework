"""聚合策略注册表：配置 aggregation.name → 策略实例。

新增策略三步：
    1. 在 aggregation/ 下新建 policy.py，实现 AggregationPolicy
       （核心：on_arrivals 返回 [要聚合的更新列表]，**列表顺序 = 施加顺序**）
    2. 在此处注册 name → 工厂
    3. 在 config.validate_config 的 known_agg 里加名字
"""

from __future__ import annotations

from .base import AggregationPolicy, ServerState, UpdateRecord
from .immediate_fifo import ImmediateFifoPolicy
from .fedbuff import FedBuffPolicy
from .staleness_weighted import StalenessWeightedPolicy
from .fixed_order import FixedOrderPolicy


def build_policy(agg_cfg) -> AggregationPolicy:
    name = agg_cfg.name
    if name == "immediate_fifo":
        return ImmediateFifoPolicy()
    if name == "fedbuff":
        return FedBuffPolicy(K=agg_cfg.K)
    if name == "staleness_weighted":
        return StalenessWeightedPolicy(
            K=agg_cfg.K,
            staleness_lambda=agg_cfg.staleness_lambda,
            min_weight=agg_cfg.min_weight,
        )
    if name == "fixed_order":
        return FixedOrderPolicy(batch_size=agg_cfg.batch_size, order=agg_cfg.order)
    raise ValueError(f"未注册的聚合策略: {name}")


__all__ = [
    "AggregationPolicy", "UpdateRecord", "ServerState",
    "build_policy",
    "ImmediateFifoPolicy", "FedBuffPolicy", "StalenessWeightedPolicy", "FixedOrderPolicy",
]
