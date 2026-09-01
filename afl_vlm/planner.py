"""虚拟时钟调度器（timing: virtual）——"计划–执行"两阶段的阶段1。

阶段1（本模块，秒级、零训练开销）：按虚拟时钟做离散事件模拟，产出完整调度表 TrainPlan：
    - 每个客户端每轮在哪个虚拟时刻、同步哪个全局版本（sync_version）
    - 每次上传的虚拟到达时刻与到达顺序
    - 聚合策略在"只有元数据、没有 delta"的计划记录上运行，产出全局施加顺序与权重
到达时间线 = start_offset + train_seconds/speed_factor + net_delay（+ staleness 模式
的 τ 持留），是 config + seed 的纯函数——与 GPU 卡数、线程竞争完全无关，
跨运行严格可复现。这就是"计划"：策略行为先在计划里完整确定，阶段2 只负责回放。

阶段2（server/client 按 TrainPlan 执行）：客户端按计划同步指定版本快照、真实训练
产出 delta；服务器按计划的施加顺序回放，delta 未产出就等——只拉伸墙钟、绝不重排。
排队等卡、线程竞争等执行细节对时间线零影响（wallclock 模式的失真来源被根除）。

可行性/语义要点：
    - sync_version 只引用更早施加出的版本 → 阶段2 无死锁（因果链版本严格递增）；
    - delta 早于/晚于计划虚拟时刻产出都无影响（服务器只认计划的顺序）；
    - train_seconds 建议取 ≥ 单卡真实单轮耗时（可先用 wallclock 跑一轮，从
      arrivals.jsonl 的 train_seconds 字段标定）；取小了也只是服务器等待，不出错；
    - time_scale（真实睡眠缩放）在 virtual 模式无意义，被忽略；
    - speed_factor 在 virtual 模式允许 >1（虚拟时间可压缩，无真实算力约束）。
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass

from .aggregation import ServerState, UpdateRecord, build_policy
from .config import ExperimentConfig, stable_seed


@dataclass
class PlannedApply:
    """计划中的一次施加：施加哪个更新（seq_no）、什么权重、第几批、批内序号。"""
    seq_no: int
    weight: float
    batch: int
    order_index: int
    batch_size: int
    t_apply: float          # 虚拟施加时刻（= 触发本次批次的到达时刻）


class TrainPlan:
    """完整虚拟调度表。rounds 按 (客户端, 轮次) 排列；applies 按全局施加顺序排列。"""

    def __init__(self) -> None:
        self.rounds: list[UpdateRecord] = []        # 元数据计划（delta 恒为空 dict）
        self.applies: list[PlannedApply] = []
        self.arrival_order: list[int] = []          # seq_no 的到达顺序
        self.total_clients: int = 0

    def rounds_of(self, client_id: str) -> list[UpdateRecord]:
        return [r for r in self.rounds if r.client_id == client_id]

    def sync_consumers(self) -> dict[int, int]:
        """每个版本被计划同步的次数——管理器据此保留版本快照、用完即裁剪（控内存）。"""
        out: dict[int, int] = {}
        for r in self.rounds:
            out[r.version_trained_on] = out.get(r.version_trained_on, 0) + 1
        return out

    def to_dict(self) -> dict:
        """完整序列化（写 plan.json）：调度表是时间线的权威记录，供复现/分析。"""
        return {
            "rounds": [{
                "seq_no": r.seq_no, "client": r.client_id, "task": r.task,
                "round": r.round, "version_trained_on": r.version_trained_on,
                "t_train_start": r.t_train_start, "t_train_end": r.t_train_end,
                "t_arrival": r.t_arrival, "net_delay": r.net_delay,
                "train_seconds": r.train_seconds, "tau": r.tau,
                "hold_until_version": r.hold_until_version,
                "download_lag": r.download_lag,
            } for r in self.rounds],
            "applies": [{
                "seq_no": a.seq_no, "weight": a.weight, "batch": a.batch,
                "order_index": a.order_index, "batch_size": a.batch_size,
                "t_apply": a.t_apply,
            } for a in self.applies],
            "arrival_order": list(self.arrival_order),
        }


def build_plan(cfg: ExperimentConfig) -> TrainPlan:
    """阶段1：虚拟时钟离散事件模拟。

    事件三类（按虚拟时刻出堆，同刻按入堆顺序）：
        round  : 客户端开始第 r 轮 → 采样 download_lag、定下 sync_version，
                 排出训练结束事件（虚拟时长 = train_seconds / speed_factor）
        end    : 训练结束 → 采样 tau / net_delay，构造计划记录，排出到达事件；
                 下一轮在到达时刻立即开始（客户端提交后马上回来同步，与
                 wallclock 客户端行为一致）
        arrive : 上传到达 → 与运行时服务器完全相同的 admit/持留/放行语义，
                 策略在计划记录上产出施加列表
    模拟结束：放行仍在持留的更新 + 策略 on_finish 最终 flush（与运行时收尾一致）。
    """
    policy = build_policy(cfg.server.aggregation)
    clients = {c["id"]: c for c in cfg.clients.expand()}
    total = len(clients)
    staleness_mode = cfg.clients.delay_mode == "staleness"

    # 与 wallclock 客户端相同的独立随机流（同 seed 时两种模式的延迟样本一致）
    rng = {cid: {
        "net": random.Random(stable_seed(cfg.seed, cid, "net")),
        "tau": random.Random(stable_seed(cfg.seed, cid, "tau")),
        "lag": random.Random(stable_seed(cfg.seed, cid, "lag")),
    } for cid in clients}

    plan = TrainPlan()
    plan.total_clients = total
    seq = itertools.count()
    tie = itertools.count()
    events: list[tuple[float, int, str, object]] = []
    version = 0
    batch = 0
    buffer: list[UpdateRecord] = []
    pending: list[UpdateRecord] = []        # staleness 持留（与服务器同语义）

    def state(t: float) -> ServerState:
        return ServerState(version, t, total, 0)

    def apply_list(ordered: list[tuple[UpdateRecord, float]], t: float) -> None:
        nonlocal version, batch
        if not ordered:
            return
        applied: set[int] = set()
        for order_index, (u, w) in enumerate(ordered):
            plan.applies.append(PlannedApply(
                seq_no=u.seq_no, weight=w, batch=batch, order_index=order_index,
                batch_size=len(ordered), t_apply=t))
            version += 1                    # 计划施加瞬时完成（版本推进即生效）
            applied.add(u.seq_no)
        buffer[:] = [u for u in buffer if u.seq_no not in applied]
        batch += 1

    def admit(u: UpdateRecord, t: float) -> None:
        buffer.append(u)
        apply_list(policy.on_arrivals(buffer, state(t)), t)

    def promote(t: float) -> None:
        """版本推进后放行到期的持留更新，循环到稳定（镜像服务器 _promote_pending）。"""
        nonlocal pending
        while pending:
            ready = [r for r in pending if r.hold_until_version <= version]
            if not ready:
                break
            pending = [r for r in pending if r.hold_until_version > version]
            for r in sorted(ready, key=lambda x: x.seq_no):
                buffer.append(r)
            apply_list(policy.on_arrivals(buffer, state(t)), t)

    def on_round(cid: str, r: int, t: float) -> None:
        c = clients[cid]
        dl = int(c["download_lag"].sample(rng[cid]["lag"]))
        sync_version = max(0, version - dl)         # 下载侧延迟：同步到旧版
        t_end = t + c["train_seconds"] / c["speed_factor"]
        heapq.heappush(events, (t_end, next(tie), "end", (cid, r, sync_version, t, dl)))

    def on_end(cid: str, r: int, sync_version: int, t_start: float, dl: int, t_end: float) -> None:
        c = clients[cid]
        if staleness_mode:
            tau = int(c["staleness_tau"].sample(rng[cid]["tau"]))
            net_delay = 0.0
            t_arrival = t_end                       # staleness 模式：训练完立即提交
        else:
            tau = 0
            net_delay = float(c["net_delay"].sample(rng[cid]["net"]))
            t_arrival = t_end + net_delay
        rec = UpdateRecord(
            seq_no=next(seq), client_id=cid, task=c["task"], round=r,
            version_trained_on=sync_version,
            t_train_start=t_start, t_train_end=t_end, t_arrival=t_arrival,
            net_delay=net_delay, train_seconds=t_end - t_start,
            tau=tau, hold_until_version=sync_version + tau,
            download_lag=dl,
        )
        plan.rounds.append(rec)
        heapq.heappush(events, (t_arrival, next(tie), "arrive", rec))
        if r + 1 < c["num_rounds"]:                 # 提交后立即回来开下一轮
            heapq.heappush(events, (t_arrival, next(tie), "round", (cid, r + 1)))

    def on_arrive(u: UpdateRecord, t: float) -> None:
        plan.arrival_order.append(u.seq_no)
        if u.hold_until_version > version:          # τ 未到期 → 进持留池
            pending.append(u)
        else:
            admit(u, t)
        promote(t)

    for cid in clients:                             # 起始轮按客户端序（同刻稳定）
        heapq.heappush(events, (clients[cid]["start_offset"], next(tie), "round", (cid, 0)))

    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == "round":
            cid, r = payload
            on_round(cid, r, t)
        elif kind == "end":
            cid, r, sv, ts, dl = payload
            on_end(cid, r, sv, ts, dl, t)
        else:
            on_arrive(payload, t)

    # 模拟结束：放行仍在持留的更新 + 最终 flush（镜像服务器 _flush_pending/on_finish）
    t_sim = max((r.t_arrival for r in plan.rounds), default=0.0)
    for r in sorted(pending, key=lambda x: x.seq_no):
        buffer.append(r)
    pending.clear()
    apply_list(policy.on_finish(buffer, state(t_sim)), t_sim)
    return plan


def print_plan_schedule(plan: TrainPlan, limit: int = 20) -> None:
    """计划摘要（--dry-run 也用它）：到达顺序表 + 施加批次统计。"""
    by_seq = {r.seq_no: r for r in plan.rounds}
    n = min(limit, len(plan.arrival_order))
    print(f"[plan] 到达顺序（前 {n}/{len(plan.arrival_order)} 条）：")
    print(f"{'seq':<5}{'client':<10}{'task':<14}{'round':<7}{'sync_v':<8}{'t_arrival':<12}")
    for s in plan.arrival_order[:limit]:
        r = by_seq[s]
        print(f"{s:<5}{r.client_id:<10}{r.task:<14}{r.round:<7}"
              f"{r.version_trained_on:<8}{r.t_arrival:<12.2f}")
    if len(plan.applies):
        n_batches = plan.applies[-1].batch + 1
        sizes = {}
        for a in plan.applies:
            sizes[a.batch] = a.batch_size
        print(f"[plan] 施加：共 {len(plan.applies)} 次 / {n_batches} 批"
              f"（批大小: {[sizes[b] for b in sorted(sizes)]}）")
    else:
        print("[plan] 施加：0 次（无计划轮次或策略未触发任何聚合）")
    consumers = plan.sync_consumers()
    print(f"[plan] 版本保留：{sorted(consumers)} 共 {len(consumers)} 个版本会被同步"
          f"（管理器同时最多保留这么多份 LoRA 快照，注意 CPU 内存）")
