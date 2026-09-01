"""异步服务器：到达队列 → 调度策略 → **按策略指定的顺序逐个施加** → 版本推进。

关键不变量：
    1. 聚合是**顺序施加**（不是一步加权平均）——非交换效应真实存在于中间态；
    2. 施加顺序完全由 AggregationPolicy 返回的列表顺序决定——这是论文的研究杠杆；
    3. 每施加一个更新全局版本 +1，staleness 以版本差记账。

延迟模式（clients.delay_mode）：
    wallclock : 更新到达即进入 buffer（延迟已在客户端侧用睡眠模拟）；
    staleness : FedAsync 官方式 τ 注入——更新提交后先进入 pending 持留池，
                全局版本推进到 version_trained_on + τ 才"到达"（进入 buffer）。
                wallclock 更新的 hold_until_version = version_trained_on，
                恒满足，走同一条 admit 路径，无需分支。

虚拟时钟模式（experiment.timing: virtual，传入 plan=TrainPlan）：
    主循环退化为"计划回放器"——阶段1（planner.build_plan）已经用同一策略在
    元数据上算好全部施加顺序/权重/批次；运行时只做三件事：
        1. 收 delta（按 seq_no 存入就绪表）；
        2. 按计划顺序施加，delta 未产出就等（只拉伸墙钟，绝不重排）；
        3. 批次结束处照常触发周期评估。
    τ 持留/download_lag/到达时间线全部已编码进计划，运行时不再模拟。
"""

from __future__ import annotations

import itertools
import queue
import threading
import time

import torch

from .aggregation import ServerState, UpdateRecord
from .config import ExperimentConfig
from .logging_utils import JsonlWriter, ensure_dir
from .models.base import SharedModelManager


class AsyncServer:
    SENTINEL = object()  # 客户端完成信号

    def __init__(self, cfg: ExperimentConfig, manager: SharedModelManager, policy,
                 tasks: dict, collator, plan=None):
        self.cfg = cfg
        self.manager = manager
        self.policy = policy
        self.tasks = tasks
        self.collator = collator
        # 虚拟时钟模式的调度表（afl_vlm.planner.TrainPlan）；None = wallclock 实时模式
        self.plan = plan
        self._plan_ptr = 0                  # 计划回放指针（已施加的计划条数）

        self.queue: queue.Queue = queue.Queue()
        self.buffer: list[UpdateRecord] = []
        self.pending: list[UpdateRecord] = []   # staleness 模式的 τ 持留池
        self._seq = itertools.count()
        self._seq_lock = threading.Lock()

        self.clients_done = 0
        self.total_clients = sum(cfg.clients.task_mix.values())
        self.batch_count = 0

        out = ensure_dir(cfg.output_dir)
        self.w_arrivals = JsonlWriter(f"{out}/arrivals.jsonl")
        self.w_aggs = JsonlWriter(f"{out}/aggregations.jsonl")
        self.w_evals = JsonlWriter(f"{out}/evals.jsonl")

        from .evaluation.evaluator import Evaluator
        self.evaluator = Evaluator(cfg, manager, collator, self.w_evals, tasks=tasks)

    # -- 客户端调用 -----------------------------------------------------------

    def next_seq(self) -> int:
        with self._seq_lock:
            return next(self._seq)

    def submit(self, record: UpdateRecord) -> None:
        self.w_arrivals.write({
            "client": record.client_id, "task": record.task, "round": record.round,
            "version_trained_on": record.version_trained_on,
            "t_train_start": record.t_train_start, "t_train_end": record.t_train_end,
            "t_arrival": record.t_arrival, "net_delay": record.net_delay,
            "train_seconds": record.train_seconds,
            "tau": record.tau, "hold_until_version": record.hold_until_version,
            "download_lag": record.download_lag,
            # 真实提交时刻：wallclock 模式 ≈ t_arrival；virtual 模式下时间线是
            # 虚拟的，此字段是唯一的真实时间对照
            "t_submit_real": time.time(),
        })
        self.queue.put(record)

    def client_finished(self, client_id: str) -> None:
        self.queue.put((self.SENTINEL, client_id))

    # -- 主循环 ---------------------------------------------------------------

    def run(self) -> dict:
        if self.plan is not None:
            return self._run_planned()
        while True:
            item = self.queue.get()
            if isinstance(item, tuple) and item[0] is self.SENTINEL:
                self.clients_done += 1
                if self.clients_done >= self.total_clients:
                    break
                continue
            record: UpdateRecord = item
            self._receive(record)
            self._promote_pending()

        # 全部客户端结束：先放行仍在持留的更新（模拟结束时"在途"的部分），
        # 再交给策略做最终 flush（策略可自定义剩余更新的处置）
        self._flush_pending()
        ordered = self.policy.on_finish(self.buffer, self._state())
        self._apply(ordered)

        summary = self._finalize()
        self.w_arrivals.close(); self.w_aggs.close(); self.w_evals.close()
        return summary

    # -- 虚拟时钟模式：计划回放 --------------------------------------------------

    def _run_planned(self) -> dict:
        """按 TrainPlan 回放：收 delta → 按计划序施加 → 排空。

        客户端全部完成后，所有计划内 delta 必已提交（计划覆盖了每个客户端的
        每一轮），此时一次性排空剩余计划并校验完整性。
        """
        ready: dict[int, UpdateRecord] = {}
        while True:
            item = self.queue.get()
            if isinstance(item, tuple) and item[0] is self.SENTINEL:
                self.clients_done += 1
                if self.clients_done >= self.total_clients:
                    break
                continue
            record: UpdateRecord = item
            ready[record.seq_no] = record
            self._drain_planned(ready)

        self._drain_planned(ready)
        if self._plan_ptr != len(self.plan.applies):
            raise RuntimeError(
                f"计划未排空：{self._plan_ptr}/{len(self.plan.applies)}"
                "（计划与执行不一致——有客户端未按计划提交？）")
        summary = self._finalize()
        self.w_arrivals.close(); self.w_aggs.close(); self.w_evals.close()
        return summary

    def _drain_planned(self, ready: dict[int, UpdateRecord]) -> None:
        """按计划顺序施加所有 delta 已就绪的更新；批次结束处触发周期评估。

        计划条目按施加顺序排列、同批次连续——施加到批次边界即计一批。
        """
        applies = self.plan.applies
        while self._plan_ptr < len(applies) and applies[self._plan_ptr].seq_no in ready:
            pa = applies[self._plan_ptr]
            u = ready.pop(pa.seq_no)
            self._apply_planned(u, pa)
            self._plan_ptr += 1
            nxt = applies[self._plan_ptr].batch if self._plan_ptr < len(applies) else None
            if nxt != pa.batch:                      # 批次结束
                self.batch_count = pa.batch + 1
                every = self.cfg.server.eval_every_batches
                if every and self.batch_count % every == 0 \
                        and self.clients_done < self.total_clients:
                    self._run_eval(tag=f"batch{self.batch_count}")

    def _apply_planned(self, u: UpdateRecord, pa) -> None:
        """施加计划条目（权重/批次/批内序号全部来自计划，不再询问策略）。"""
        staleness = self.manager.current_version() - u.version_trained_on
        new_version = self.manager.apply_delta(
            u.delta, pa.weight, self.cfg.server.aggregation.server_lr)
        self.w_aggs.write({
            "batch": pa.batch, "order_index": pa.order_index,
            "client": u.client_id, "task": u.task, "round": u.round,
            "version_trained_on": u.version_trained_on,
            "staleness": staleness, "weight": pa.weight,
            "global_version_after": new_version,
            "t_apply": pa.t_apply,          # 虚拟施加时刻（= 计划里的到达触发时刻）
            "t_apply_real": time.time(),    # 真实施加时刻（仅对照用）
            "batch_size": pa.batch_size,
        })

    # -- 到达与持留（wallclock 实时模式）----------------------------------------

    def _receive(self, record: UpdateRecord) -> None:
        """τ 未到期的更新先进持留池；到期（含 wallclock 更新，恒到期）直接进入 buffer。"""
        if record.hold_until_version > self.manager.current_version():
            self.pending.append(record)
            return
        self._admit(record)

    def _admit(self, record: UpdateRecord) -> None:
        self.buffer.append(record)
        ordered = self.policy.on_arrivals(self.buffer, self._state())
        self._apply(ordered)

    def _promote_pending(self) -> None:
        """版本推进后放行到期的持留更新；放行可能触发新聚合、进而释放更多，循环到稳定。"""
        while self.pending:
            v = self.manager.current_version()
            ready = [r for r in self.pending if r.hold_until_version <= v]
            if not ready:
                break
            self.pending = [r for r in self.pending if r.hold_until_version > v]
            for r in sorted(ready, key=lambda x: x.seq_no):
                self.buffer.append(r)
            ordered = self.policy.on_arrivals(self.buffer, self._state())
            self._apply(ordered)

    def _flush_pending(self) -> None:
        if self.pending:
            print(f"[server] 模拟结束，放行 {len(self.pending)} 个仍在持留的更新")
            for r in sorted(self.pending, key=lambda x: x.seq_no):
                self.buffer.append(r)
            self.pending = []

    # -- 聚合 -----------------------------------------------------------------

    def _state(self) -> ServerState:
        return ServerState(
            global_version=self.manager.current_version(),
            sim_time=time.time(),
            num_clients_total=self.total_clients,
            clients_done=self.clients_done,
        )

    def _apply(self, ordered: list[tuple[UpdateRecord, float]]) -> None:
        if not ordered:
            return
        applied = set()
        for order_index, (u, weight) in enumerate(ordered):
            staleness = self.manager.current_version() - u.version_trained_on
            new_version = self.manager.apply_delta(
                u.delta, weight, self.cfg.server.aggregation.server_lr)
            self.w_aggs.write({
                "batch": self.batch_count,
                "order_index": order_index,          # ← 施加顺序（顺序效应核心字段）
                "client": u.client_id, "task": u.task, "round": u.round,
                "version_trained_on": u.version_trained_on,
                "staleness": staleness, "weight": weight,
                "global_version_after": new_version,
                "t_apply": time.time(),
                "batch_size": len(ordered),
            })
            applied.add(u.seq_no)
        self.buffer = [u for u in self.buffer if u.seq_no not in applied]
        self.batch_count += 1

        every = self.cfg.server.eval_every_batches
        if every and self.batch_count % every == 0 and self.clients_done < self.total_clients:
            self._run_eval(tag=f"batch{self.batch_count}")

    # -- 评估与收尾 ------------------------------------------------------------

    def _run_eval(self, tag: str) -> None:
        with self.manager.acquire_model() as h:
            h.adapter.load_trainable_state(self.manager.export_global())
            self.evaluator.run_all(tag=tag, adapter=h.adapter, device=h.device)

    def _finalize(self) -> dict:
        if self.cfg.server.eval_every_batches:
            self._run_eval(tag="final")
        out = ensure_dir(self.cfg.output_dir)
        torch.save(self.manager.export_global(), f"{out}/global_trainable.pt")
        with self.manager.acquire_model() as h:
            h.adapter.load_trainable_state(self.manager.export_global())
            h.adapter.save_pretrained(f"{out}/model_final")
        summary = {
            "aggregate_batches": self.batch_count,
            "final_version": self.manager.current_version(),
            "output_dir": out,
        }
        print(f"[server] 完成: {summary}")
        return summary
