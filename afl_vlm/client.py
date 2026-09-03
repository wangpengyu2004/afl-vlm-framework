"""客户端线程：{同步全局 LoRA → 本地 SFT → 算 delta → 模拟延迟上传} × num_rounds。

延迟模型（对应主流 AFL 模拟的两种范式，clients.delay_mode 选择）：
    wallclock : 时间单位均为真实秒（start_offset/net_delay 的睡眠乘 time_scale）
        speed_factor ≤ 1 : 训练结束后补睡 实测耗时 × (1/speed − 1)，模拟慢设备
        speed_factor > 1 : 无法压缩真实计算时间，自动退化为 1（打印一次警告）
        net_delay        : 每次上传采样一次（fixed/uniform/lognormal/exponential）
    staleness : FedAsync 官方做法——延迟以服务器聚合**步数** τ 计量
        staleness_tau    : 每轮采样 τ，更新由服务器持留到版本推进
                           version_trained_on + τ 才进入 buffer（net_delay 不生效）
    download_lag（两种模式叠加可用）：每轮采样，同步到全局版本 − lag 的旧快照上训练

timing: virtual（计划回放模式）：以上睡眠/补睡全部不存在——时间线完全来自
    调度计划（planner.build_plan 的产物）。客户端按计划同步"指定版本"的快照
    （wait_state_of_version，必要时阻塞等待该版本被施加），真实训练产出 delta
    后立即提交；seq_no/虚拟时间戳/tau 均取自计划，与执行进度无关。
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from .aggregation import UpdateRecord
from .config import ExperimentConfig, stable_seed
from .models.base import SharedModelManager
from .models.batch_utils import move_batch_to, model_dtype_of
from .data.tasks import TaskData


class ClientThread(threading.Thread):
    def __init__(self, client_cfg: dict, task: TaskData, manager: SharedModelManager,
                 server, cfg: ExperimentConfig, plan_rounds=None):
        super().__init__(name=f"client-{client_cfg['id']}", daemon=True)
        self.cid = client_cfg["id"]
        self.ccfg = client_cfg
        self.task = task
        self.manager = manager
        self.server = server
        self.cfg = cfg
        # 虚拟时钟模式：本客户端的计划轮次（UpdateRecord 元数据，含 seq_no/
        # sync_version/虚拟时间戳）；wallclock 模式为 None
        self.plan_rounds = plan_rounds
        self._speed_warned = False

    # -- 线程主体 -------------------------------------------------------------

    def run(self) -> None:
        try:
            if self.plan_rounds is not None:
                self._run_planned_rounds()
            else:
                self._run_rounds()
        except Exception as e:  # 崩溃也要发完成信号，否则服务器主循环会永久阻塞
            print(f"[client {self.cid}] 异常退出: {e!r}")
            raise
        finally:
            self.server.client_finished(self.cid)

    def _run_rounds(self) -> None:
        time.sleep(self.ccfg["start_offset"] * self.cfg.time_scale)
        rng = random.Random(stable_seed(self.cfg.seed, self.cid, "net"))
        rng_tau = random.Random(stable_seed(self.cfg.seed, self.cid, "tau"))
        rng_lag = random.Random(stable_seed(self.cfg.seed, self.cid, "lag"))
        staleness_mode = self.ccfg["delay_mode"] == "staleness"

        for r in range(self.ccfg["num_rounds"]):
            t_start = time.time()
            download_lag = int(self.ccfg["download_lag"].sample(rng_lag))
            with self.manager.acquire_model() as h:
                state, version_trained_on = self.manager.sync_global_and_version(
                    behind=download_lag)
                h.adapter.load_trainable_state(state)
                before = h.adapter.trainable_state()
                self._train_one_round(r, h, anchor=before)
                after = h.adapter.trainable_state()
            train_seconds = time.time() - t_start
            train_seconds_sim = self._compensate_speed(train_seconds)

            delta = {k: after[k] - before[k] for k in after}

            if staleness_mode:
                # FedAsync 式 τ 注入：不睡眠，延迟由服务器按聚合步数持留实现
                tau = int(self.ccfg["staleness_tau"].sample(rng_tau))
                net_delay = 0.0
            else:
                tau = 0
                net_delay = self.ccfg["net_delay"].sample(rng)
                time.sleep(net_delay * self.cfg.time_scale)

            self.server.submit(UpdateRecord(
                seq_no=self.server.next_seq(),
                client_id=self.cid, task=self.task.name, round=r,
                version_trained_on=version_trained_on,
                t_train_start=t_start, t_train_end=t_start + train_seconds,
                t_arrival=time.time(),
                net_delay=net_delay, train_seconds=train_seconds_sim,
                tau=tau, hold_until_version=version_trained_on + tau,
                download_lag=download_lag,
                delta=delta,
            ))
        # 完成信号由 run() 的 finally 统一发送，避免重复计数

    def _run_planned_rounds(self) -> None:
        """虚拟时钟模式：按计划回放（阶段2）。

        无任何睡眠——时间线全部来自计划；各客户端尽早产出 delta（服务器按计划
        顺序施加，早产出只是在服务器就绪区排队）。同步哪个版本由计划指定：
        wait_state_of_version 阻塞到该版本被施加为止（计划的因果序保证无死锁）；
        先等版本再租卡，避免占着卡干等。
        """
        for pr in self.plan_rounds:
            state = self.manager.wait_state_of_version(pr.version_trained_on)
            with self.manager.acquire_model() as h:
                h.adapter.load_trainable_state(state)
                before = h.adapter.trainable_state()
                self._train_one_round(pr.round, h, anchor=before)
                after = h.adapter.trainable_state()
            delta = {k: after[k] - before[k] for k in after}
            # 整条计划记录原样回传（含计划 seq_no/虚拟时间戳/tau）——服务器的
            # 施加顺序就是按计划 seq_no 回放的
            self.server.submit(replace(pr, delta=delta))

    # -- 本地训练 --------------------------------------------------------------

    def _train_one_round(self, round_idx: int, h, anchor=None) -> None:
        """anchor：本轮下载到的全局可训练状态（dict）。prox_mu>0 时加 FedProx 近端项
        μ/2·Σ‖θ−θ_anchor‖²（只对 LoRA 可训练参数，锚搬上设备一次）。"""
        model = h.adapter.model
        params = h.adapter.trainable_params()
        opt = torch.optim.AdamW(params, lr=self.ccfg["client_lr"])

        prox_mu = float(self.ccfg.get("prox_mu") or 0.0)
        anchor_dev = None
        if prox_mu > 0 and anchor:
            dtype = model_dtype_of(model)
            anchor_dev = {k: v.to(h.device, dtype=dtype) for k, v in anchor.items()}
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        gen = torch.Generator().manual_seed(
            stable_seed(self.cfg.seed, self.cid, round_idx))
        loader = DataLoader(
            self.task.train,
            batch_size=self.ccfg["batch_size"],
            shuffle=True,
            generator=gen,
            collate_fn=self.server.collator,
            num_workers=0,
        )
        dtype = model_dtype_of(model)
        model.train()
        for batch in loader:
            batch = move_batch_to(batch, h.device, model_dtype=dtype)
            out = model(**batch)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            if anchor_dev is not None:
                prox = None
                for n, p in named:
                    diff = p - anchor_dev[n]
                    prox = diff.pow(2).sum() if prox is None else prox + diff.pow(2).sum()
                loss = loss + 0.5 * prox_mu * prox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.ccfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, self.ccfg["grad_clip"])
            opt.step()
        model.eval()

    # -- 延迟补偿 ---------------------------------------------------------------

    def _compensate_speed(self, train_seconds: float) -> float:
        """返回记录用的模拟训练时长；慢设备在此补睡差额。"""
        speed = self.ccfg["speed_factor"]
        if speed < 1.0:
            extra = train_seconds * (1.0 / speed - 1.0)
            time.sleep(extra)
            return train_seconds + extra
        if speed > 1.0 and not self._speed_warned:
            print(f"[client {self.cid}] speed_factor>1 无法压缩真实计算，按 1.0 处理")
            self._speed_warned = True
        return train_seconds
