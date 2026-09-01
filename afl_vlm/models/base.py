"""模型适配器基类与共享模型管理器（全局 LoRA 状态的唯一权威副本）。"""

from __future__ import annotations

import collections
import copy
import threading
from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseModelAdapter(ABC):
    """统一接口：加载模型、取可训练参数、按模块分组。

    约定：adapter.model 是可直接 forward/generate 的 nn.Module（peft 模型或普通模型），
    collator 产出的 batch dict 直接 **model(**batch)。
    """

    #: 模块分组：组名 -> 参数名前缀列表（用于日志里的分组 staleness 统计；
    #: connector 只是若干组之一，不做特殊化处理）
    module_groups: dict[str, list[str]] = {}

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module: ...

    @abstractmethod
    def trainable_params(self) -> list[torch.nn.Parameter]: ...

    @abstractmethod
    def trainable_state(self) -> dict[str, torch.Tensor]:
        """可训练参数快照 {参数名: cpu float32 张量}。"""

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        """把全局可训练状态写回模型参数。"""
        with torch.no_grad():
            for name, p in self._trainable_named():
                if name not in state:
                    raise KeyError(f"全局状态缺少参数 {name}")
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))

    def group_of(self, param_name: str) -> str:
        for group, prefixes in self.module_groups.items():
            for prefix in prefixes:
                if prefix in param_name:
                    return group
        return "other"

    @abstractmethod
    def _trainable_named(self) -> list[tuple[str, torch.nn.Parameter]]: ...

    def save_pretrained(self, out_dir: str) -> None:  # pragma: no cover - 子类按需实现
        raise NotImplementedError


class SharedModelManager:
    """全局可训练状态（CPU float32，唯一权威副本）+ 每设备一个模型副本。

    - ``state_lock``  保护 global_state / global_version（服务器聚合时持有，开销极小）
    - 模型副本按设备各存一份（replicas[device]）；DevicePool 的设备锁保证同一副本
      同一时刻只被一个线程使用。多卡时客户端训练**真并发**（CUDA kernel 执行期间
      释放 GIL），异步语义不变：同步/版本/delta 逻辑与单卡完全一致

    客户端流程: acquire_model(租一张空闲卡) → sync_global_and_version(记下 version)
                → 把全局状态载入该卡副本 → 训练 → 算 delta → release
    虚拟时钟模式(timing: virtual)则改为: wait_state_of_version(计划版本) 取指定版本
                快照 → 载入训练（同步哪个版本由计划决定，不由执行进度决定）
    服务器流程: apply_delta(delta, weight, lr)（纯 CPU 张量运算，逐更新推进版本）
    """

    def __init__(self, adapter: BaseModelAdapter, pool, history_size: int = 0):
        self.adapter = adapter           # 基座 adapter（同时充当 devices[0] 上的副本）
        self.pool = pool
        self.device = pool.devices[0]    # 参考设备（兼容/日志用；训练用哪张卡由句柄决定）
        self.global_state: dict[str, torch.Tensor] = {}
        self.global_version: int = 0
        self._state_lock = threading.Lock()
        self._state_cond = threading.Condition(self._state_lock)   # 按版本等待用
        self.replicas: dict[str, BaseModelAdapter] = {}
        # 版本历史（download_lag > 0 时客户端要同步到旧版快照）；0 = 关闭以省内存
        self._history = collections.deque(maxlen=history_size) if history_size > 0 else None
        # 虚拟时钟模式（timing: virtual）的版本快照表：按计划保留会被同步的版本，
        # 其余施加后即裁剪。set_plan_retention() 激活
        self._history_map: Optional[dict[int, dict[str, torch.Tensor]]] = None
        self._retain: dict[int, int] = {}

    # -- 生命周期 -----------------------------------------------------------

    def startup(self) -> None:
        self.adapter.load()              # 母版加载到 CPU
        self.global_state = self.adapter.trainable_state()
        # 每卡一个副本：CPU 母版 deepcopy → 移到目标卡，逐个进行
        # （母版留在 CPU 直到全部复制完 → CPU 内存峰值 ≈ 2× 模型大小）
        devices = self.pool.devices
        for d in devices[1:]:
            rep = copy.deepcopy(self.adapter)
            rep.model.to(d)
            self.replicas[d] = rep
        self.adapter.model.to(devices[0])
        self.replicas[devices[0]] = self.adapter
        if self._history is not None:
            self._history.append((0, {k: v.clone() for k, v in self.global_state.items()}))
        print(f"[manager] 模型副本 ×{len(self.replicas)}: {list(self.replicas)}")

    def shutdown(self) -> None:
        pass

    # -- 客户端侧 -----------------------------------------------------------

    def acquire_model(self):
        return ModelHandle(self)

    def current_version(self) -> int:
        with self._state_lock:
            return self.global_version

    def snapshot_global(self) -> dict[str, torch.Tensor]:
        with self._state_lock:
            return {k: v.clone() for k, v in self.global_state.items()}

    def sync_global_and_version(self, behind: int = 0) -> tuple[dict[str, torch.Tensor], int]:
        """原子地取（全局状态快照, 版本号）——消除同步与读版本之间的竞态。

        behind > 0：取 behind 个版本之前的快照（模拟下载延迟——客户端拿到的是
        旧版全局模型）。需要以 history_size > 0 构造管理器；版本尚未攒够 behind
        步（或已超出历史窗口）时退回最旧可用版本——真实系统此时也只能拿到
        服务器还留存的旧版。
        """
        with self._state_lock:
            if behind <= 0:
                state = {k: v.clone() for k, v in self.global_state.items()}
                return state, self.global_version
            if self._history is None:
                raise RuntimeError(
                    "download_lag > 0 需要 history_size>0 的 SharedModelManager"
                    "（run_train 会按 required_history 自动设置）")
            target = self.global_version - behind
            target = max(target, self._history[0][0])   # 不足则退回最旧可用版本
            for v, state in self._history:
                if v == target:
                    return {k: t.clone() for k, t in state.items()}, v
            raise RuntimeError(
                f"版本历史中没有 v{target}（当前 v{self.global_version}，"
                f"history_size={self._history.maxlen}）")

    def set_plan_retention(self, consumers: dict[int, int]) -> None:
        """虚拟时钟模式：声明每个版本将被同步的次数（来自 TrainPlan.sync_consumers）。

        激活 _history_map 快照表：每次施加都存版本快照，消费计数减到 0 且不是最新版
        即裁剪——同时最多保留"仍会被同步"的那么多份 LoRA 状态，内存有界。
        须在 startup() 之后、线程启动之前调用（此时全局状态即 v0）。
        """
        self._retain = dict(consumers)
        self._history_map = {}
        if self._retain.get(0, 0) > 0:
            self._history_map[0] = {k: v.clone() for k, v in self.global_state.items()}

    def wait_state_of_version(self, version: int) -> dict[str, torch.Tensor]:
        """虚拟时钟模式：阻塞直到指定版本被施加，返回该版本的状态快照（按计划同步）。

        客户端因此能训练在"计划指定"的旧版本上，而不管真实执行进度走到哪——
        delta 早于或晚于计划的虚拟时刻产出都无影响，服务器只认计划的施加顺序。
        """
        with self._state_cond:
            while version not in self._history_map:
                if version > self.global_version:
                    self._state_cond.wait()      # 还没施加到 → 等服务器推进
                else:
                    raise RuntimeError(
                        f"版本 v{version} 已被裁剪却仍有消费者（retention 配置错误）；"
                        f"当前 v{self.global_version}，保留: {sorted(self._history_map)}")
            state = {k: t.clone() for k, t in self._history_map[version].items()}
            if version in self._retain:
                self._retain[version] -= 1
            if self._retain.get(version, 0) <= 0 and version != self.global_version:
                self._history_map.pop(version, None)   # 最后一个消费者已取走 → 裁剪
        return state

    # -- 服务器侧 -----------------------------------------------------------

    def apply_delta(self, delta: dict[str, torch.Tensor], weight: float, server_lr: float) -> int:
        """按施加顺序逐个调用：全局状态 += lr * weight * delta，版本 +1。

        返回施加后的全局版本号。顺序施加是设计核心：不同顺序产生不同中间态。
        """
        with self._state_lock:
            for k, g in self.global_state.items():
                if k not in delta:
                    raise KeyError(f"delta 缺少参数 {k}")
                g.add_(delta[k].to(g.dtype), alpha=server_lr * weight)
            self.global_version += 1
            if self._history_map is not None:
                # 虚拟时钟模式：快照表（按计划保留 + 用完即裁剪）
                self._history_map[self.global_version] = {
                    k: v.clone() for k, v in self.global_state.items()}
                for v in [v for v in self._history_map
                          if v != self.global_version and self._retain.get(v, 0) <= 0]:
                    del self._history_map[v]
                self._state_cond.notify_all()    # 唤醒等待该版本的客户端
            elif self._history is not None:
                self._history.append((self.global_version,
                                      {k: v.clone() for k, v in self.global_state.items()}))
            return self.global_version

    def export_global(self) -> dict[str, torch.Tensor]:
        with self._state_lock:
            return {k: v.clone() for k, v in self.global_state.items()}


class ModelHandle:
    """with 语义的设备/副本占用句柄：持有期间独占该设备上的模型副本。

    用法:
        with manager.acquire_model() as h:
            h.adapter.load_trainable_state(state)   # 该卡副本
            batches 移到 h.device，在 h.adapter.model 上训练
    多卡时 acquire 拿第一张空闲卡 → 客户端训练真并发；单卡时等同一张卡（串行）。
    """

    def __init__(self, manager: SharedModelManager):
        self._m = manager
        self.device: Optional[str] = None
        self.adapter: Optional[BaseModelAdapter] = None

    def __enter__(self) -> "ModelHandle":
        self.device = self._m.pool.acquire()         # 任一空闲设备（按序取第一台空闲的）
        self.adapter = self._m.replicas[self.device]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._m.pool.release(self.device)
        self.device = None
        self.adapter = None
