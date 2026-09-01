"""设备解析与租借。

v1 采用"单实例共享模型"：所有客户端线程轮流使用同一个模型实例做本地训练，
GPU 计算因此串行，但**到达顺序逻辑不受影响**（异步性体现在到达时间戳与
聚合调度上，这是少卡模拟 AFL 的标准做法）。多卡场景下评估与训练会通过
DevicePool 互斥，避免显存超用。
"""

from __future__ import annotations

import threading
from typing import Optional

import torch


class DevicePool:
    """管理一组 torch 设备，支持租借/归还（互斥占用）。"""

    def __init__(self, devices: list[str]):
        if not devices:
            devices = ["cpu"]
        self._devices = devices
        self._locks = {d: threading.Lock() for d in devices}

    @property
    def devices(self) -> list[str]:
        return list(self._devices)

    def acquire(self, prefer: Optional[str] = None) -> str:
        """租借一台设备；优先 prefer，否则等第一台空闲的。"""
        if prefer and prefer in self._locks:
            self._locks[prefer].acquire()
            return prefer
        while True:
            for d in self._devices:
                if self._locks[d].acquire(blocking=False):
                    return d
            # 全忙则等第一个锁释放
            import time
            time.sleep(0.05)

    def release(self, device: str) -> None:
        if device in self._locks:
            self._locks[device].release()


def resolve_devices(device_cfg: str) -> DevicePool:
    """根据配置字符串解析设备池。auto → 有 cuda 用全部 cuda，否则 cpu。"""
    if device_cfg == "auto":
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
    elif device_cfg == "cuda":
        devices = ["cuda:0"]
    elif device_cfg.startswith("cuda:") or device_cfg == "cpu":
        devices = [device_cfg]
    else:
        raise ValueError(f"无法解析 device 配置: {device_cfg}")
    return DevicePool(devices)
