"""JSONL 日志与工具函数。所有论文图表的原始数据都来自这里的三个日志文件：
    arrivals.jsonl     每次上传到达（客户端/任务/版本/时间/延迟）
    aggregations.jsonl 每次聚合批次的**施加顺序**（顺序效应研究的核心数据）
    evals.jsonl        分任务评估曲线（能力随聚合阶段的演化）
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional


class JsonlWriter:
    """线程安全的 JSONL 追加写。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    @property
    def path(self) -> str:
        return self._path


def read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_jsonl_optional(path: Optional[str]) -> list[dict]:
    return read_jsonl(path) if path else []


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
