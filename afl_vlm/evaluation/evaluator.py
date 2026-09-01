"""分任务评估：eval loss + 可选生成匹配（containment）。结果写 evals.jsonl。

指标选择：
    loss       始终计算（便宜、确定性）
    match_rate 任务答案在生成文本中的包含率（ground-truth substring match，小写化）
tiny_mock 没有可解码的词表，生成匹配只对 Qwen 系启用。
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..logging_utils import JsonlWriter
from ..models.base import SharedModelManager
from ..models.batch_utils import move_batch_to, model_dtype_of


class Evaluator:
    def __init__(self, cfg: ExperimentConfig, manager: SharedModelManager, collator,
                 writer: JsonlWriter, tasks: dict):
        self.cfg = cfg
        self.manager = manager
        self.collator = collator
        self.writer = writer
        self.tasks = tasks  # {task_name: TaskData}

    # ------------------------------------------------------------------

    def run_all(self, tag: str, adapter=None, device: str = None) -> None:
        # 多卡副本模式下由调用方传入句柄绑定的 (adapter, device)；缺省回退基座
        adapter = adapter if adapter is not None else self.manager.adapter
        device = device if device is not None else self.manager.device
        for name, task in self.tasks.items():
            loss = self._eval_loss(task, adapter, device)
            row = {
                "tag": tag, "task": name,
                "global_version": self.manager.current_version(),
                "t": time.time(),
                "eval_loss": loss,
            }
            match = self._eval_generation(task, adapter, device)
            if match is not None:
                row["match_rate"] = match
            self.writer.write(row)
            print(f"[eval {tag}] {name}: loss={loss:.4f}"
                  + (f" match={match:.3f}" if match is not None else ""))

    # -- loss ---------------------------------------------------------------

    def _eval_loss(self, task, adapter, device) -> float:
        model = adapter.model
        dtype = model_dtype_of(model)
        loader = DataLoader(
            task.eval, batch_size=self.cfg.server.eval_batch_size,
            shuffle=False, collate_fn=self.collator, num_workers=0,
        )
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to(batch, device, model_dtype=dtype)
                out = model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out.loss
                total += float(loss.item())
                n += 1
        return total / max(1, n)

    # -- 生成匹配 -------------------------------------------------------------
    def _eval_generation(self, task, adapter, device):
        gen_n = self.cfg.server.gen_eval_samples
        if gen_n <= 0 or not hasattr(adapter, "processor"):
            return None
        model = adapter.model
        processor = adapter.processor
        dtype = model_dtype_of(model)
        samples = [task.eval[i] for i in range(min(gen_n, len(task.eval)))]

        model.eval()
        hits = 0
        batch = self.collator.prompt_batch(samples)
        batch = move_batch_to(batch, device, model_dtype=dtype)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                max_new_tokens=self.cfg.server.max_new_tokens,
                do_sample=False,
            )
        texts = processor.batch_decode(
            out_ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for text, s in zip(texts, samples):
            gt = s["answer"].strip().lower()
            if gt and gt in text.strip().lower():
                hits += 1
        return hits / len(samples)
