"""共享 batch 的设备搬运工具（client 训练与 evaluator 共用）。"""

from __future__ import annotations

import torch

_TENSOR_KEYS = ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")


def move_batch_to(batch: dict, device: str, model_dtype: torch.dtype | None = None) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if k == "pixel_values" and model_dtype is not None and t.is_floating_point():
                t = t.to(model_dtype)
            out[k] = t
        else:
            out[k] = v
    return out


def model_dtype_of(module: torch.nn.Module) -> torch.dtype:
    for p in module.parameters():
        return p.dtype
    return torch.float32
