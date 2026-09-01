"""tiny_mock：CPU 秒级跑通全流程的微型伪 VLM（不下载任何权重）。

模拟 Qwen-VL 的接口形状（input_ids / labels / pixel_values / image_grid_thw），
用于在碰真模型之前验证：多客户端线程、延迟模拟、四种聚合策略、日志与评估链路。
"""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModelAdapter


TINY_VOCAB = 128
TINY_D = 32
TINY_PIXEL_DIM = 12
PAD_ID = 0


def tiny_tokenize(text: str, max_len: int = 32) -> list[int]:
    """词级 hash 分词（稳定、无依赖）。首位置保留给图像向量。"""
    ids = [1]  # 1 = <image> 占位
    for w in text.lower().split():
        ids.append(int(w.encode("utf-8").hex(), 16) % (TINY_VOCAB - 2) + 2)
    return ids[:max_len]


class TinyMockModel(nn.Module):
    """embedding + 图像投影 + MLP + LM head。接口对齐 VLM 的 forward。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TINY_VOCAB, TINY_D, padding_idx=PAD_ID)
        self.img_proj = nn.Linear(TINY_PIXEL_DIM, TINY_D)
        self.block = nn.Sequential(
            nn.Linear(TINY_D, TINY_D * 2), nn.GELU(), nn.Linear(TINY_D * 2, TINY_D),
        )
        self.head = nn.Linear(TINY_D, TINY_VOCAB)

    def forward(self, input_ids, attention_mask=None, labels=None,
                pixel_values=None, image_grid_thw=None, **kwargs):
        h = self.embed(input_ids)
        if pixel_values is not None:
            h = h + self.img_proj(pixel_values).unsqueeze(1)
        logits = self.head(self.block(h))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, TINY_VOCAB),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return out

    def generate(self, input_ids, attention_mask=None, pixel_values=None,
                 image_grid_thw=None, max_new_tokens: int = 16, **kwargs):
        # 贪心续写（argmax），测试评估链路足够
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=ids, attention_mask=attention_mask,
                                  pixel_values=pixel_values)["logits"]
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_id)], dim=1)
            if (next_id == 2).all():  # 2 = <eos>
                break
        return ids


class TinyMockAdapter(BaseModelAdapter):
    """全部参数可训练；模块分组对齐 Qwen 命名以便日志统一。"""

    module_groups = {"vision": ["img_proj"], "connector": ["block.2"], "llm": ["embed", "block.0", "block.1", "head"]}

    def __init__(self, hf_id: Optional[str] = None, dtype: str = "fp32",
                 lora_cfg=None, gradient_checkpointing: bool = False):
        self.torch_dtype = torch.float32
        self._model = TinyMockModel()

    def load(self) -> None:
        pass

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def trainable_params(self) -> list[torch.nn.Parameter]:
        return list(self._model.parameters())

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().to("cpu", torch.float32).clone()
                for n, p in self._trainable_named()}

    def _trainable_named(self):
        return list(self._model.named_parameters())

    def save_pretrained(self, out_dir: str) -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(out_dir, "tiny_mock.pt"))
