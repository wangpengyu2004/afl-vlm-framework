"""Qwen2.5-VL 适配器：LoRA 微调 + vision/connector/llm 模块分组。"""

from __future__ import annotations

from typing import Optional

import torch

from .base import BaseModelAdapter


class QwenVLAdapter(BaseModelAdapter):
    """Qwen2.5-VL (Instruct) + peft LoRA。

    模块分组（按参数名子串匹配）:
        vision    visual.blocks     （ViT blocks）
        connector visual.merger     （视觉-语言投影，三组之一，不做主角）
        llm       language_model    （语言模型层）
    """

    module_groups = {
        "vision": ["visual.blocks"],
        "connector": ["visual.merger"],
        "llm": ["language_model"],
    }

    def __init__(self, hf_id: str, dtype: str, lora_cfg, gradient_checkpointing: bool = False):
        self.hf_id = hf_id
        self.torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.lora_cfg = lora_cfg
        self.gradient_checkpointing = gradient_checkpointing
        self._model = None
        self._processor = None

    # -- BaseModelAdapter ---------------------------------------------------

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        self._processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_id,
            torch_dtype=self.torch_dtype,
            attn_implementation="sdpa",
        )
        lcfg = LoraConfig(
            r=self.lora_cfg.r,
            lora_alpha=self.lora_cfg.alpha,
            lora_dropout=self.lora_cfg.dropout,
            target_modules=self._target_modules(),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self._model = model

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError("先调用 load()")
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("先调用 load()")
        return self._processor

    def trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            n: p.detach().to("cpu", torch.float32).clone()
            for n, p in self._trainable_named()
        }

    def _trainable_named(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def save_pretrained(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)  # peft: 保存 adapter 权重

    # -- 内部 ---------------------------------------------------------------

    def _target_modules(self) -> list[str]:
        mods = list(self.lora_cfg.target_modules)
        if self.lora_cfg.include_visual:
            # Qwen2.5-VL vision block: 注意力 qkv/proj + MLP linear_fc1/fc2
            mods += ["qkv", "proj", "linear_fc1", "linear_fc2"]
        return mods
