"""模型注册表：配置里的 model.name → 适配器工厂。

新增模型三步：
    1. 在 models/ 下新建 XxxAdapter(BaseModelAdapter)
    2. 在 MODEL_REGISTRY 注册 {name: {"hf_id": ..., "factory": ...}}
    3. 在 config.validate_config 的 known_models 里加名字
"""

from __future__ import annotations

from .base import BaseModelAdapter, SharedModelManager
from .tiny_mock import TinyMockAdapter, TINY_PIXEL_DIM

MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-vl-7b": {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen2.5-vl-3b": {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    "tiny_mock": {"hf_id": None, "factory": TinyMockAdapter},
}


def build_model_adapter(model_cfg):
    entry = MODEL_REGISTRY[model_cfg.name]
    hf_id = model_cfg.hf_id or entry["hf_id"]
    if model_cfg.name == "tiny_mock":
        return TinyMockAdapter()
    # Qwen 系：延迟导入，避免 CPU 冒烟环境也要装 transformers
    from .qwen_vl import QwenVLAdapter
    return QwenVLAdapter(
        hf_id=hf_id,
        dtype=model_cfg.dtype,
        lora_cfg=model_cfg.lora,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
    )


__all__ = [
    "BaseModelAdapter", "SharedModelManager", "MODEL_REGISTRY",
    "build_model_adapter", "TinyMockAdapter", "TINY_PIXEL_DIM",
]
