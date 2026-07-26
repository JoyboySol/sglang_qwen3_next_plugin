"""SGLang 0.5.9 专用 compat 补丁。"""

from __future__ import annotations

import torch

from sglang.srt.configs.qwen3_next import HybridLayerType, Qwen3NextConfig
from sglang.srt.environ import envs
import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_cache_mixin_module


def _plugin_layers_block_type(config: Qwen3NextConfig):
    explicit_layer_types = getattr(config, "layer_types", None)
    if explicit_layer_types is not None:
        if len(explicit_layer_types) != config.num_hidden_layers:
            raise ValueError(
                "config.layer_types length does not match num_hidden_layers: "
                f"{len(explicit_layer_types)} != {config.num_hidden_layers}"
            )
        mapping = {
            "full_attention": HybridLayerType.full_attention.value,
            HybridLayerType.full_attention.value: HybridLayerType.full_attention.value,
            "linear_attention": HybridLayerType.linear_attention.value,
            HybridLayerType.linear_attention.value: HybridLayerType.linear_attention.value,
        }
        try:
            return [mapping[layer_type] for layer_type in explicit_layer_types]
        except KeyError as exc:
            raise ValueError(f"unsupported layer_types entry: {exc.args[0]}") from exc

    layer_type_list = []
    for layer_idx in range(config.num_hidden_layers):
        if (layer_idx + 1) % config.full_attention_interval == 0:
            layer_type_list.append(HybridLayerType.full_attention.value)
        else:
            layer_type_list.append(HybridLayerType.linear_attention.value)
    return layer_type_list


def apply_runtime_compat() -> list[str]:
    applied: list[str] = []

    if not getattr(Qwen3NextConfig, "_sglang_qwen3_next_plugin_layer_types_patch", False):
        original_init = Qwen3NextConfig.__init__

        def _patched_qwen3_next_config_init(self, *args, layer_types=None, **kwargs):
            original_init(self, *args, layer_types=layer_types, **kwargs)
            self.layer_types = layer_types

        Qwen3NextConfig.__init__ = _patched_qwen3_next_config_init
        Qwen3NextConfig.layers_block_type = property(_plugin_layers_block_type)
        Qwen3NextConfig._sglang_qwen3_next_plugin_layer_types_patch = True
    applied.append("layer_types")

    if not getattr(
        kv_cache_mixin_module,
        "_sglang_qwen3_next_plugin_mamba_conv_dtype_patch",
        False,
    ):
        original_init_memory_pool = (
            kv_cache_mixin_module.ModelRunnerKVCacheMixin.init_memory_pool
        )

        def _patched_init_memory_pool(self, *args, **kwargs):
            hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
            text_config = (
                hf_config.get_text_config()
                if hasattr(hf_config, "get_text_config")
                else hf_config
            )
            model_dtype = getattr(getattr(self, "model_config", None), "dtype", None)
            if isinstance(text_config, Qwen3NextConfig) and model_dtype is not None:
                text_config.torch_dtype = model_dtype
                text_config.dtype = model_dtype
                if not envs.SGLANG_MAMBA_CONV_DTYPE.is_set():
                    if model_dtype is torch.float16:
                        envs.SGLANG_MAMBA_CONV_DTYPE.set("float16")
                    elif model_dtype is torch.bfloat16:
                        envs.SGLANG_MAMBA_CONV_DTYPE.set("bfloat16")
                    elif model_dtype is torch.float32:
                        envs.SGLANG_MAMBA_CONV_DTYPE.set("float32")
            return original_init_memory_pool(self, *args, **kwargs)

        kv_cache_mixin_module.ModelRunnerKVCacheMixin.init_memory_pool = (
            _patched_init_memory_pool
        )
        kv_cache_mixin_module._sglang_qwen3_next_plugin_mamba_conv_dtype_patch = True
    applied.append("mamba_conv_dtype")

    return applied
