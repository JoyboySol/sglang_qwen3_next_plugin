import importlib
from importlib.metadata import version as package_version

import pytest
import torch


def test_sglang_0_5_2_runtime_compat_contract() -> None:
    if package_version("sglang") != "0.5.2":
        pytest.skip("requires sglang==0.5.2")
    compat = importlib.import_module("sglang_qwen3_next_plugin.compat.sglang_0_5_2")
    applied = compat.apply_runtime_compat()

    assert "layer_types" in applied
    assert "hybrid_conv_backend" in applied
    assert "model_runner_dtype" in applied

    from sglang.srt.configs.qwen3_next import Qwen3NextConfig
    import sglang.srt.layers.attention.hybrid_linear_attn_backend as backend
    import sglang.srt.model_executor.model_runner as model_runner_module

    assert getattr(Qwen3NextConfig, "_sglang_qwen3_next_plugin_layer_types_patch", False)
    assert getattr(backend, "_sglang_qwen3_next_plugin_conv_patch", False)
    assert getattr(model_runner_module, "_sglang_qwen3_next_plugin_dtype_patch", False)


def test_sglang_0_5_9_runtime_compat_contract() -> None:
    if package_version("sglang") != "0.5.9":
        pytest.skip("requires sglang==0.5.9")
    compat = importlib.import_module("sglang_qwen3_next_plugin.compat.sglang_0_5_9")
    applied = compat.apply_runtime_compat()

    assert "layer_types" in applied
    assert "mamba_conv_dtype" in applied
    assert "soft_clamped_swiglu" in applied

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_cache_mixin_module
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module

    assert getattr(
        kv_cache_mixin_module,
        "_sglang_qwen3_next_plugin_mamba_conv_dtype_patch",
        False,
    )
    assert getattr(
        fused_moe_module,
        "_sglang_qwen3_next_plugin_soft_clamp_patch",
        False,
    )


def test_sglang_0_5_9_yulan_soft_clamp_numerics() -> None:
    if package_version("sglang") != "0.5.9":
        pytest.skip("requires sglang==0.5.9")
    compat = importlib.import_module("sglang_qwen3_next_plugin.compat.sglang_0_5_9")
    compat.apply_runtime_compat()

    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module

    x = torch.tensor([[14.0, -14.0, 3.5, -3.5]], dtype=torch.float32)
    actual = fused_moe_module._swiglu_silu_clamp_mul(x, -7.0)
    gate, up = x.chunk(2, dim=-1)
    gate = 7.0 * torch.tanh(gate / 7.0)
    up = 7.0 * torch.tanh(up / 7.0)
    expected = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(actual, expected)
