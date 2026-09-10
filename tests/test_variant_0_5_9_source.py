from pathlib import Path


def test_variant_0_5_9_has_custom_rmsnorm_path() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "from sglang.srt.layers.layernorm import RMSNorm" in source
    assert "GemmaRMSNorm(" not in source


def test_variant_0_5_9_disables_attn_output_gate_by_default() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert 'self.attn_output_gate = getattr(config, "attn_output_gate", False)' in source


def test_variant_0_5_9_supports_num_experts_zero() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert 'if getattr(config, "num_experts", 0) <= 0:' in source
    assert "return False" in source


def test_variant_0_5_9_removes_qk_norm_members() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "self.q_norm =" not in source
    assert "self.k_norm =" not in source


def test_variant_0_5_9_enables_qkv_bias() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "bias=True," in source


def test_variant_0_5_9_skips_removed_qk_norm_weights() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert '"q_norm" in name or "k_norm" in name' in source


def test_variant_0_5_9_uses_single_logical_expert_for_non_moe() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "num_logical_experts = num_experts if num_experts > 0 else 1" in source


def test_variant_0_5_9_uses_per_layer_sparse_mlp_layout() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "def _is_sparse_mlp_layer(config: Qwen3NextConfig, layer_id: int)" in source
    assert 'layer_id in getattr(config, "mlp_only_layers", [])' in source
    assert 'decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)' in source
    assert "self.is_layer_sparse = _is_sparse_mlp_layer(config, layer_id)" in source


def test_variant_0_5_9_respects_attention_nope() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert 'config, "attn_position_embedding_type", "rope"' in source
    assert 'if self.attn_position_embedding_type == "rope":' in source
    assert "if self.rotary_emb is not None:" in source


def test_variant_0_5_9_supports_moe_router_sqrt_gate() -> None:
    source = (
        Path("sglang_qwen3_next_plugin/variants/sglang_0_5_9.py").read_text()
    )

    assert "class Qwen3NextSparseMoeBlock(Qwen2MoeSparseMoeBlock)" in source
    assert 'getattr(config, "moe_router_sqrt_gate", False)' in source
    assert "topk_weights.clamp_min(tiny).sqrt()" in source
    assert "custom_routing_function=sqrt_gate_routing" in source
