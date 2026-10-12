import torch

from sglang_qwen3_next_plugin.variants.sglang_0_5_9 import (
    SoftClampedSiluAndMul,
)


def test_soft_clamped_silu_and_mul_matches_checkpoint_formula() -> None:
    activation = SoftClampedSiluAndMul(7.0)
    x = torch.tensor([[14.0, -14.0, 3.5, -3.5]], dtype=torch.float32)

    actual = activation(x)
    gate, up = x.chunk(2, dim=-1)
    gate = 7.0 * torch.tanh(gate / 7.0)
    up = 7.0 * torch.tanh(up / 7.0)
    expected = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(actual, expected)
