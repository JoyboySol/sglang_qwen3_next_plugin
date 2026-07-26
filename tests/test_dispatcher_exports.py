import importlib
from pathlib import Path

import sglang

from sglang_qwen3_next_plugin.versioning import get_active_variant_module_name


def test_dispatcher_reexports_variant_symbols() -> None:
    dispatcher = importlib.import_module("sglang_qwen3_next_plugin.qwen3_next")
    variant = importlib.import_module(get_active_variant_module_name(sglang.__version__))

    assert dispatcher.Qwen3NextForCausalLM is variant.Qwen3NextForCausalLM
    assert dispatcher.Qwen3NextModel is variant.Qwen3NextModel

    if hasattr(variant, "gdn_with_output"):
        assert dispatcher.gdn_with_output is variant.gdn_with_output


def test_sglang_0_5_2_variant_keeps_divide_symbol() -> None:
    source = Path("sglang_qwen3_next_plugin/variants/sglang_0_5_2.py").read_text()

    assert "from sglang.srt.distributed import (" in source
    assert "divide," in source
