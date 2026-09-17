---
name: yulan-moe-sglang-adaptation
description: Adapt YuLan-MoE Qwen3-Next checkpoints to SGLang, including mixed dense/MoE layers, checkpoint-specific attention and router semantics, and SGLang 0.5.9 Hybrid GDN runtime compatibility.
---

# YuLan-MoE SGLang 适配

本 skill 用于把 YuLan-MoE（Qwen3-Next / Hybrid GDN）checkpoint 接入 SGLang，并定位“模型语义不一致”和“SGLang 版本运行时兼容”两类问题。适配时必须先以 checkpoint 配置、权重键和当前 SGLang 上游源码为事实，再决定是否需要插件 patch。

## 适配边界

把改动分成两层：

- `variants/sglang_x_y_z.py`：checkpoint 语义定制。它描述 YuLan-MoE 本身要求的模型结构或数值规则，不应因为 SGLang 小版本变化而重复复制。
- `compat/sglang_x_y_z.py`：版本运行时兼容。它只修复特定 SGLang 版本的 API、dtype、cache 或 backend 缺口。

不要把二者合并成一个“大模型文件”，也不要因为旧版本需要某个 patch，就默认把 patch 继承到新版本。

## 第一步：采集事实

先检查实际环境和 checkpoint：

```bash
python - <<'PY'
import torch, sglang
print("sglang", sglang.__version__)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

python -m pip show flashinfer-python sgl-kernel vllm
```

记录 `sglang/srt/models/qwen3_next.py`、`sglang/srt/configs/qwen3_next.py`、`registry.py` 的真实路径和内容，并确认 registry 是否原生消费 `SGLANG_EXTERNAL_MODEL_PACKAGE`。读取 checkpoint `config.json`，至少确认：

- `layer_types`、`full_attention_interval`、`num_hidden_layers`
- `num_experts`、`num_experts_per_tok`
- `mlp_only_layers`、`decoder_sparse_step`
- `attn_position_embedding_type`、`rope_theta`
- `moe_router_score_function`、`moe_router_sqrt_gate`、`norm_topk_prob`
- checkpoint dtype，以及服务实际使用的 `model_config.dtype`

同时抽样检查权重键：第 0、1 层若出现 `mlp.gate_proj/up_proj/down_proj`，而后续层出现 `mlp.gate`、`mlp.experts.*`、`mlp.shared_expert.*`，说明这是混合 dense/MoE 布局，不能只依据全局 `num_experts` 建模。

## 第二步：实现 YuLan-MoE 模型语义

### 逐层判断 dense 或 sparse MoE

在 variant 中使用与 HF Qwen3-Next 一致的判定，而不是 `num_experts > 0`：

```python
def _is_sparse_mlp_layer(config, layer_id):
    if getattr(config, "num_experts", 0) <= 0:
        return False
    if layer_id in getattr(config, "mlp_only_layers", []):
        return False
    step = getattr(config, "decoder_sparse_step", 1)
    if step <= 0:
        raise ValueError(f"decoder_sparse_step must be positive: {step}")
    return (layer_id + 1) % step == 0
```

对 linear-attention 和 full-attention 两种 decoder layer 都使用该函数，并用它计算当前层、前一层和后一层的 `LayerScatterModes` 状态。YuLan-MoE 的典型配置 `mlp_only_layers=[0, 1]`、`decoder_sparse_step=1`、`num_experts=192` 的结果是：第 0、1 层 dense，第 2 层起 sparse MoE。

### 支持 `moe_router_sqrt_gate`

如果 checkpoint 设置 `moe_router_sqrt_gate=true`，不能直接复用忽略该字段的通用 `Qwen2MoeSparseMoeBlock`。继承其专家/shared-expert 实现，替换 `TopK` 的 routing function：

1. 按 `moe_router_score_function` 做 top-k，并遵循 `norm_topk_prob` 的原始归一化设置。
2. 对已选且已归一化的 routing weight 做 `clamp_min(torch.finfo(dtype).tiny).sqrt()`。
3. 不要在开平方后再次归一化；这会改变 checkpoint 语义。
4. 只有 `moe_router_sqrt_gate` 打开时才安装该自定义 `TopK`。

模型收集 MoE 权重时，也应识别自定义的 YuLan-MoE sparse block，而不是只识别父类实例。

### 按配置决定是否使用 RoPE

读取 `attn_position_embedding_type`，只接受 `rope` 和 `nope`：

- `rope`：创建 rotary embedding，并在 q/k 上应用。
- `nope`：不创建 rotary embedding，也不调用它。
- 其他值：立即抛出 `ValueError`，不要静默回退到 RoPE。

`rope_theta=None` 时才使用合理默认值（当前实现为 `10000`）；不能因为默认值存在就无条件启用 RoPE。YuLan-MoE checkpoint 若配置 `nope`，无条件 RoPE 会造成与 HF/vLLM 结果的系统性偏差。

## 第三步：实现 SGLang 0.5.9 runtime compat

YuLan-MoE Hybrid GDN 在 SGLang 0.5.9 的已知问题是：服务可能把 checkpoint 的 `float32` 配置 downcast 到 `float16`，但 `mamba2_state_dtype()` 仍按默认值分配 `bfloat16` conv cache；服务启动和 warmup 可能正常，首个真实 `/generate` 才触发：

```text
conv_states_.scalar_type() == input_type
```

将补丁放在 `compat/sglang_0_5_9.py`，patch `ModelRunnerKVCacheMixin.init_memory_pool()`：

1. 从 `self.model_config.hf_config` 取得 text config。
2. 用 `self.model_config.dtype` 同步 `text_config.torch_dtype` 和 `text_config.dtype`。
3. 用户没有显式设置 `SGLANG_MAMBA_CONV_DTYPE` 时，按模型 dtype 设置 `float16`、`bfloat16` 或 `float32`。
4. 用户显式设置该环境变量时必须保留用户选择。
5. 使用模块级幂等标志，避免外部包重复导入时重复包裹方法。

该补丁属于 SGLang 0.5.9 compat，不要原样复用 0.5.2 的 `ModelRunner.init_memory_pool()` patch。若后续上游已让 Mamba conv cache 自动跟随模型 dtype，应删除该 compat，而不是继续保留。

## 环境排障顺序

如果导入 `registry` 或模型时失败，先验证环境而不是立即改模型：

- 单独执行 `import vllm._C`，排除 `torch`/`vllm` ABI 不兼容。
- 对 SGLang 0.5.9，历史环境中的 `vllm 0.11.0` 与 `torch 2.9.1` 可能冲突；确认冲突后移除不兼容的 vLLM，再验证插件。
- 确认 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin` 是否已被当前 registry 原生消费。
- 只有隔离环境冲突后仍失败，才判断是插件逻辑或 SGLang 上游缺陷。

## 代码组织和回归

每个新增 SGLang 版本先保存上游快照到 `upstream/sglang_x_y_z/`，再把实现放进对应 `variants/` 和 `compat/`。同步更新 `progress.md` 与 `changelog.md`，记录 checkpoint 事实、patch 分类、被上游吸收的旧 patch 以及验证结果。

至少执行：

```bash
python -m pytest tests/test_versioning.py -q
python -m pytest tests/test_dispatcher_exports.py tests/test_compat_modules.py tests/test_variant_0_5_9_source.py tests/test_versioning.py -q
python test_plugin.py
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin python scripts/check_plugin_import.py
python scripts/run_acceptance.py --skip-runtime
```

环境允许时必须做真实服务验证，不要只验证 import、启动或 warmup：

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin \
  python -m sglang.launch_server \
  --model-path <YuLan-MoE-checkpoint> \
  --host 127.0.0.1 --port 30110 --tp 2 \
  --trust-remote-code --disable-cuda-graph

python scripts/validate_generation.py --host 127.0.0.1 --port 30110
python scripts/validate_logprob.py --host 127.0.0.1 --port 30110
python scripts/run_acceptance.py --host 127.0.0.1 --port 30110
```

若要与 vLLM 进行精度对照，依据 checkpoint 权重和参考服务显式传入 `--dtype bfloat16`；不要修改 `config.json`，也不要让 SGLang 的 `auto` 从 `float32` 配置自行选择 `float16` 后再比较结果。只有真实生成和 logprob 都通过，才能认为 YuLan-MoE 适配完成。
