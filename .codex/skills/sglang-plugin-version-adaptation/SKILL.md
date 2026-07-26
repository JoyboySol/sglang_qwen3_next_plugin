---
name: sglang-plugin-version-adaptation
description: Use when adding or reviewing support for a new SGLang version in this repository, especially when you need to separate checkpoint-specific customizations from version-specific compat patches.
---

# SGLang Plugin Version Adaptation

## Overview

这是本仓库给 `sglang` 新版本做插件适配的标准流程。核心原则只有两个：

- 以上游真实代码和真实环境事实为准，不以历史记忆为准
- 先区分“checkpoint 自定义”与“版本 compat”，再写代码

## 何时使用

在以下情况必须使用本 skill：

- 仓库要新增一个 `sglang` 版本支持
- 需要判断旧 patch 是否仍然需要
- 需要确认 `SGLANG_EXTERNAL_MODEL_PACKAGE` 是否已被上游原生支持
- 新版本出现导入失败、runtime 报错、dtype/backend 行为差异

以下情况不要单独使用本 skill：

- 只是修改 prompt、README 文案或无关脚本
- 只是验证现有某个版本的服务状态，不涉及新增版本

## 固定流程

### 1. 先采集新环境事实

必须先记录：

- `sglang.__version__`
- `sglang/srt/models/qwen3_next.py` 路径
- `sglang/srt/configs/qwen3_next.py` 路径
- `sglang/srt/models/registry.py` 是否原生消费 `SGLANG_EXTERNAL_MODEL_PACKAGE`
- 关键运行时栈版本：
  - `torch`
  - `vllm`
  - `flashinfer-python`
  - `sgl-kernel`

如果新环境连 `sglang` 主包都没装，先安装，再继续。

### 2. 抽取上游基线

每新增一个版本，都至少复制两份上游快照到仓库：

- `sglang_qwen3_next_plugin/upstream/sglang_x_y_z/qwen3_next.py`
- `sglang_qwen3_next_plugin/upstream/sglang_x_y_z/configs_qwen3_next.py`

不允许一边看 venv 文件、一边直接改插件实现而不留快照。

旧版本基线也应收敛到同一目录结构里，不要把历史上游文件继续散落在包根。

### 3. 先判定哪些是公共 custom

优先按“checkpoint 语义”归类的候选项：

- `GemmaRMSNorm -> RMSNorm`
- 去掉 `q_norm` / `k_norm`
- `attn_output_gate` 默认值变化
- projection bias 变化
- `num_experts=0` 走普通 MLP
- `mlp_only_layers` / `decoder_sparse_step` 逐层 dense/MoE MLP 判定

这些逻辑只有在新上游已原生满足 checkpoint 结构时才能删除。

### 4. 再判定哪些是版本 compat

优先按“上游缺口”归类的候选项：

- `layer_types` 保存与消费
- `full_attention_interval` 推导问题
- dtype 同步问题
- Hybrid GDN / Mamba conv cache dtype 与模型实际 dtype 不一致
- Hybrid backend / kernel 路径问题
- 外部插件包发现机制

compat 必须单独放在：

- `sglang_qwen3_next_plugin/compat/sglang_x_y_z.py`

不能重新塞回单一大模型文件里。

### 5. 先排除环境冲突，再判断插件问题

如果新环境导入 `registry` 或 `qwen3_next` 就崩：

- 先单独验证 `import vllm._C`
- 看 `torch` 与 `vllm` 是否 ABI 不兼容
- 必要时用临时假 `vllm` 包做隔离验证
- 如果确认是环境二进制冲突，优先移除或替换不兼容包
  - 例如本仓库在 `sglang 0.5.9` 适配时，`.venv-sgl0_5_9` 中的 `vllm 0.11.0` 与 `torch 2.9.1` ABI 不兼容
  - 直接卸载该 `vllm` 后，插件接管与真实服务验收才恢复正常

只有当上游模块在隔离环境冲突后仍失败，才继续怀疑插件或 `sglang` 主包逻辑。

### 6. 维护分层入口

仓库中的角色分工固定为：

- `versioning.py`
  - 版本键、variant 模块、compat 模块、基线路径
- `variants/`
  - 各版本实际模型实现挂载点
- `upstream/`
  - 各版本上游快照
- `compat/`
  - 各版本 compat 补丁
- `qwen3_next.py`
  - 分发器，不承载具体版本实现

### 7. 每完成一阶段就更新文档

必须同步更新：

- `progress.md`
- `changelog.md`

至少记录：

- 新环境事实
- 哪些旧 patch 被上游吸收
- 哪些 patch 仍保留
- 当前阻塞是环境问题、上游问题还是插件问题

## 最小验收

新增版本支持后，至少做以下检查：

1. 轻量版本工具检查
   - `python -m pytest tests/test_versioning.py -q`
   - `python -m pytest tests/test_dispatcher_exports.py tests/test_compat_modules.py tests/test_variant_0_5_9_source.py tests/test_versioning.py -q`
2. 当前旧版本契约回归
   - `python test_plugin.py`
3. 新版本分发检查
   - 确认 `EntryClass.__module__` 命中对应 `variants/` 或 `upstream/` 路径
   - 运行 `python scripts/check_plugin_import.py`
4. 若环境允许，再做服务级验收
   - `python scripts/run_acceptance.py --skip-runtime`
   - `python scripts/run_acceptance.py --host 127.0.0.1 --port 30110`
   - `python scripts/validate_generation.py`
   - `python scripts/validate_logprob.py`

如果新版本服务已经能真实启动，要优先以完整服务验收为准，而不是停在静态 source diff 或 import 成功。

## 常见错误

- 看到新版本能 import，就直接把旧 patch 全量搬过去
- 不留上游快照，只在插件文件里硬改
- 把环境 ABI 冲突误判成插件逻辑问题
- 把 `0.5.2` 的 `sitecustomize/env_override` 当成所有版本都需要的公共主路径
- 把 `0.5.2` 的 dtype patch 或 hybrid backend patch 默认继承到 `0.5.9+`，却不做真实服务验证
- 只验证服务启动和 warmup，不发送真实 `/generate` 请求；YuLan-MoE 在 `0.5.9` 中就是首个真实请求才暴露 Mamba conv cache dtype mismatch
- 更新了代码，但没更新 `progress.md` 和 `changelog.md`

## 当前仓库的已知事实

- `sglang 0.5.2` 不原生支持 `SGLANG_EXTERNAL_MODEL_PACKAGE`
- `sglang 0.5.9` 的 `registry.py` 已原生消费 `SGLANG_EXTERNAL_MODEL_PACKAGE`
- `0.5.9` 的配置签名虽然已有 `layer_types`，但仍需 compat 才能优先消费显式层布局
- `.venv-sgl0_5_9` 曾因 `vllm 0.11.0` 与 `torch 2.9.1` ABI 冲突阻塞；卸载不兼容 `vllm` 后，最小插件接管与真实服务验收均恢复正常
- YuLan-MoE Hybrid GDN checkpoint 已暴露 `sglang 0.5.9` 仍需一个版本 compat：
  - 服务将 checkpoint `torch.float32` downcast 为 `torch.float16`
  - SGLang 0.5.9 的 Mamba conv cache 默认可能按 `bfloat16` 分配
  - 首个真实 `/generate` 请求会触发 `conv_states_.scalar_type() == input_type`
  - 当前修复放在 `compat/sglang_0_5_9.py`，patch `ModelRunnerKVCacheMixin.init_memory_pool()`，同步 `hf_config.dtype/torch_dtype` 并在用户未显式设置 `SGLANG_MAMBA_CONV_DTYPE` 时按 `model_config.dtype` 设置 conv cache dtype
- YuLan-MoE Hybrid GDN checkpoint 还需要公共 custom：
  - `mlp_only_layers=[0, 1]`
  - `decoder_sparse_step=1`
  - 第 0、1 层为普通 dense MLP，第 2 层起为 sparse MoE
  - 不能只用全局 `num_experts > 0` 判断所有层都是 MoE
- 当前基于 YuLan-MoE checkpoint 的 `sglang 0.5.9` 完整验收命令为：
  - 启动服务：
    - `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin python -m sglang.launch_server --model-path /root/nas/share/vc0e4b0o65t5/lvzhihao/PostTrain/models/hf_models/YuLan-MoE/hf/iter_0032780 --host 127.0.0.1 --port 30110 --tp 2 --trust-remote-code --disable-cuda-graph`
  - 验收：
    - `python scripts/run_acceptance.py --host 127.0.0.1 --port 30110`
  - 结果为 `ALL CHECKS PASSED`
- 历史基于测试 checkpoint 的 `sglang 0.5.9` 完整验收命令为：
  - `python scripts/run_acceptance.py --host 127.0.0.1 --port 30110`
  - 结果为 `ALL CHECKS PASSED`
- 当前没有证据表明 `0.5.9` 仍需要继承 `0.5.2` 的 hybrid linear attention backend 强制切换 patch
- 当前已有证据表明 `0.5.9` 需要自己的 Mamba conv cache dtype compat；不要把它误写成 `0.5.2` 的旧 `ModelRunner.init_memory_pool()` patch 原样继承
