# SGLang Qwen3 Next Plugin

这是一个用于替换 `Qwen3NextForCausalLM` 实现的 SGLang 外部插件仓库。目标不是演示级插件，而是让自定义 Qwen3 Next checkpoint 在不同版本的 `sglang` 中都能以原生插件入口稳定运行。

当前 release 版本：`0.2.0`

## 功能特性

- ✅ 同时适配 `sglang 0.5.2` 与 `sglang 0.5.9`
- ✅ 主入口统一为 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
- ✅ 仓库按 `upstream / variants / compat / versioning` 分层组织
- ✅ 已支持 checkpoint 显式 `layer_types`
- ✅ 已完成真实服务级生成与 logprob 验收
- ✅ 提供可复用的轻量验收与完整验收脚本

## 安装

```bash
pip install -e .
```

或：

```bash
pip install .
```

## 使用方法

推荐的主入口：

```bash
source /path/to/venv/bin/activate
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin

python -m sglang.launch_server \
  --model-path <your_model_path> \
  --host 127.0.0.1 \
  --port 30110 \
  --tp 1
```

注意：

- 环境变量必须在 Python 进程启动前设置
- `sglang 0.5.2` 仍需要仓库内的接入层兼容逻辑
- `sglang 0.5.9` 已原生支持外部模型包发现
- 若新环境中存在与当前 `torch` ABI 不兼容的 `vllm`，应先移除再做插件验收

## 当前支持状态

- `sglang 0.5.2`
  - 插件发现、registry 接管、自然语言生成、logprob 验收均已通过
- `sglang 0.5.9`
  - 插件发现、registry 接管、真实服务启动、自然语言生成、logprob 验收均已通过

## 自定义修改

本仓库当前包含两类修改。

1. 公共 checkpoint 结构定制
   - `GemmaRMSNorm -> RMSNorm`
   - 移除 `q_norm / k_norm`
   - `qkv_proj` 使用 bias
   - `attn_output_gate` 默认关闭
   - `num_experts=0` 时走普通 MLP 路径
   - `num_experts=0` 时 expert-location 兜底为 1 个逻辑 expert

2. 按版本区分的运行时 compat
   - `sglang 0.5.2`
     - `layer_types` compat
     - hybrid linear attention backend 切换
     - `ModelRunner.init_memory_pool()` dtype patch
     - `sitecustomize.py` / `env_override.py` 接入层
   - `sglang 0.5.9`
     - `layer_types` compat
     - Hybrid GDN / Mamba conv cache dtype 同步
     - 当前没有证据表明还需要继承 `0.5.2` 的 hybrid backend 强制切换 patch

详细原因与升级复核点请看：

- `progress.md`
- `changelog.md`
- `.codex/skills/sglang-plugin-version-adaptation/SKILL.md`

## 项目结构

```text
sglang_qwen3_next_plugin/
├── pyproject.toml
├── README.md
├── progress.md
├── changelog.md
├── sitecustomize.py
├── scripts/
│   ├── check_plugin_import.py
│   ├── run_acceptance.py
│   ├── validate_generation.py
│   └── validate_logprob.py
├── sglang_qwen3_next_plugin/
│   ├── __init__.py
│   ├── qwen3_next.py
│   ├── versioning.py
│   ├── env_override.py
│   ├── compat/
│   ├── upstream/
│   │   ├── sglang_0_5_2/
│   │   └── sglang_0_5_9/
│   └── variants/
│       ├── sglang_0_5_2.py
│       └── sglang_0_5_9.py
├── tests/
└── .codex/skills/sglang-plugin-version-adaptation/
```

说明：

- `qwen3_next.py` 现在是按版本分发的入口，不再是单一版本实现
- `variants/` 放各版本实际模型实现
- `compat/` 放各版本运行时兼容补丁
- `upstream/` 放各版本上游快照，作为升级对照基线

## 验证

### 插件导入契约

```bash
source /path/to/venv/bin/activate
XDG_CACHE_HOME=/tmp HOME=/tmp python test_plugin.py
```

### 插件 registry 接管

```bash
source /path/to/venv/bin/activate
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin \
XDG_CACHE_HOME=/tmp HOME=/tmp \
python scripts/check_plugin_import.py
```

### 自然语言生成

服务启动后执行：

```bash
source /path/to/venv/bin/activate
python scripts/validate_generation.py --host 127.0.0.1 --port 30110
```

### 快速聊天冒烟测试

如果只想快速打一条 OpenAI 兼容 `chat/completions` 请求，可以执行：

```bash
bash scripts/chat_smoke.sh
```

默认会请求 `127.0.0.1:30110`，发送提示词“简短介绍一下你自己，用中文回答”。常用覆盖方式：

```bash
CHAT_PORT=30110 CHAT_PROMPT="简短介绍一下你自己，用中文回答" bash scripts/chat_smoke.sh
```

### logprob / 近似 lm loss

服务启动后执行：

```bash
source /path/to/venv/bin/activate
python scripts/validate_logprob.py --host 127.0.0.1 --port 30110
```

### 一条命令串联验收

仅做本地契约与 registry 检查：

```bash
source /path/to/venv/bin/activate
python scripts/run_acceptance.py --skip-runtime
```

对已启动服务做完整验收：

```bash
source /path/to/venv/bin/activate
python scripts/run_acceptance.py --host 127.0.0.1 --port 30110
```

当前 `sglang 0.5.9` 真实环境已实测返回 `ALL CHECKS PASSED`。YuLan-MoE Hybrid GDN checkpoint 在 `tp=2`、两张 H100 80GB 上完成真实启动、生成和 logprob 验收。

## 开发

### 添加自定义功能

1. 先确认目标版本应修改哪个 `variants/sglang_x_y_z.py`
2. 结构性定制优先写入对应 `variants/`
3. 运行时缺口优先写入对应 `compat/sglang_x_y_z.py`
4. 重新安装插件：`pip install -e .`
5. 运行验收脚本

### 调试

如果插件未加载，请检查：

1. 插件包是否已重新安装：`pip install -e .`
2. 环境变量是否在 Python 启动前设置
3. `config.json` 的 `architectures` 是否为 `["Qwen3NextForCausalLM"]`
4. `scripts/check_plugin_import.py` 是否显示 registry 已指向插件实现
5. `scripts/validate_generation.py` 与 `scripts/validate_logprob.py` 是否仍返回 `PASS`

若后续需要给仓库新增 `sglang` 版本支持，请优先参考：

- `.codex/skills/sglang-plugin-version-adaptation/SKILL.md`

## 发布说明

- 当前 release：`0.2.0`
- 该版本首次把仓库整理为多版本插件结构
- 该版本确认：
  - `sglang 0.5.2` 验收通过
  - `sglang 0.5.9` 验收通过

## 版本兼容性

- SGLang: `0.5.2`, `0.5.9`
- Python: `>= 3.10`

## 许可证

请参考 SGLang 的许可证要求。
