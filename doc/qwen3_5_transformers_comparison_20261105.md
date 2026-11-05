# Qwen3.5-0.8B MATLAB / Transformers-PyTorch FP32 比对

日期：2026-11-05。当前入口使用 Transformers/PyTorch，默认只加载本地权重。

## 直接运行摘要脚本

在 MATLAB 中运行：

```matlab
TestQwen3_5Summarize
```

脚本默认配置为：

```matlab
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
allowDownload = false;
setupPython = true;
forceReload = false;
promptTemplate = "Summarize this: %s Summary:";
useChatTemplate = false;
maxNewTokens = 30;
```

`allowDownload=false` 时只检查并加载已有 safetensors，缺失文件会明确报错。
实际测试了缺失权重分支，确认未调用下载函数，也未创建模型目录。
只有显式打开该选项才允许下载模型。

脚本自动检查仓库内的 Python 环境，首次需要时在 `work/deps/qwen3_5_python/`
安装依赖；系统中供其他模型使用的 Python 包未升级。`setupPython=false` 可关闭包安装。
模型文件、Python 环境、输出均位于被 Git 忽略的仓库相对目录。

与 `TestQwen2Summarize.m` 的对应关系：

| 步骤 | 当前 Qwen3.5 入口 |
|---|---|
| 配置 | 本地路径、下载开关、输入文本、prompt、30-token 上限均位于脚本开头 |
| 权重准备 | 直接读取 BF16/F32 safetensors，展开到 FP32，无需转换 MAT 或 GGUF |
| MATLAB | 独立执行 24 层 MATLAB decoder 和 greedy 生成 |
| Python 参考 | 官方 Transformers 模型，PyTorch FP32 CPU，`do_sample=False` |
| 输出 | 双方摘要、生成耗时，展示文本时移除特殊 token |
| 变量与缓存 | 摘要、耗时、token、比较结果保留在工作区，匹配权重签名时复用 `mdl` |

实际独立生成 30 个 tokens，输入 IDs、生成 IDs、显示文本三项均为 `true`。
重复运行验证了缓存复用；运行后调用 `generateSummary_Qwen3_5(mdl, ...)`
也得到相同文本和 token IDs。可开启 `useChatTemplate` 使用 Qwen3.5 chat prompt。

显示耗时排除模型加载、分词和解码，为当前运行的实耗时间。

## 参考设置

- MATLAB R2024a Update 4，`24.1.0.2628055`，CPU。
- Transformers **5.3.0**；PyTorch **2.10.0+cpu**；Python **3.10.12**。
- 使用 `Qwen3_5ForConditionalGeneration.from_pretrained`，输入纯文本。
- `local_files_only=True`，参考子进程设置离线模式。
- 参数为 FP32，attention 使用 eager 实现，模型处于 eval/inference 模式。
- MATLAB 与 Python 均读取 `work/models/Qwen3.5-0.8B-FULL/` 的原始 checkpoint。
- MATLAB 运行 18 层 Gated DeltaNet 和 6 层完整注意力；视觉和 MTP 不参与文本计算。

## 数值回归

Python 通过真实模块 hooks 导出每一层的输出、attention 输入归一化、attention 输出、
MLP 输入归一化、MLP 输出，并保存最终 norm、全部 logits、GDN state、卷积 history
和完整注意力的 KV caches。每个 stage 比较 **170 个张量**。

逐元素条件固定为：

```text
abs(matlab - reference) <= 1e-3 + 1e-4 * abs(reference)
```

同时检查数值有限、全部位置 top-1 一致、decode 生成 IDs 一致。
逐层回归按参考生成的 token 续算；摘要脚本另外验证双方独立 greedy 生成。

| 用例 | Prefill tokens | Decode steps | 张量比较数 | 最大张量绝对误差 | 最大 logits 绝对误差 | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 单 token | 1 | 4 | 850 | 8.86917e-5 | 4.50611e-5 | 5/5 |
| 英文摘要 | 39 | 8 | 1530 | 4.00543e-5 | 2.38419e-5 | 47/47 |
| 中文对话 | 17 | 8 | 1530 | 9.53674e-5 | 4.86374e-5 | 25/25 |
| MATLAB 7-token 分块 prefill | 17 | 8 | 1530 | 7.62939e-5 | 4.83990e-5 | 25/25 |
| 较长 prefill | 241 | 4 | 850 | 3.62396e-5 | 2.25306e-5 | 245/245 |

五组全部通过，合计 **6290 项张量比较**、**347/347** 个位置 top-1 一致。
全词表 logits 共比较 **86167040 个元素**，最大绝对误差为
`4.863739013671875e-5`。本表包含 chunk-wise GDN 接入后的 auto 路径结果。
分块用例将 MATLAB 的七-token 分块结果与 PyTorch
完整 prefill 结果比较，再验证缓存 decode。

16 项算子测试通过，覆盖卷积分块、非零 GDN state 续算、partial RoPE、
GQA cache/因果性、模型/tokenizer 的 EOS 元数据，以及 chunk-wise GDN 的
块边界、短尾、强衰减、零 beta、末态与注意力接入。

## 复现与证据

自动准备本地参考并运行 MATLAB 数值比对：

```bash
python tools/prepare_qwen3_5.py
```

已有参考时直接重放：

```matlab
TestQwen3_5Numerics( ...
    fullfile('work', 'models', 'Qwen3.5-0.8B-FULL'), ...
    fullfile('work', 'tmp', 'qwen3_5_transformers', 'references'));
```

结果保存在 `work/tmp/qwen3_5_transformers/`：`summary.json`、
`references/*/matlab_comparison.json`、每阶段 MAT 参考，以及摘要请求/响应和运行日志。
`matlab_validation.log` 记录五组数值回归；`direct_run_validation.log` 记录
直接运行、重复运行的缓存复用、摘要接口调用和 Code Analyzer 检查。

参考模型 API 见 [Hugging Face 官方文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)，
实现版本固定为 [Transformers v5.3.0](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
