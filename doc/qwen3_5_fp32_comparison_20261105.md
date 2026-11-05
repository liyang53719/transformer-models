# 历史记录：Qwen3.5-0.8B MATLAB / llama.cpp FP32 数值比对

此页保留先前版本的实测记录。当前入口已切换至 Transformers/PyTorch，
以[当前数值报告](./qwen3_5_transformers_comparison_20261105.md)为准。
以下旧流程对应 Git 提交 `1375564`，不再作为当前运行入口。

日期：2026-11-05。实现位于 `+qwen3_5/`；范围为完整的 24 层文本 decoder。

## 实现与参考

- 模型：Qwen3.5-0.8B 原始 safetensors；复现时放在 `work/models/Qwen3.5-0.8B-FULL/`。
- MATLAB：R2024a Update 4，`24.1.0.2628055`，CPU。
- MATLAB 直接读取 BF16/F32 权重，展开为 FP32；矩阵乘法、激活、KV cache 和 GDN state 走浮点通路。
- llama.cpp 提交：`183d2a04c2a666187598abb771918cfbb40dcebb`。
- 从该提交归档隔离源码并构建 CPU library；外部 llama.cpp 工作区不作修改。
- 使用该版本原始转换器的 `--outtype f32 --no-nextn` 导出参考 GGUF，检查全部 **320 个张量均为 F32**。
- 参考设置：CPU、F32 KV cache、F32 recurrent state、关闭 flash attention；同一参考 binary 用于全部五组。
- Python `tokenizers` 与 llama.cpp 的 prompt token IDs 逐项一致。

算子包括 18 层 Gated DeltaNet、6 层带输出门控的 GQA、因果 depthwise convolution、
Q/K RMSNorm、partial RoPE、SiLU、gated RMSNorm、dense MLP 和 residual。
保留卷积 history、GDN state、KV cache 和绝对 position，实现整段及分块 prefill 与 decode。
checkpoint 的 RMSNorm `1 + weight` 与 GDN `-exp(A_log)` 在加载时处理；
GDN 的独立 gated norm 使用其原始乘法权重。

## 验收方法

沿用 `tools/prepare_tinyllama.py` 导出中间结果、logits 和 greedy tokens 的方式，
新增 `tools/prepare_qwen3_5.py`。数值参考由真实 llama.cpp 执行后通过 backend callback 导出；
MATLAB 独立执行 `+qwen3_5` 中的算子。

每一 stage 比较 356 个张量，包括所有 24 层输出、归一化、GDN Q/K/V、门控、
卷积 history、更新后的 GDN state、注意力输出、MLP 输出、最终 norm 和全词表 logits。
logits 比较覆盖每个 prefill/decode 位置的全部 248320 个元素。

事先固定的逐元素容差：

```text
abs(matlab - reference) <= 1e-3 + 1e-4 * abs(reference)
```

所有比较要求有限数值；同时要求全部位置的 top-1 及实际生成 token IDs 完全一致。
生成 token ID 一致后，Python 入口记录双方解码文本。

## 实测结果

| 用例 | Prefill tokens | Decode steps | 张量比较数 | 最大张量绝对误差 | 最大 logits 绝对误差 | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 单 token | 1 | 4 | 1780 | 6.86646e-5 | 3.79086e-5 | 5/5 |
| 英文光合作用摘要 | 39 | 8 | 3204 | 1.22070e-4 | 3.91006e-5 | 47/47 |
| 中文对话 | 17 | 8 | 3204 | 9.34601e-5 | 4.48823e-5 | 25/25 |
| 同一中文输入，7-token 分块 | 17 | 8 | 3916 | 9.34601e-5 | 5.76973e-5 | 25/25 |
| 较长 prefill | 241 | 4 | 1780 | 8.77380e-5 | 3.29018e-5 | 245/245 |

- 五组均通过，**13884 项张量比较，420132544 个标量比较**。
- logits 共比较 **86167040 个元素**，347 个位置的 top-1 为 **347/347**。
- 五组的 32 个生成 token IDs 逐项一致；中文整段/分块 prefill 的生成结果一致。
- 所有层输出的最大绝对误差：`9.298324584960938e-6`。
- 所有 logits 的最大绝对误差：`5.7697296142578125e-5`。
- 独立 MATLAB 文本入口 `generateSummary_Qwen3_5` 的英文 greedy 输出与参考 8 个 tokens 一致。
- 5 项算子单元测试通过；新增的 18 个 MATLAB 文件 Code Analyzer 无诊断。

算子测试覆盖因果卷积分块 history、非零初始 GDN state 的分块续算、
RoPE 未旋转通道、GQA cache 续算和未来 token 的因果屏蔽。

终态标记：

```text
QWEN3_5_ALL_PASS cases=5
QWEN3_5_TEXT_FRONTEND_PASS
QWEN3_5_FINAL_CHECK_PASS tests=5 analyzed_files=18
```

## 复现

### 补充：与 TestQwen2Summarize 同形式的摘要脚本

`TestQwen3_5Summarize.m` 提供独立文本生成对比：同一输入和 chat prompt，
双方使用 FP32、greedy、最多 30 个新 tokens，分别打印摘要和生成耗时。
Python 入口 `tools/qwen3_5_llama_infer.py` 调用 llama.cpp；MATLAB 独立运行
`generateSummary_Qwen3_5`。参考侧摘要模式不导出中间张量。

实跑结果：双方生成 **28 个 tokens（含 `<|im_end|>`）**，输入 IDs、生成 IDs、
文本三项比较均为 `true`。生成逻辑现读取 tokenizer 的 chat EOS，
避免在 `<|im_end|>` 之后继续输出。新增两项终止符测试后，当前单元测试为 **7/7**。
新增摘要脚本和终止符解析文件也通过 Code Analyzer。

输出位于 `work/tmp/qwen3_5/summarize/`，包括 `matlab_summary_comparison.json`。
显示的时间排除权重加载；MATLAB 文本入口还包括分词和解码时间，
这些实耗时间用于查看运行过程。

### 准备参考与重放

将模型放在 `work/models/Qwen3.5-0.8B-FULL/`，将 llama.cpp Git checkout
放在 `work/deps/llama.cpp/`，然后在仓库根目录执行。两处输入目录均被 Git 忽略。
以下使用可移植的相对路径；上述实测的模型内容由文末 SHA256 标识。

```bash
python tools/prepare_qwen3_5.py \
  --model work/models/Qwen3.5-0.8B-FULL \
  --llama-cpp work/deps/llama.cpp \
  --llama-revision 183d2a04c2a666187598abb771918cfbb40dcebb
```

已有参考数据时，直接在 MATLAB 执行：

```matlab
TestQwen3_5Numerics( ...
    fullfile('work', 'models', 'Qwen3.5-0.8B-FULL'), ...
    fullfile('work', 'tmp', 'qwen3_5', 'references'));
```

原始 binary tensors、生成 tokens、逐张量误差、JSON 总结和日志保留于
`work/tmp/qwen3_5/`，受现有 `.gitignore` 排除。主要证据：

- `provenance.json`：模型输入、参考 GGUF/binary/library 和 MATLAB 源码的 SHA256。
- `summary.json`：五组汇总与双方生成 IDs/文本。
- `references/*/matlab_comparison.json`：每个 stage、每个张量的误差与 pass。
- `matlab_final.log`：最终源码重放已有参考的完整 MATLAB 日志。
- `frontend_check.log`、`final_check.log`：文本生成入口和静态检查/单元测试。

源 safetensors SHA256：

```text
04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
```

参考 F32 GGUF SHA256：

```text
1d388b2e30e4833a986e468421489477d82db645abcee8b4dcbb91848194d05f
```

本次覆盖单序列文本、上述 prefill 长度及 4/8 步 decode。视觉编码器、MTP 和
Simulink/HDL 不属于此次实现或验收范围。
