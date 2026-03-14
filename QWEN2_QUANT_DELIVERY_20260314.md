# Qwen2 Quant 交付说明（2026-03-14）

## 交付目标

本轮交付目标是将 Qwen2 量化摘要链路整理为可独立推送的源码闭包，并完成一次面向远端仓的导入、验证与推送。

## 已交付内容

- 新增 `+qwen2_quant/` 量化推理包，覆盖 GGUF、GPTQ、AWQ 三条链路。
- 新增 `generateSummary_Qwen2_quant.m` 统一摘要生成入口。
- 新增三个量化摘要测试脚本：
  - `TestQwen2QuantSummarize.m`
  - `TestQwen2GPTQQuantSummarize.m`
  - `TestQwen2AWQQuantSummarize.m`
- 新增 GPTQ/AWQ 所需 Python 桥接与导出工具：
  - `tools/prepare_qwen_hf_quant.py`
  - `tools/qwen2_hf_quant_infer.py`
- 为量化链路补齐 `+transformer/+layer/Mask.m`、`RoPE.m`、`Softmax.m` 三个底层依赖。

## 关键修复

### 1. GGUF 下载命令修复

`TestQwen2QuantSummarize.m` 中对 GGUF 文件名和目录名的字符串拼接做了修复，避免 MATLAB 字符数组尾随空格进入 Hugging Face 下载命令，导致 URL 非法。

### 2. GPTQ parity 测试模式修复

`TestQwen2GPTQQuantSummarize.m` 的 MATLAB-native 分支原先使用 `gptq_int4_quant_sim`。该模式会引入额外的激活量化仿真，更适合硬件近似研究，不适合拿来和 Python GPTQ 参考运行时做 exact parity。

现已切换为 `gptq_int4_matlab_sim`，用于按 GPTQ 打包权重做精确回放，从而更符合 parity 测试目标。

## 最小验证结果

### GGUF

- 入口：`matlab -batch TestQwen2QuantSummarize`
- 状态：通过
- 结果：`q8_0`、`q4_0`、`q4_k_m` 三种格式均可完成加载与摘要生成。

### AWQ

- 入口：`matlab -batch TestQwen2AWQQuantSummarize`
- 状态：通过
- 结果：Python reference 与 MATLAB-native 输出 `Exact Equal = true`

### GPTQ

- 入口：`matlab -batch TestQwen2GPTQQuantSummarize`
- 状态：通过
- 结果：Python reference 与 MATLAB-native 输出 `Exact Equal = true`
- 说明：MATLAB-native 分支已切换为 `gptq_int4_matlab_sim`，用于按 GPTQ 打包权重做精确回放；原 `gptq_int4_quant_sim` 更适合硬件近似研究，不适合 parity 测试

## 远端推送信息

- 目标仓：`https://github.com/liyang53719/transformer-models.git`
- 分支：`master`
- 已推送提交：`4c3e59d` `Add Qwen2 quantized summarize pipeline`

## 本轮后续项

- 若后续需要做硬件近似研究，建议保留 `gptq_int4_quant_sim` 供单独精度实验使用，不再直接作为 parity 测试模式。
- 若要继续推进，可新增一个独立测试脚本专门比较 `gptq_int4_matlab_sim` 与 `gptq_int4_quant_sim` 的生成偏差。