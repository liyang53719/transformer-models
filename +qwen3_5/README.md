# Qwen3.5-0.8B MATLAB FP32

This package implements all 24 text-decoder layers: 18 Gated DeltaNet layers
and 6 gated full-attention layers. It reads local BF16/F32 safetensors into
FP32 MATLAB arrays and supports prefill, cached decode, and greedy generation.
Vision and MTP are outside the MATLAB implementation's scope.

## Run the summary example directly

Place the original checkpoint files in `work/models/Qwen3.5-0.8B-FULL/`, then
run this script in MATLAB:

```matlab
TestQwen3_5Summarize
```

The configuration at the top of the script includes:

```matlab
allowDownload = false;
setupPython = true;
forceReload = false;
promptTemplate = "Summarize this: %s Summary:";
useChatTemplate = false;
maxNewTokens = 30;
```

By default the script loads existing local weights. If any required model file
is missing, it reports the missing files and does not download weights.
Set `allowDownload = true` explicitly to permit downloading the configured
Hugging Face model. All example data paths are relative to this repository.

The first run prepares Python dependencies under the ignored
`work/deps/qwen3_5_python/` directory. This project environment uses
Transformers 5.3.0 and PyTorch; it does not upgrade the Python environment used
by the other model examples. Set `setupPython = false` to disable package setup.

The script follows `TestQwen2Summarize.m`: it displays a MATLAB summary,
an independent Transformers/PyTorch FP32 summary, and both generation times.
Both paths receive the same prompt, use greedy decoding, and stop at EOS or
the configured token limit. Special tokens are omitted from displayed text.
The script checks exact input-ID, generated-ID, and text equality.

The variables `mdl`, `summary_matlab`, `summary_python`, `t_matlab`, `t_python`,
`inputIds`, `matlabIds`, `pythonIds`, and `comparison` remain in the workspace.
The model is reused on subsequent runs when its checkpoint signature matches.
Use `forceReload = true` to reload it. `mdl.Tokenizer` also provides `encode`
and `decode` for use with `generateSummary_Qwen3_5`.

Set `useChatTemplate = true` for a Qwen3.5 chat prompt with thinking disabled.
The default plain prompt matches the existing Qwen2 example's format.
Generation times exclude model loading, tokenization, and text decoding.
These elapsed times describe the current run.

Results and request/response JSON files are saved in
`work/tmp/qwen3_5_transformers/summarize/`.

## MATLAB interfaces and operators

```matlab
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
parameters = qwen3_5.load(modelFolder);
[logits, state] = qwen3_5.model(int32([760 6511 314 9338 369]), parameters);
[~, next] = max(logits(:, end));
[logits, state] = qwen3_5.model(int32(next-1), parameters, state);
```

Token IDs are zero based, and logits are vocabulary-by-time. An empty state
starts a new sequence. `qwen3_5.generate` runs greedy decoding, and
`generateSummary_Qwen3_5` accepts the repository's `Parameters`/`Tokenizer`
model structure. Weight loading and token-ID inference use base MATLAB.

| Operator | Behavior |
|---|---|
| `rmsNormalization` | First-dimension RMSNorm with FP32 squares and a wide reduction |
| `l2Normalization` | GDN query/key normalization with epsilon inside the squared norm |
| `causalDepthwiseConv` | Four causal taps and three samples of history |
| `gatedDeltaNet` | Decay, delta correction, state update, and query scaling |
| `linearAttention` | QKV/Z/A/B projections, convolution, GDN, gated norm, output projection |
| `rotaryEmbedding` | Text positions; rotates 64 of 256 head channels with theta 1e7 |
| `fullAttention` | Q/gate split, Q/K norm, causal GQA, and sigmoid output gate |
| `gatedMLP` | SiLU-gated up projection followed by down projection |
| `block` | Attention, MLP, and both residual connections |

The loader converts checkpoint RMSNorm offset weights to `1 + weight` once,
except for GDN's already multiplicative norm. `A_log` becomes `-exp(A_log)`.
The output head shares the token embedding weights. State preserves GDN
convolution/SSM history and the full-attention KV cache. Stop IDs include
the chat EOS declared by tokenizer metadata.

## Numerical comparison against Transformers/PyTorch

Run the local-only preparation and full numerical comparison:

```bash
python tools/prepare_qwen3_5.py
```

The default model path is `work/models/Qwen3.5-0.8B-FULL/`. No model download is
attempted unless `--allow-download` is supplied. Add `--setup-only` to check
the local model and prepare Python without running numerical cases.

`tools/qwen3_5_transformers.py` executes the official Transformers model on CPU
with FP32 parameters and eager attention. It exports reference layer outputs,
normalizations, attention/MLP outputs, recurrent state, convolution history,
KV caches, and every vocabulary logit. MATLAB independently runs its operators.

Five cases cover one token, English, Chinese, MATLAB chunked prefill, and a
241-token prefill, each followed by cached decode. For the chunked case, MATLAB
receives seven-token chunks and is compared with the full-prefill PyTorch
reference. Each element must satisfy `1e-3 + 1e-4 * abs(reference)`, with finite
values and exact top-1/generated-ID agreement required.

To replay existing references:

```matlab
TestQwen3_5Numerics( ...
    fullfile('work', 'models', 'Qwen3.5-0.8B-FULL'), ...
    fullfile('work', 'tmp', 'qwen3_5_transformers', 'references'));
```

See the [measured comparison report](../doc/qwen3_5_transformers_comparison_20261105.md).
The reference implementation is documented by
[Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
