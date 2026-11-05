# Transformer Models for MATLAB
[![Upstream CircleCI](https://img.shields.io/circleci/build/github/matlab-deep-learning/transformer-models?label=upstream%20tests)](https://app.circleci.com/pipelines/github/matlab-deep-learning/transformer-models)
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=liyang53719/transformer-models)

This fork of [matlab-deep-learning/transformer-models](https://github.com/matlab-deep-learning/transformer-models) implements transformer inference in MATLAB. It retains BERT, FinBERT, and GPT-2 and adds TinyLlama, Qwen2/Qwen2.5, quantized Qwen2 inference, and an FP32 Qwen3.5-0.8B text decoder with numerical comparisons against llama.cpp.

## Models and entry points

| Model | MATLAB entry points | Examples and validation |
|---|---|---|
| BERT / FinBERT | `bert`, `bert.model`, `finbert`, `finbert.sentimentModel` | Classification, fine-tuning, and masked-token examples below |
| GPT-2 | `gpt2`, `generateSummary` | [GPT-2 summary example](./SummarizeTextUsingTransformersExample.m) |
| TinyLlama | `llama.load`, `llama.model`, `generateSummary_tinyllama` | [MATLAB/Python comparison](./TestTinyLlamaSummarize.m), [Python preparation](./tools/prepare_tinyllama.py) |
| Qwen2 / Qwen2.5 | `qwen2.load`, `qwen2.model`, `generateSummary_Qwen2` | [MATLAB/Python comparison](./TestQwen2Summarize.m) |
| Qwen2 quantized | `qwen2_quant.load_gguf`, `qwen2_quant.load_hf_quant`, `generateSummary_Qwen2_quant` | [GGUF test](./TestQwen2QuantSummarize.m), [GPTQ test](./TestQwen2GPTQQuantSummarize.m), [AWQ test](./TestQwen2AWQQuantSummarize.m), [delivery notes](./QWEN2_QUANT_DELIVERY_20260314.md) |
| Qwen3.5-0.8B FP32 | `qwen3_5.load`, `qwen3_5.model`, `generateSummary_Qwen3_5` | [Package guide](./+qwen3_5/README.md), [Python preparation](./tools/prepare_qwen3_5.py), [numerical comparison](./TestQwen3_5Numerics.m) |

Current development focuses on MATLAB inference and numerical validation. Existing Simulink/HDL research assets are retained; those workflows are not currently maintained.

## Translations
* [日本語](./README_JP.md)

## Requirements
### BERT and FinBERT
- MATLAB R2021a or later
- Deep Learning Toolbox
- Text Analytics Toolbox
### GPT-2
- MATLAB R2020a or later
- Deep Learning Toolbox

### Qwen3.5-0.8B FP32

- MATLAB R2024a was used for validation. Loading weights and evaluating token IDs uses base MATLAB.
- Python with `tokenizers` is needed for the optional text tokenizer.
- The llama.cpp reference workflow also needs Python with `torch`, `numpy`, and `safetensors`, the converter dependencies from the selected llama.cpp checkout, CMake, and a C++17 compiler.
- A local copy of the original Qwen3.5-0.8B checkpoint is required. BF16/F32 checkpoint weights are evaluated through FP32 operators.

TinyLlama, Qwen2, and quantized Qwen2 have their Python preparation/runtime dependencies documented in their linked scripts and delivery notes.

## Getting Started
Download or [clone](https://www.mathworks.com/help/matlab/matlab_prog/use-source-control-with-projects.html#mw_4cc18625-9e78-4586-9cc4-66e191ae1c2c) this repository to your machine and open it in MATLAB. 

## Qwen3.5-0.8B quick start

Run the following examples from the repository root. Place the checkpoint's `config.json`, `tokenizer.json`, safetensors index, and weight shards in `work/models/Qwen3.5-0.8B-FULL/`. Place a llama.cpp Git checkout in `work/deps/llama.cpp/` for reference preparation. Both directories are ignored by Git.

```matlab
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
mdl.Parameters = qwen3_5.load(modelFolder);
mdl.Tokenizer = qwen3_5.tokenizer.QwenTokenizer(modelFolder);
[text, ids] = generateSummary_Qwen3_5(mdl, 'The capital of France is', ...
    'TopK', 1, 'MaxNewTokens', 8);
disp(text)
```

The implementation executes all 24 text-decoder layers: 18 Gated DeltaNet and 6 gated full-attention layers. It includes causal convolution, RMSNorm, partial RoPE, dense MLP, convolution/SSM state, and FP32 KV cache. Token IDs are zero based. Generation uses greedy decoding; supply a formatted chat prompt when needed. The scope is text inference, without the vision encoder or MTP.

To reproduce the llama.cpp comparison:

```bash
python tools/prepare_qwen3_5.py \
  --model work/models/Qwen3.5-0.8B-FULL \
  --llama-cpp work/deps/llama.cpp \
  --llama-revision 183d2a04c2a666187598abb771918cfbb40dcebb
```

The Python entry point follows the existing TinyLlama preparation workflow: export intermediate tensors, logits, and greedy token IDs, then run the independent MATLAB implementation. Reference weights, KV cache, and recurrent state are FP32. Relative command-line paths resolve from this repository root. Logs and generated artifacts are stored in the ignored `work/tmp/qwen3_5/` directory.

The [2026-11-05 numerical report](./doc/qwen3_5_fp32_comparison_20261105.md) records five passing cases: single-token input, English summary, Chinese chat, chunked prefill, and a 241-token prefill. Across 13,884 tensor comparisons, top-1 IDs matched at all 347 evaluated positions; maximum logits absolute error was `5.77e-5`. Five operator tests passed, and all 18 new MATLAB files had no Code Analyzer diagnostics.

To replay prepared references in MATLAB:

```matlab
TestQwen3_5Numerics( ...
    fullfile('work', 'models', 'Qwen3.5-0.8B-FULL'), ...
    fullfile('work', 'tmp', 'qwen3_5', 'references'));
```

## Functions
### bert
`mdl = bert` loads a pretrained BERT transformer model and if necessary, downloads the model weights. The output `mdl` is structure with fields `Tokenizer` and `Parameters` that contain the BERT tokenizer and the model parameters, respectively.

`mdl = bert("Model",modelName)` specifies which BERT model variant to use:
- `"base"` (default) - A 12 layer model with hidden size 768.
- `"multilingual-cased"` - A 12 layer model with hidden size 768. The tokenizer is case-sensitive. This model was trained on multi-lingual data.
- `"medium"` - An 8 layer model with hidden size 512. 
- `"small"` - A 4 layer model with hidden size 512.
- `"mini"` - A 4 layer model with hidden size 256.
- `"tiny"` - A 2 layer model with hidden size 128.
- `"japanese-base"` - A 12 layer model with hidden size 768, pretrained on texts in the Japanese language.
- `"japanese-base-wwm"` - A 12 layer model with hidden size 768, pretrained on texts in the Japanese language. Additionally, the model is trained with the whole word masking enabled for the masked language modeling (MLM) objective.

### bert.model
`Z = bert.model(X,parameters)` performs inference with a BERT model on the input `1`-by-`numInputTokens`-by-`numObservations` array of encoded tokens with the specified parameters. The output `Z` is an array of size (`NumHeads*HeadSize`)-by-`numInputTokens`-by-`numObservations`. The element `Z(:,i,j)` corresponds to the BERT embedding of input token `X(1,i,j)`.

`Z = bert.model(X,parameters,Name,Value)` specifies additional options using one or more name-value pairs:
- `"PaddingCode"` - Positive integer corresponding to the padding token. The default is `1`.
- `"InputMask"` - Mask indicating which elements to include for computation, specified as a logical array the same size as `X` or as an empty array. The mask must be false at indices positions corresponds to padding, and true elsewhere. If the mask is `[]`, then the function determines padding according to the `PaddingCode` name-value pair. The default is `[]`.
- `"DropoutProb"` - Probability of dropout for the output activation. The default is `0`.
- `"AttentionDropoutProb"` - Probability of dropout used in the attention layer. The default is `0`.
- `"Outputs"` - Indices of the layers to return outputs from, specified as a vector of positive integers, or `"last"`. If `"Outputs"` is `"last"`, then the function returns outputs from the final encoder layer only. The default is `"last"`.
- `"SeparatorCode"` - Separator token specified as a positive integer. The default is `103`.

### finbert
`mdl = finbert` loads a pretrained BERT transformer model for sentiment analysis of financial text. The output `mdl` is structure with fields `Tokenizer` and `Parameters` that contain the BERT tokenizer and the model parameters, respectively.

`mdl = finbert("Model",modelName)` specifies which FinBERT model variant to use:
- `"sentiment-model"` (default) - The fine-tuned sentiment classifier model.
- `"language-model"` - The FinBERT pretrained language model, which uses a BERT-Base architecture.

### finbert.sentimentModel
`sentiment = finbert.sentimentModel(X,parameters)` classifies the sentiment of the input `1`-by-`numInputTokens`-by-`numObservations` array of encoded tokens with the specified parameters. The output sentiment is a categorical array with categories `"positive"`, `"neutral"`, or `"negative"`.

`[sentiment, scores] = finbert.sentimentModel(X,parameters)` also returns the corresponding sentiment scores in the range `[-1 1]`.

### gpt2
`mdl = gpt2` loads a pretrained GPT-2 transformer model and if necessary, downloads the model weights.

### Qwen3.5-0.8B (FP32 text decoder)
`parameters = qwen3_5.load(modelFolder)` reads local BF16/F32 safetensors into FP32 MATLAB arrays. `qwen3_5.model` implements all 24 text-decoder layers, including Gated DeltaNet, gated GQA, and cached decoding. Text generation is available through `generateSummary_Qwen3_5`.

See the [Qwen3.5 package guide](./+qwen3_5/README.md) for usage and the Python preparation workflow, and the [FP32 numerical comparison report](./doc/qwen3_5_fp32_comparison_20261105.md) for MATLAB/llama.cpp results. This path covers text inference; the vision encoder and MTP are outside its scope.

### generateSummary
`summary = generateSummary(mdl,text)` generates a summary of the string or `char` array `text` using the transformer model `mdl`. The output summary is a char array.

`summary = generateSummary(mdl,text,Name,Value)` specifies additional options using one or more name-value pairs.

* `"MaxSummaryLength"` - The maximum number of tokens in the generated summary. The default is 50.
* `"TopK"` - The number of tokens to sample from when generating the summary. The default is 2.
* `"Temperature"` - Temperature applied to the GPT-2 output probability distribution. The default is 1.
* `"StopCharacter"` - Character to indicate that the summary is complete. The default is `"."`.

## Example: Classify Text Data Using BERT
The simplest use of a pretrained BERT model is to use it as a feature extractor. In particular, you can use the BERT model to convert documents to feature vectors which you can then use as inputs to train a deep learning classification network.

The example [`ClassifyTextDataUsingBERT.m`](./ClassifyTextDataUsingBERT.m) shows how to use a pretrained BERT model to classify failure events given a data set of factory reports. This example requires the `factoryReports.csv` data set from the Text Analytics Toolbox example [Prepare Text Data for Analysis](https://www.mathworks.com/help/textanalytics/ug/prepare-text-data-for-analysis.html).

## Example: Fine-Tune Pretrained BERT Model
To get the most out of a pretrained BERT model, you can retrain and fine tune the BERT parameters weights for your task.

The example [`FineTuneBERT.m`](./FineTuneBERT.m) shows how to fine-tune a pretrained BERT model to classify failure events given a data set of factory reports. This example requires the `factoryReports.csv` data set from the Text Analytics Toolbox example [Prepare Text Data for Analysis](https://www.mathworks.com/help/textanalytics/ug/prepare-text-data-for-analysis.html).

The example [`FineTuneBERTJapanese.m`](./FineTuneBERTJapanese.m) shows the same workflow using a pretrained Japanese-BERT model. This example requires the `factoryReportsJP.csv` data set from the Text Analytics Toolbox example [Analyze Japanese Text Data](https://www.mathworks.com/help/textanalytics/ug/analyze-japanese-text.html), available in R2023a or later.

## Example: Analyze Sentiment with FinBERT
FinBERT is a sentiment analysis model trained on financial text data and fine-tuned for sentiment analysis.

The example [`SentimentAnalysisWithFinBERT.m`](./SentimentAnalysisWithFinBERT.m) shows how to classify the sentiment of financial news reports using a pretrained FinBERT model.

## Example: Predict Masked Tokens Using BERT and FinBERT
BERT models are trained to perform various tasks. One of the tasks is known as masked language modeling which is the task of predicting tokens in text that have been replaced by a mask value.

The example [`PredictMaskedTokensUsingBERT.m`](./PredictMaskedTokensUsingBERT.m) shows how to predict masked tokens and calculate the token probabilities using a pretrained BERT model.

The example [`PredictMaskedTokensUsingFinBERT.m`](./PredictMaskedTokensUsingFinBERT.m) shows how to predict masked tokens for financial text using and calculate the token probabilities using a pretrained FinBERT model.

## Example: Summarize Text Using GPT-2
Transformer networks such as GPT-2 can be used to summarize a piece of text. The trained GPT-2 transformer can generate text given an initial sequence of words as input. The model was trained on comments left on various web pages and internet forums.

Because lots of these comments themselves contain a summary indicated by the statement "TL;DR" (Too long, didn't read), you can use the transformer model to generate a summary by appending "TL;DR" to the input text. The `generateSummary` function takes the input text, automatically appends the string `"TL;DR"` and generates the summary.

The example [`SummarizeTextUsingTransformersExample.m`](./SummarizeTextUsingTransformersExample.m) shows how to summarize a piece of text using GPT-2.
