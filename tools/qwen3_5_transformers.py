#!/usr/bin/env python3
"""Local FP32 Transformers/PyTorch reference and tokenizer for MATLAB."""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from qwen3_5_runtime import local_checkpoint, native_environment, resolve_repo_path, write_json


def tokenizer_and_prompt(request):
    from transformers import AutoTokenizer
    model, _ = local_checkpoint(request["model_folder"], "Qwen/Qwen3.5-0.8B", False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompt = request.get("prompt", "")
    if request.get("use_chat_template", False):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return tokenizer, prompt


def load_model(request):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    torch.set_num_threads(8)
    model, _ = local_checkpoint(request["model_folder"], "Qwen/Qwen3.5-0.8B", False)
    network, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    missing = [key for key in info["missing_keys"] if "language_model" in key or "lm_head" in key]
    if missing:
        raise RuntimeError("Missing decoder weights: " + str(missing))
    network.eval()
    if any(parameter.dtype != torch.float32 for parameter in network.parameters()):
        raise RuntimeError("Reference model parameters must be FP32")
    return network


def stop_ids(tokenizer, network):
    value = network.config.text_config.eos_token_id
    ids = list(value) if isinstance(value, list) else [value]
    vocabulary = tokenizer.get_vocab()
    for token in [tokenizer.eos_token, "<|fim_suffix|>"]:
        if token in vocabulary:
            ids.append(vocabulary[token])
    return list(dict.fromkeys(ids))


def generate(request):
    import torch
    import transformers
    tokenizer, prompt = tokenizer_and_prompt(request)
    network = load_model(request)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([ids], dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        output = network.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=int(request.get("max_new_tokens", 30)), do_sample=False,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            eos_token_id=stop_ids(tokenizer, network), pad_token_id=tokenizer.pad_token_id,
            use_cache=True)
    elapsed = time.perf_counter() - started
    generated = output[0, len(ids):].tolist()
    return {"ok": True, "backend": "Transformers/PyTorch FP32 CPU",
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "prompt_token_ids": ids, "generated_token_ids": generated,
            "summary": tokenizer.decode(generated, skip_special_tokens=True),
            "generation_seconds": elapsed, "generated_tokens": len(generated)}


def numerical_references(request):
    import numpy as np
    import scipy.io
    import torch
    tokenizer, _ = tokenizer_and_prompt(request)
    network = load_model(request)
    decoder = network.model.language_model
    captured = {}

    def capture(name):
        def hook(module, arguments, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value[0].detach().cpu().numpy().T.copy()
        return hook

    handles = [layer.register_forward_hook(capture(f"l_out_{i}"))
               for i, layer in enumerate(decoder.layers)]
    for i, layer in enumerate(decoder.layers):
        handles.append(layer.input_layernorm.register_forward_hook(capture(f"attn_norm_{i}")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(capture(f"attn_post_norm_{i}")))
        handles.append(layer.mlp.register_forward_hook(capture(f"ffn_out_{i}")))
        attention = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        handles.append(attention.register_forward_hook(capture(f"attention_{i}")))
    handles.append(decoder.norm.register_forward_hook(capture("result_norm")))
    chinese = ("<|im_start|>user\n请用一句话解释光合作用。<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    english = ("Summarize this: The process of photosynthesis is how plants convert light energy "
               "into chemical energy. Chlorophyll absorbs sunlight and uses it to convert "
               "carbon dioxide and water into glucose. Summary:")
    cases = [("single_token", "Hello", 0), ("english", english, 0),
             ("chinese", chinese, 0), ("chinese_chunked", chinese, 7),
             ("long_prefill", "The quick brown fox jumps over the lazy dog. " * 24, 0)]
    root = resolve_repo_path(request["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, prompt, chunk in cases:
        folder = root / name
        folder.mkdir(exist_ok=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        stages = []
        cache = None
        position = 0
        current = ids
        generated = []
        steps = 4 if name in ["single_token", "long_prefill"] else 8
        for step in range(steps + 1):
            captured.clear()
            input_ids = torch.tensor([current], dtype=torch.long)
            with torch.inference_mode():
                result = network(input_ids=input_ids, past_key_values=cache, use_cache=True,
                                 cache_position=torch.arange(position, position + len(current)),
                                 return_dict=True)
            cache = result.past_key_values
            tensors = dict(captured)
            tensors["input_ids"] = np.array(current, dtype=np.int32)[None, :]
            tensors["logits"] = result.logits[0].detach().cpu().numpy().T.copy()
            for i, kind in enumerate(network.config.text_config.layer_types):
                if kind == "linear_attention":
                    tensors[f"ssm_{i}"] = cache.recurrent_states[i][0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                    tensors[f"conv_{i}"] = cache.conv_states[i][0, :, -3:].detach().cpu().numpy().copy()
                else:
                    tensors[f"keys_{i}"] = cache.key_cache[i][0].detach().cpu().numpy().transpose(2, 0, 1).copy()
                    tensors[f"values_{i}"] = cache.value_cache[i][0].detach().cpu().numpy().transpose(2, 0, 1).copy()
            filename = f"stage_{step:03d}.mat"
            scipy.io.savemat(folder / filename, tensors, do_compression=False)
            stages.append(filename)
            position += len(current)
            if step < steps:
                current = [int(result.logits[0, -1].argmax())]
                generated += current
        metadata = {"name": name, "MatlabPrefillChunk": chunk, "Stages": stages,
                    "PrefillTokens": len(ids), "GeneratedTokens": generated,
                    "GeneratedText": tokenizer.decode(generated, skip_special_tokens=True)}
        write_json(folder / "case_metadata.json", metadata)
        summaries.append(metadata)
        print(f"TRANSFORMERS_REFERENCE {name} prefill={len(ids)} decode={steps}", flush=True)
    for handle in handles:
        handle.remove()
    write_json(root / "reference_cases.json", summaries)
    return {"ok": True, "backend": "Transformers/PyTorch FP32 CPU", "cases": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["tokenize", "decode", "generate", "numerics"], required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    code = 0
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        os.environ.update(native_environment(request.get("matlab_root")))
        if args.action == "generate":
            response = generate(request)
        elif args.action == "numerics":
            response = numerical_references(request)
        else:
            tokenizer, prompt = tokenizer_and_prompt(request)
            response = {"ok": True}
            if args.action == "tokenize":
                response.update(prompt=prompt, token_ids=tokenizer.encode(prompt, add_special_tokens=False))
            else:
                response["summary"] = tokenizer.decode(request["token_ids"], skip_special_tokens=True)
    except Exception as error:
        response = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
        code = 1
    write_json(args.response, response)
    if code:
        print(response["traceback"], flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
