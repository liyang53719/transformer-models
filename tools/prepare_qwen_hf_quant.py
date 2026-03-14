#!/usr/bin/env python3
import argparse
import os

import numpy as np
import scipy.io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_qwen_hf_quant(model_name_or_path: str, output_mat: str, local_files_only: bool, trust_remote_code: bool):
    _clear_socks_proxy_env()

    tok_kwargs = {"trust_remote_code": trust_remote_code}
    mdl_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float32,
        "attn_implementation": "eager",
    }
    if local_files_only:
        tok_kwargs["local_files_only"] = True
        mdl_kwargs["local_files_only"] = True

    try:
        _ = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **mdl_kwargs)
    except Exception:
        if not local_files_only:
            raise
        tok_kwargs.pop("local_files_only", None)
        mdl_kwargs.pop("local_files_only", None)
        _ = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **mdl_kwargs)

    cfg = model.config
    mat_params = {
        "NumLayers": int(cfg.num_hidden_layers),
        "HiddenSize": int(cfg.hidden_size),
        "NumHeads": int(cfg.num_attention_heads),
        "NumKVHeads": int(cfg.num_key_value_heads),
        "HeadDim": int(cfg.hidden_size // cfg.num_attention_heads),
        "VocabSize": int(cfg.vocab_size),
        "RopeTheta": float(getattr(cfg, "rope_theta", 1000000.0)),
    }

    mat_params["embed_tokens"] = model.model.embed_tokens.weight.detach().cpu().numpy().T.astype(np.float32)

    for i, layer in enumerate(model.model.layers):
        p = f"layer_{i}_"
        _export_linear(mat_params, f"{p}self_attn_q_proj", layer.self_attn.q_proj)
        _export_linear(mat_params, f"{p}self_attn_k_proj", layer.self_attn.k_proj)
        _export_linear(mat_params, f"{p}self_attn_v_proj", layer.self_attn.v_proj)
        _export_linear(mat_params, f"{p}self_attn_o_proj", layer.self_attn.o_proj)

        if layer.self_attn.q_proj.bias is not None:
            mat_params[f"{p}self_attn_q_bias"] = layer.self_attn.q_proj.bias.detach().cpu().numpy().astype(np.float32)
        if layer.self_attn.k_proj.bias is not None:
            mat_params[f"{p}self_attn_k_bias"] = layer.self_attn.k_proj.bias.detach().cpu().numpy().astype(np.float32)
        if layer.self_attn.v_proj.bias is not None:
            mat_params[f"{p}self_attn_v_bias"] = layer.self_attn.v_proj.bias.detach().cpu().numpy().astype(np.float32)
        if layer.self_attn.o_proj.bias is not None:
            mat_params[f"{p}self_attn_o_bias"] = layer.self_attn.o_proj.bias.detach().cpu().numpy().astype(np.float32)

        _export_linear(mat_params, f"{p}mlp_gate_proj", layer.mlp.gate_proj)
        _export_linear(mat_params, f"{p}mlp_up_proj", layer.mlp.up_proj)
        _export_linear(mat_params, f"{p}mlp_down_proj", layer.mlp.down_proj)

        mat_params[f"{p}input_layernorm"] = layer.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
        mat_params[f"{p}post_attention_layernorm"] = layer.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)

    mat_params["norm"] = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
    _export_linear(mat_params, "lm_head", model.lm_head)

    os.makedirs(os.path.dirname(output_mat) or ".", exist_ok=True)
    scipy.io.savemat(output_mat, mat_params, do_compression=False)


def _extract_linear_weight(mod):
    if hasattr(mod, "weight"):
        w = mod.weight.detach().cpu().numpy().astype(np.float32)
        return _normalize_linear_layout(mod, w)
    if hasattr(mod, "dequantize_weight"):
        w = mod.dequantize_weight()
        w = w.detach().cpu().numpy().astype(np.float32)
        return _normalize_linear_layout(mod, w)
    raise AttributeError(f"Unsupported linear module type for weight export: {type(mod)}")


def _export_linear(dst, prefix, mod):
    if hasattr(mod, "qweight") and hasattr(mod, "qzeros") and hasattr(mod, "scales") and hasattr(mod, "g_idx"):
        quant_type = _infer_quant_type(mod)
        dst[f"{prefix}_quant_type"] = np.array([[quant_type]], dtype=object)
        dst[f"{prefix}_qweight"] = mod.qweight.detach().cpu().numpy().astype(np.int32)
        dst[f"{prefix}_qzeros"] = mod.qzeros.detach().cpu().numpy().astype(np.int32)
        dst[f"{prefix}_scales"] = mod.scales.detach().cpu().numpy().astype(np.float32)
        g_idx = getattr(mod, "g_idx", None)
        if quant_type == "awq_int4" or g_idx is None:
            in_features = int(getattr(mod, "in_features"))
            group_size = int(getattr(mod, "group_size", 128))
            g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
        dst[f"{prefix}_g_idx"] = g_idx.detach().cpu().numpy().astype(np.int32)
        dst[f"{prefix}_bits"] = np.array([[int(getattr(mod, "bits", 4))]], dtype=np.int32)
        dst[f"{prefix}_group_size"] = np.array([[int(getattr(mod, "group_size", 128))]], dtype=np.int32)
        dst[f"{prefix}_in_features"] = np.array([[int(getattr(mod, "in_features"))]], dtype=np.int32)
        dst[f"{prefix}_out_features"] = np.array([[int(getattr(mod, "out_features"))]], dtype=np.int32)
        if getattr(mod, "bias", None) is not None:
            dst[f"{prefix}_bias"] = mod.bias.detach().cpu().numpy().astype(np.float32)
        return

    # Fallback: export dense weight for non-quantized modules.
    dst[prefix] = _extract_linear_weight(mod)
    if getattr(mod, "bias", None) is not None:
        dst[f"{prefix}_bias"] = mod.bias.detach().cpu().numpy().astype(np.float32)


def _infer_quant_type(mod):
    module_name = mod.__class__.__module__.lower()
    class_name = mod.__class__.__name__.lower()
    if "awq" in module_name or "awq" in class_name:
        return "awq_int4"
    return "gptq_int4"


def _normalize_linear_layout(mod, w):
    in_f = getattr(mod, "in_features", None)
    out_f = getattr(mod, "out_features", None)
    if in_f is None or out_f is None or w.ndim != 2:
        return w

    if w.shape == (out_f, in_f):
        return w
    if w.shape == (in_f, out_f):
        return w.T

    return w


def _clear_socks_proxy_env():
    for key in [
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ]:
        val = os.environ.get(key, "")
        if isinstance(val, str) and "socks://" in val.lower():
            os.environ.pop(key, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    export_qwen_hf_quant(
        model_name_or_path=args.model,
        output_mat=args.output,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
