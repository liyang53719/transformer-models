#!/usr/bin/env python3
import argparse
import json
import os
import traceback


def _clear_socks_proxy_env():
    bad_keys = [
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ]
    for key in bad_keys:
        val = os.environ.get(key, "")
        if isinstance(val, str) and "socks://" in val.lower():
            os.environ.pop(key, None)


def _load_model_and_tokenizer(req):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    endpoint = req.get("hf_endpoint", "")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    _clear_socks_proxy_env()

    model_name = req["model_name"]
    trust_remote_code = bool(req.get("trust_remote_code", True))
    local_files_only = bool(req.get("local_files_only", True))
    auto_retry_online = bool(req.get("auto_retry_online", True))
    use_gpu = bool(req.get("use_gpu", True))
    backend_type = str(req.get("backend_type", "")).lower()

    has_cuda = bool(torch.cuda.is_available()) if use_gpu else False
    if backend_type == "hf_awq" and not has_cuda:
        raise RuntimeError("AWQ branch requires CUDA GPU runtime.")

    tok_args = {"trust_remote_code": trust_remote_code}
    model_args = {"trust_remote_code": trust_remote_code}
    if local_files_only:
        tok_args["local_files_only"] = True
        model_args["local_files_only"] = True

    model_args["device_map"] = "auto" if has_cuda else "cpu"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, **tok_args)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        _configure_reference_mode(model, backend_type, req)
        return tok, model
    except Exception:
        if not (local_files_only and auto_retry_online):
            raise

    tok_args.pop("local_files_only", None)
    model_args.pop("local_files_only", None)
    tok = AutoTokenizer.from_pretrained(model_name, **tok_args)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
    _configure_reference_mode(model, backend_type, req)
    return tok, model


def _configure_reference_mode(model, backend_type: str, req):
    if backend_type != "hf_gptq":
        return

    force_unfused = bool(req.get("force_unfused_gptq", True))
    if not force_unfused:
        return

    for mod in model.modules():
        if hasattr(mod, "qweight") and hasattr(mod, "dequantize_weight") and hasattr(mod, "linear_mode"):
            mod.linear_mode = "matlab_ref"


def run(req):
    tok, model = _load_model_and_tokenizer(req)
    prompt = req["prompt"]

    inputs = tok(prompt, return_tensors="pt")
    try:
        inputs = inputs.to(model.device)
    except Exception:
        pass

    do_sample = bool(req.get("top_k", 1) > 1 and req.get("temperature", 1.0) > 0)
    gen_kwargs = {
        "max_new_tokens": int(req.get("max_new_tokens", 50)),
        "do_sample": do_sample,
        "attention_mask": inputs["attention_mask"],
    }
    if do_sample:
        gen_kwargs["temperature"] = float(req.get("temperature", 1.0))
        gen_kwargs["top_k"] = int(req.get("top_k", 1))

    outputs = model.generate(inputs["input_ids"], **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = outputs[0][prompt_len:]
    summary = tok.decode(gen_only, skip_special_tokens=True).strip()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    try:
        with open(args.request, "r", encoding="utf-8") as f:
            req = json.load(f)
        summary = run(req)
        resp = {"ok": True, "summary": summary}
        code = 0
    except Exception as exc:
        resp = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        code = 1

    with open(args.response, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False)

    raise SystemExit(code)


if __name__ == "__main__":
    main()
