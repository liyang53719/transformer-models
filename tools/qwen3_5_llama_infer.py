#!/usr/bin/env python3
"""Run an independent greedy FP32 llama.cpp summary for the MATLAB demo."""

import argparse
import json
import os
from pathlib import Path
import traceback

from prepare_qwen3_5 import ROOT, resolve_repo_path, run


def native_environment(matlab_root):
    environment = os.environ.copy()
    if not matlab_root:
        return environment
    root = Path(matlab_root)
    for name in ["LD_LIBRARY_PATH", "LD_PRELOAD"]:
        separator = " " if name == "LD_PRELOAD" else os.pathsep
        entries = environment.get(name, "").replace(":", separator).split(separator)
        retained = [entry for entry in entries if entry and not Path(entry).is_relative_to(root)]
        if retained:
            environment[name] = separator.join(retained)
        else:
            environment.pop(name, None)
    return environment


def infer(request):
    model = resolve_repo_path(Path(request["model_folder"]))
    work = resolve_repo_path(Path(request["reference_work"]))
    output = resolve_repo_path(Path(request["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    source = work / "llama.cpp"
    build = work / "llama-build"
    gguf = work / "Qwen3.5-0.8B-F32.gguf"
    if not gguf.is_file() or not (build / "bin/libllama.so").is_file():
        raise FileNotFoundError(
            "Prepare the FP32 reference first with tools/prepare_qwen3_5.py --prepare-only")
    compiler = None
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
            compiler = line.split("=", 1)[1]
    if not compiler:
        raise RuntimeError("Cannot find the reference C++ compiler")
    executable = work / "qwen3_5_llama_summary"
    environment = native_environment(request.get("matlab_root"))
    run([compiler, "-std=c++17", "-O2", ROOT / "tools/qwen3_5_llama_reference.cpp",
         "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
         "-L" + str(build / "bin"), "-Wl,-rpath,$ORIGIN/llama-build/bin",
         "-lllama", "-lggml", "-lggml-base", "-o", executable], output / "build.log", env=environment)
    prompt = str(request["prompt"])
    max_tokens = int(request.get("max_new_tokens", 30))
    if max_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_file = output / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    run([executable, gguf, output, prompt_file, max_tokens, 0, "--summary"],
        output / "reference.log", env=environment)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    prompt_ids = [int(x) for x in (output / "tokens.txt").read_text().split()]
    if prompt_ids != tokenizer.encode(prompt, add_special_tokens=False).ids:
        raise RuntimeError("Python and llama.cpp prompt token IDs differ")
    generated = [int(x) for x in (output / "generated.txt").read_text().split()]
    metrics = json.loads((output / "generation_metrics.json").read_text())
    return {"ok": True, "backend": "llama.cpp FP32 CPU", "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "summary": tokenizer.decode(generated, skip_special_tokens=False), **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    code = 0
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = infer(request)
    except Exception as error:
        response = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
        code = 1
    args.response.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
