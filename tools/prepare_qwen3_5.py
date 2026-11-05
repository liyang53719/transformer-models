#!/usr/bin/env python3
"""Prepare FP32 llama.cpp references and run the independent MATLAB decoder.

Like prepare_tinyllama.py, this exports intermediate tensors, logits, and
greedy token IDs. The numerical reference here is llama.cpp, as requested.
No Python implementation is used as a substitute for MATLAB model execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log, cwd=ROOT, env=None):
    command = [str(arg) for arg in command]
    print("RUN", json.dumps(command, ensure_ascii=False), flush=True)
    with Path(log).open("w") as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = Path(log).read_text(errors="replace").splitlines()[-35:]
        raise RuntimeError(f"Command failed ({result.returncode}): {log}\n" + "\n".join(tail))


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resolve_repo_path(path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def prepare(args):
    model = resolve_repo_path(args.model)
    work = resolve_repo_path(args.work_dir)
    llama_repo = resolve_repo_path(args.llama_cpp)
    work.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(llama_repo), "rev-parse", args.llama_revision], text=True).strip()
    source = work / "llama.cpp"
    stamp = work / "llama_revision.txt"
    if stamp.exists() and stamp.read_text().strip() != revision:
        raise RuntimeError("Work directory belongs to another llama.cpp revision; use a new --work-dir")
    if not stamp.exists():
        source.mkdir(exist_ok=True)
        archive = subprocess.Popen(["git", "-C", str(llama_repo), "archive", revision], stdout=subprocess.PIPE)
        extracted = subprocess.run(["tar", "-x", "-C", str(source)], stdin=archive.stdout, check=False)
        archive.stdout.close()
        if archive.wait() or extracted.returncode:
            raise RuntimeError("Cannot extract pinned llama.cpp sources")
        stamp.write_text(revision + "\n")
    build = work / "llama-build"
    run(["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_CUDA=OFF", "-DGGML_BLAS=OFF", "-DGGML_NATIVE=ON",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
         "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF"], work / "cmake.log")
    run(["cmake", "--build", build, "--target", "llama", "-j", str(args.jobs)], work / "build.log")
    compiler = None
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
            compiler = line.split("=", 1)[1]
    if not compiler:
        raise RuntimeError("Cannot find the compiler used by CMake")
    executable = work / "qwen3_5_llama_reference"
    run([compiler, "-std=c++17", "-O2", ROOT / "tools/qwen3_5_llama_reference.cpp",
         "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
         "-L" + str(build / "bin"), "-Wl,-rpath,$ORIGIN/llama-build/bin",
         "-lllama", "-lggml", "-lggml-base", "-o", executable], work / "reference_build.log")

    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    metadata_files = [name for name in ["tokenizer_config.json", "generation_config.json"]
                      if (model / name).is_file()]
    inputs = {name: sha256(model / name) for name in
              ["config.json", "tokenizer.json", *metadata_files, *shards]}
    conversion_key = {"inputs": inputs, "llama_revision": revision, "outtype": "f32", "nextn": False}
    key_path = work / "conversion.json"
    gguf_path = work / "Qwen3.5-0.8B-F32.gguf"
    if not (gguf_path.exists() and key_path.exists() and json.loads(key_path.read_text()) == conversion_key):
        run([sys.executable, source / "convert_hf_to_gguf.py", model,
             "--outfile", gguf_path, "--outtype", "f32", "--no-nextn"], work / "convert.log")
        write_json(key_path, conversion_key)
    sys.path.insert(0, str(source / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    if not reader.tensors or any(int(t.tensor_type) != 0 for t in reader.tensors):
        raise RuntimeError("Reference GGUF must contain only F32 tensors")
    tensor_count = len(reader.tensors)
    del reader

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    chinese = ("<|im_start|>user\n请用一句话解释光合作用。<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    english = ("Summarize this: The process of photosynthesis is how plants convert light energy "
               "into chemical energy. Chlorophyll absorbs sunlight and uses it to convert "
               "carbon dioxide and water into glucose. Summary:")
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 24
    cases = [
        ("single_token", "Hello", 4, 0),
        ("english", english, 8, 0),
        ("chinese", chinese, 8, 0),
        ("chinese_chunked", chinese, 8, 7),
        ("long_prefill", long_prompt, 4, 0),
    ]
    references = work / "references"
    references.mkdir(exist_ok=True)
    case_metadata = []
    for name, prompt, steps, chunk in cases:
        case_dir = references / name
        case_dir.mkdir(exist_ok=True)
        prompt_path = case_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        old_report = case_dir / "matlab_comparison.json"
        old_report.unlink(missing_ok=True)
        run([executable, gguf_path, case_dir, prompt_path, steps, chunk], case_dir / "reference.log")
        ids = [int(value) for value in (case_dir / "tokens.txt").read_text().split()]
        if ids != tokenizer.encode(prompt, add_special_tokens=False).ids:
            raise RuntimeError(f"Python/llama.cpp prompt tokenization differs in {name}")
        generated = [int(value) for value in (case_dir / "generated.txt").read_text().split()]
        item = {"name": name, "prompt_tokens": len(ids), "decode_tokens": len(generated),
                "prefill_chunk": chunk, "input_ids": ids, "generated_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=False)}
        write_json(case_dir / "reference_summary.json", item)
        case_metadata.append(item)
        print(f"REFERENCE {name}: {len(ids)} prompt tokens, {len(generated)} generated tokens", flush=True)

    provenance = {
        "status": "REFERENCE_READY", "model_folder": str(model),
        "source_input_sha256": inputs, "llama_cpp_repository": str(llama_repo),
        "llama_cpp_revision": revision, "reference_gguf": str(gguf_path),
        "reference_gguf_sha256": sha256(gguf_path), "gguf_f32_tensor_count": tensor_count,
        "reference_executable_sha256": sha256(executable),
        "libllama_sha256": sha256(build / "bin/libllama.so"),
        "reference_source_sha256": sha256(ROOT / "tools/qwen3_5_llama_reference.cpp"),
        "precision": "F32 weights, F32 KV cache, F32 recurrent state, CPU, flash attention disabled",
        "scope": "all 24 text-decoder layers; no vision encoder or MTP",
        "matlab_source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted([*(ROOT / "+qwen3_5").rglob("*.m"),
                                ROOT / "TestQwen3_5Numerics.m",
                                ROOT / "generateSummary_Qwen3_5.m",
                                ROOT / "test/qwen3_5/qwen3_5OperatorsTest.m"])
        },
        "cases": case_metadata,
    }
    write_json(work / "provenance.json", provenance)
    if args.prepare_only:
        return
    expression = (f"addpath({matlab_string(ROOT)}); maxNumCompThreads(8); "
                  f"TestQwen3_5Numerics({matlab_string(model)},{matlab_string(references)});")
    run([args.matlab, "-nodisplay", "-nosplash", "-batch", expression], work / "matlab.log")
    summaries = []
    for item in case_metadata:
        report = json.loads((references / item["name"] / "matlab_comparison.json").read_text())
        if not report["Pass"]:
            raise RuntimeError(f"Numerical comparison failed: {item['name']}")
        expected = report["ReferenceGeneratedTokens"]
        observed = report["MatlabGeneratedTokens"]
        if expected != observed:
            raise RuntimeError("MATLAB greedy output differs")
        report["ReferenceGeneratedText"] = tokenizer.decode(expected, skip_special_tokens=False)
        report["MatlabGeneratedText"] = tokenizer.decode(observed, skip_special_tokens=False)
        write_json(references / item["name"] / "matlab_comparison.json", report)
        summaries.append({k: v for k, v in report.items() if k != "Tensors"})
    summary = {"status": "PASS", "cases": summaries,
               "tensor_comparisons": sum(r["TensorComparisons"] for r in summaries),
               "top1_matches": sum(r["Top1Matches"] for r in summaries),
               "top1_count": sum(r["Top1Count"] for r in summaries)}
    write_json(work / "summary.json", summary)
    provenance["status"] = "MATLAB_COMPARISON_PASSED"
    write_json(work / "provenance.json", provenance)
    print("QWEN3_5_ALL_PASS", json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("work/models/Qwen3.5-0.8B-FULL"),
                        help="Checkpoint folder; relative paths resolve from the repository root")
    parser.add_argument("--llama-cpp", type=Path, required=True,
                        help="llama.cpp Git checkout; relative paths resolve from the repository root")
    parser.add_argument("--llama-revision", default="HEAD")
    parser.add_argument("--work-dir", type=Path, default=Path("work/tmp/qwen3_5"),
                        help="Output folder; relative paths resolve from the repository root")
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    prepare(args)
    print(f"Elapsed: {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
