#!/usr/bin/env python3
"""Prepare local Qwen3.5 weights and a project Transformers/PyTorch runtime."""

import argparse
import json
from pathlib import Path
import subprocess
import traceback

from qwen3_5_runtime import (ROOT, local_checkpoint, native_environment,
                             python_runtime, resolve_repo_path, write_json)


def numerical_summary(work):
    references = work / "references"
    cases = json.loads((work / "reference_response.json").read_text())["cases"]
    reports = [json.loads((references / case["name"] / "matlab_comparison.json").read_text())
               for case in cases]
    if not reports or not all(report["Pass"] for report in reports):
        raise RuntimeError("A MATLAB/PyTorch numerical case failed")
    summary = {"status": "PASS", "backend": "Transformers/PyTorch FP32 CPU",
               "tensor_comparisons": sum(r["TensorComparisons"] for r in reports),
               "top1_matches": sum(r["Top1Matches"] for r in reports),
               "top1_count": sum(r["Top1Count"] for r in reports),
               "max_logits_abs_error": max(t["MaxAbsError"] for r in reports
                                            for t in r["Tensors"] if t["Tensor"] == "logits"),
               "cases": [{k: v for k, v in r.items() if k != "Tensors"} for r in reports]}
    write_json(work / "summary.json", summary)
    return summary


def prepare(args):
    model, signature = local_checkpoint(args.model, args.model_name, args.allow_download)
    python = python_runtime(args.python_env, not args.no_setup_python, args.matlab_root)
    work = resolve_repo_path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    response = {"ok": True, "model_folder": str(model), "model_signature": signature,
                "python_executable": str(python), "allow_download": args.allow_download,
                "backend": "Transformers/PyTorch FP32 CPU"}
    write_json(work / "preparation.json", response)
    if args.response:
        write_json(args.response, response)
    if args.setup_only:
        print("LOCAL_QWEN3_5_READY allow_download=" + str(args.allow_download).lower(), flush=True)
        return
    request = {"model_folder": str(model), "output_dir": str(work / "references"),
               "matlab_root": args.matlab_root, "max_new_tokens": 8}
    request_file = work / "numerics_request.json"
    write_json(request_file, request)
    subprocess.run([str(python), str(ROOT / "tools/qwen3_5_transformers.py"),
                    "--action", "numerics", "--request", str(request_file),
                    "--response", str(work / "reference_response.json")],
                   cwd=ROOT, env=native_environment(args.matlab_root), check=True)
    model_arg = str(model).replace("'", "''")
    reference_arg = str(work / "references").replace("'", "''")
    root_arg = str(ROOT).replace("'", "''")
    expression = (f"addpath('{root_arg}'); maxNumCompThreads(8); "
                  f"TestQwen3_5Numerics('{model_arg}','{reference_arg}');")
    subprocess.run([args.matlab, "-nodisplay", "-nosplash", "-batch", expression], cwd=ROOT, check=True)
    summary = numerical_summary(work)
    print(f"QWEN3_5_PYTORCH_PASS cases={len(summary['cases'])} "
          f"top1={summary['top1_matches']}/{summary['top1_count']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("work/models/Qwen3.5-0.8B-FULL"))
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow missing model downloads")
    parser.add_argument("--python-env", type=Path, default=Path("work/deps/qwen3_5_python"))
    parser.add_argument("--no-setup-python", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("work/tmp/qwen3_5_transformers"))
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--response", type=Path)
    parser.add_argument("--matlab-root")
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as error:
        if args.response:
            write_json(args.response, {"ok": False, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
