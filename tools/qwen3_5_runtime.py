"""Local checkpoint policy and project-scoped Python runtime for Qwen3.5."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_VERSION = "5.3.0"


def resolve_repo_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def native_environment(matlab_root=None):
    environment = os.environ.copy()
    if matlab_root:
        root = Path(matlab_root)
        for name in ["LD_LIBRARY_PATH", "LD_PRELOAD"]:
            separator = " " if name == "LD_PRELOAD" else os.pathsep
            entries = environment.get(name, "").replace(":", separator).split(separator)
            retained = [p for p in entries if p and not Path(p).is_relative_to(root)]
            if retained:
                environment[name] = separator.join(retained)
            else:
                environment.pop(name, None)
    environment.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return environment


def run(command, log, env=None):
    command = [str(value) for value in command]
    with Path(log).open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = Path(log).read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"Command failed: {log}\n" + "\n".join(tail))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_checkpoint(model_folder, model_name, allow_download=False):
    model = resolve_repo_path(model_folder)
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    if index.is_file():
        names += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if (model / "generation_config.json").is_file():
        names.append("generation_config.json")
    missing = [name for name in names if not (model / name).is_file()]
    if missing and not allow_download:
        raise FileNotFoundError(
            f"Local weights are missing in {model_folder}: {', '.join(missing)}. "
            "allowDownload=false; no model download was attempted.")
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=model,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE"])
        return local_checkpoint(model_folder, model_name, False)
    data = json.loads(config.read_text())
    text_config = data.get("text_config", data)
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError("Expected a local Qwen3.5 text checkpoint")
    signature = hashlib.sha256(config.read_bytes())
    signature.update(str(model).encode())
    for name in names:
        stat = (model / name).stat()
        signature.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return model, signature.hexdigest()


def python_runtime(folder, setup_packages=True, matlab_root=None):
    folder = resolve_repo_path(folder)
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = native_environment(matlab_root)
    probe = ("import importlib.metadata as m; "
             f"assert m.version('transformers') == '{TRANSFORMERS_VERSION}'; "
             "m.version('torch'); m.version('scipy'); "
             "assert int(m.version('Pillow').split('.')[0]) >= 10")
    if python.is_file():
        result = subprocess.run([str(python), "-c", probe], env=environment,
                                capture_output=True, check=False)
        if result.returncode == 0:
            return python
    if not setup_packages:
        raise RuntimeError("Qwen3.5 Python runtime is not ready; enable setupPython or install its dependencies")
    print("Preparing project Python dependencies.", flush=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(folder)],
                       env=environment, check=True)
    log = folder / "setup.log"
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
         f"transformers=={TRANSFORMERS_VERSION}", "scipy>=1.10", "Pillow>=10.0.1"], log, environment)
    result = subprocess.run([str(python), "-c", "import torch"], env=environment,
                            capture_output=True, check=False)
    if result.returncode:
        run([python, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch==2.10.0"], folder / "torch_setup.log", environment)
    run([python, "-c", probe], folder / "check.log", environment)
    return python
