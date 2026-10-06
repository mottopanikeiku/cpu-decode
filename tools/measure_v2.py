"""Freeze settings, then collect one externally scheduled v2 benchmark window."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from tools.download_model import file_hash
from tools.portable import portable
from tools.prepare_llama import LLAMA_COMMIT

ROOT = Path(__file__).resolve().parents[1]
THREADS = [1, 2, 4, 6, 12]
CONTEXTS = [128, 1024, 4096]
ABLATION_CELLS = [(2, 128), (6, 4096)]
ABLATION_LABELS = ["per-row-scalar", "simd256", "simd512x4", "blocked", "f16-kv", "pool", "grouped", "vnni"]
DEFAULT_TOKENS = ",".join(map(str, json.loads((ROOT / "configs/prompts.json").read_text())["benchmark"]["seed_token_ids"]))


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidates(polls: list[int]) -> list[dict]:
    if not polls or len(set(polls)) != len(polls) or any(not 0 <= p <= 100 for p in polls):
        raise ValueError("poll values must be unique and between 0 and 100")
    return [{"id": f"{fa}-{affinity}-poll{poll}", "flash_attn": fa,
             "affinity": affinity, "poll": poll, "repack": True}
            for poll in polls for fa in ["on", "off", "auto"] for affinity in ["pinned", "unpinned", "defaults"]]


def cpu_order(metadata: dict, explicit: str | None, needed: int) -> list[int]:
    allowed = metadata["allowed_cpu_ids"]
    order = [int(x) for x in explicit.split(",")] if explicit else metadata["preferred_cpu_ids"]
    if len(set(order)) != len(order) or any(x not in allowed for x in order) or len(order) < needed:
        raise ValueError("CPU order must contain unique allowed CPUs and cover every requested thread")
    return order


def baseline_cpu_set(protocol: dict, thread: int, candidate: dict) -> list[int]:
    return (protocol["cpu_metadata"]["allowed_cpu_ids"] if candidate["affinity"] == "defaults"
            else protocol["cpu_order"][:thread])


def cpu_mask(cpus: list[int]) -> str:
    return hex(sum(1 << cpu for cpu in cpus))


def cache_settings(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key.startswith(("CMAKE_CXX_", "CMAKE_BUILD_TYPE", "GGML_", "CPU_DECODE_NATIVE")):
            result[key] = value
    return result


def engine_command(engine: Path, model: Path, protocol: dict, thread: int, context: int,
                   settings: dict | None = None) -> list[str]:
    config = settings or protocol["native"]
    cpus = protocol["cpu_order"][:thread]
    return [str(engine), "bench", "--model", str(model), "--tokens", protocol["tokens"],
            "--threads", str(thread), "--context", str(context), "--steps", str(protocol["steps"]),
            "--repeats", str(protocol["repeats"]), "--kernel", config["kernel"],
            "--kv", config["kv_dtype"], "--attention", config["attention"],
            "--scheduler", config["scheduler"], "--cpu-set", ",".join(map(str, cpus)),
            "--rope", config["rope"]]


def llama_command(llama: Path, gguf: Path, protocol: dict, thread: int, context: int, candidate: dict) -> list[str]:
    cpus = baseline_cpu_set(protocol, thread, candidate)
    pinned = candidate["affinity"] == "pinned"
    return ["taskset", "-c", ",".join(map(str, cpus)), str(llama), "-m", str(gguf), "-p", "0", "-n", str(protocol["steps"]),
            "-d", str(context), "-t", str(thread), "-r", str(protocol["repeats"]),
            "-ngl", "0", "-ctk", "f16", "-ctv", "f16", "-fa", candidate["flash_attn"],
            "--cpu-mask", cpu_mask(cpus) if pinned else "0x0", "--cpu-strict", "1" if pinned else "0",
            "--poll", str(candidate["poll"]), "--repack", "1", "--verbose", "-o", "json"]


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stamp(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_flags(directory: Path) -> dict:
    commands = directory / "compile_commands.json"
    if commands.exists():
        return {"compile_commands": json.loads(commands.read_text())}
    flags = {str(path.relative_to(directory)): path.read_text()
             for path in sorted(directory.rglob("flags.make"))}
    if not flags:
        raise ValueError("build needs compile_commands.json or generated flags.make records")
    return {"generated_flags": flags}


def observation(command: list[str]) -> dict:
    try:
        run = subprocess.run(command, capture_output=True, text=True)
        return {"command": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def environment(locations: dict) -> dict:
    return portable({"started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                     "python": sys.version, "cpu": observation(["lscpu"]),
                     "compiler": observation(["c++", "--version"]), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                     "allowed_cpu_ids": sorted(os.sched_getaffinity(0)),
                     "environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND", "OMP_WAIT_POLICY", "LD_LIBRARY_PATH"]}}, locations)


def execute(command: list[str], stem: Path, locations: dict, deadline: float,
            env: dict | None = None) -> dict:
    """Always retain output, including timeouts, malformed JSON and nonzero exits."""
    started = datetime.now(timezone.utc).isoformat()
    stdout, stderr, returncode, error = "", "", None, None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("window deadline exhausted")
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             timeout=remaining, env=env)
        stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        error = "window timeout; subprocess terminated"
    except (OSError, TimeoutError) as exc:
        error = str(exc)
    stdout_path, stderr_path = stem.with_suffix(".stdout.txt"), stem.with_suffix(".stderr.txt")
    stdout_path.write_text(portable(stdout, locations))
    stderr_path.write_text(portable(stderr, locations))
    data = None
    if returncode == 0:
        try:
            data = json.loads(stdout, parse_constant=reject_constant)
        except ValueError as exc:
            error = f"invalid JSON: {exc}"
    elif error is None:
        error = f"nonzero exit {returncode}"
    record = portable({"command": command, "started_utc": started, "returncode": returncode,
                       "success": returncode == 0 and error is None, "error": error,
                       "stdout_file": str(stdout_path), "stderr_file": str(stderr_path),
                       "data": data, "environment_overrides": env and {k: env[k] for k in ["OMP_PLACES", "OMP_PROC_BIND", "OMP_DYNAMIC"]}}, locations)
    save(stem.with_suffix(".json"), record)
    return record


def freeze(args: argparse.Namespace, locations: dict) -> None:
    for key in ["model", "gguf", "llama", "model_manifest"]:
        if getattr(args, key) is None:
            raise ValueError(f"--{key.replace('_', '-')} is required for freeze")
    if args.kv != "f16":
        raise ValueError("comparison protocol requires F16 KV in both engines")
    if args.steps < 64 or args.repeats < 5:
        if not args.development:
            raise ValueError("final protocol requires >=64 measured tokens and >=5 repeats per invocation")
    if args.steps < 1 or args.repeats < 1:
        raise ValueError("steps and repeats must be positive")
    if args.development and args.output.resolve() == (ROOT / "results/v2").resolve():
        raise ValueError("development protocol needs its own --output under results/v2")
    manifest = json.loads(args.model_manifest.read_text())
    preparation = json.loads(args.preparation.read_text())
    if preparation["llama_commit"] != LLAMA_COMMIT or manifest["source"] != preparation["source_model"]:
        raise ValueError("baseline and native model must share the exact pinned source manifest")
    if preparation["build"] != {**preparation["build"], "type": "Release", "native_cpu": True, "gpu": False}:
        raise ValueError("baseline must be native CPU Release")
    weights = args.model / "model.safetensors"
    if file_hash(weights) != manifest["weights"]["sha256"] or file_hash(args.model / "config.json") != manifest["config_sha256"]:
        raise ValueError("native model/config hash mismatch")
    if file_hash(args.gguf) != preparation["artifacts"]["Q8_0"]["sha256"]:
        raise ValueError("Q8_0 GGUF hash mismatch")
    native_build = cache_settings(args.engine.resolve().parent / "CMakeCache.txt")
    llama_build = cache_settings(args.llama.resolve().parent.parent / "CMakeCache.txt")
    if native_build.get("CMAKE_BUILD_TYPE") != "Release" or native_build.get("CPU_DECODE_NATIVE") != "ON":
        raise ValueError("native engine must use CPU_DECODE_NATIVE=ON and Release")
    if llama_build.get("CMAKE_BUILD_TYPE") != "Release" or llama_build.get("GGML_NATIVE") != "ON":
        raise ValueError("llama.cpp must use GGML_NATIVE=ON and Release")
    metadata = execute([str(args.engine), "cpus"], args.output / "cpu-discovery", locations, time.monotonic() + 30)
    if not metadata["success"]:
        raise ValueError("CPU discovery unsuccessful; stdout/stderr retained")
    cpus = metadata["data"]
    order = cpu_order(cpus, args.cpu_order, max(THREADS))
    model_settings = {"group_size": manifest["group_size"], "scale_dtype": manifest["scale_dtype"]}
    config = json.loads((args.model / "config.json").read_text())
    geometry = {"layers": config["num_hidden_layers"], "kv_heads": config["num_key_value_heads"],
                "head_dim": config["hidden_size"] // config["num_attention_heads"]}
    artifacts = {"engine": {"sha256": file_hash(args.engine), **stamp(args.engine)},
                 "llama": {"sha256": file_hash(args.llama), **stamp(args.llama)},
                 "bandwidth": {"sha256": file_hash(args.bandwidth), **stamp(args.bandwidth)},
                 "weights": {"sha256": manifest["weights"]["sha256"], **stamp(weights)},
                 "config": {"sha256": manifest["config_sha256"], **stamp(args.model / "config.json")},
                 "gguf": {"sha256": preparation["artifacts"]["Q8_0"]["sha256"], **stamp(args.gguf)}}
    # Shared objects influence dispatch just as much as the benchmark executable.
    libraries = {path.name: {"sha256": file_hash(path), **stamp(path)}
                 for path in sorted(args.llama.resolve().parent.glob("*.so*")) if path.is_file()}
    protocol = {"schema": "cpu-decode-v2-protocol", "development": args.development,
                "threads": THREADS, "contexts": CONTEXTS, "steps": args.steps, "repeats": args.repeats,
                "rounds": 2, "warmup_steps": 1, "tokens": args.tokens, "cpu_order": order,
                "cpu_metadata": cpus, "candidates": candidates([int(x) for x in args.polls.split(",")]),
                "native": {"kernel": args.kernel, "kv_dtype": "f16", "attention": args.attention,
                           "scheduler": args.scheduler, "rope": args.rope, **model_settings},
                "model_geometry": geometry,
                "artifacts": artifacts, "llama_libraries": libraries, "native_build": native_build,
                "llama_build": llama_build, "source_model": manifest["source"], "llama_commit": LLAMA_COMMIT,
                "native_compile_flags": build_flags(args.engine.resolve().parent),
                "llama_compile_flags": build_flags(args.llama.resolve().parent.parent),
                "compiler_versions": {"native": observation([native_build["CMAKE_CXX_COMPILER"], "--version"]),
                                      "llama": observation([llama_build["CMAKE_CXX_COMPILER"], "--version"])},
                "model_manifest": manifest, "preparation": preparation, "environment": environment(locations),
                "selection": "highest pooled median baseline rate; native uses only the same candidate's ABAB window"}
    protocol = portable(protocol, locations)
    protocol["id"] = digest(protocol)
    with (args.output / "protocol.json").open("x") as stream:
        stream.write(json.dumps(protocol, indent=2) + "\n")


def check_artifacts(args: argparse.Namespace, protocol: dict, ablation: bool = False) -> None:
    files = {"engine": args.engine, "llama": args.llama, "gguf": args.gguf}
    if not ablation:
        files.update({"weights": args.model and args.model / "model.safetensors",
                      "config": args.model and args.model / "config.json"})
    elif file_hash(args.model / "config.json") != protocol["artifacts"]["config"]["sha256"]:
        raise ValueError("ablation must use the same pinned model geometry/configuration")
    for name, path in files.items():
        if path is None or stamp(path) != {k: protocol["artifacts"][name][k] for k in ["bytes", "mtime_ns"]}:
            raise ValueError(f"frozen artifact changed or missing: {name}")
        if name in {"engine", "llama"} and file_hash(path) != protocol["artifacts"][name]["sha256"]:
            raise ValueError(f"frozen binary changed: {name}")
    for name, identity in protocol["llama_libraries"].items():
        path = args.llama.resolve().parent / name
        if stamp(path) != {k: identity[k] for k in ["bytes", "mtime_ns"]}:
            raise ValueError(f"frozen shared library changed: {name}")


class InteractiveNative:
    """One model/prefix and warmup, repeated run requests; logs survive failures."""

    def __init__(self, command: list[str], stem: Path, locations: dict, deadline: float):
        self.command, self.stem, self.locations, self.deadline = command, stem, locations, deadline
        self.lines: queue.Queue = queue.Queue()
        self.log_path = stem.with_suffix(".stderr.txt")
        self.log = self.log_path.open("w")
        self.process = None
        self.error = None
        self.ready = None
        self.reader_thread = None
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, text=True, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log, bufsize=1)
            def reader():
                for line in self.process.stdout:
                    self.lines.put(line)
                self.lines.put(None)
            self.reader_thread = threading.Thread(target=reader, daemon=True)
            self.reader_thread.start()
            line = self.line()
            stem.with_suffix(".stdout.txt").write_text(portable(line, locations))
            self.ready = json.loads(line, parse_constant=reject_constant)
            if not isinstance(self.ready, dict) or self.ready.get("ready") is not True:
                raise ValueError("interactive engine did not emit ready:true")
        except (OSError, ValueError, TimeoutError) as exc:
            self.error = str(exc)

    def line(self) -> str:
        try:
            line = self.lines.get(timeout=max(0, self.deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError("interactive engine exceeded window deadline") from exc
        if line is None:
            raise ValueError("interactive engine exited before emitting result")
        return line

    def run(self, stem: Path) -> dict:
        data, line, error = None, "", self.error
        try:
            if error:
                raise ValueError(error)
            self.process.stdin.write("run\n")
            self.process.stdin.flush()
            line = self.line()
            data = json.loads(line, parse_constant=reject_constant)
            if not isinstance(data, dict):
                raise ValueError("interactive result must be a JSON object")
        except (OSError, ValueError, TimeoutError) as exc:
            error = str(exc)
            self.error = error
        stdout_path = stem.with_suffix(".stdout.txt")
        stdout_path.write_text(portable(line, self.locations))
        result = portable({"command": self.command, "request": "run", "returncode": None,
                           "success": error is None, "error": error, "data": data,
                           "stdout_file": str(stdout_path), "stderr_file": str(self.log_path)}, self.locations)
        save(stem.with_suffix(".json"), result)
        return result

    def close(self) -> dict:
        code = None
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
                code = self.process.wait(timeout=max(0.01, self.deadline - time.monotonic()))
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                code = self.process.wait()
                self.error = self.error or "interactive engine timeout or broken pipe"
            try:
                self.process.stdin.close()
            except OSError:
                pass  # Process status/error above remains the retained failure.
            if self.reader_thread is not None:
                self.reader_thread.join(timeout=1)
            self.process.stdout.close()
        self.log.close()
        self.log_path.write_text(portable(self.log_path.read_text(), self.locations))
        record = portable({"command": self.command, "ready": self.ready, "returncode": code,
                           "success": code == 0 and self.error is None, "error": self.error,
                           "stdout_file": str(self.stem.with_suffix(".stdout.txt")),
                           "stderr_file": str(self.log_path)}, self.locations)
        save(self.stem.with_suffix(".json"), record)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "window", "bandwidth", "ablation"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--bandwidth", type=Path, default=Path("build/read-bandwidth"))
    parser.add_argument("--preparation", type=Path, default=Path("results/llama-preparation.json"))
    parser.add_argument("--quality", type=Path, default=Path("results/v2/quality.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v2"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--contexts", type=int)
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--cpu-order")
    parser.add_argument("--polls", default="50")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--kernel", choices=["auto", "scalar", "simd256", "simd512", "simd512x4", "vnni"], default="auto")
    parser.add_argument("--kv", choices=["f16", "f32"], default="f16")
    parser.add_argument("--attention", choices=["blocked", "scalar"], default="blocked")
    parser.add_argument("--scheduler", choices=["pool", "openmp"], default="pool")
    parser.add_argument("--rope", choices=["cached", "direct"], default="cached")
    parser.add_argument("--label", help="Ablation ladder label, e.g. scalar-f32 or blocked-f16")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    result_root = (ROOT / "results").resolve()
    if args.output.is_relative_to(result_root) and not args.output.is_relative_to(result_root / "v2"):
        parser.error("v2 output must not overwrite v1 results")
    args.output.mkdir(parents=True, exist_ok=True)
    output_alias = str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else "$OUTPUT"
    locations = {args.output: output_alias, Path.home(): "$HOME", sys.executable: "$PYTHON"}
    for name, alias in [("model", "$INT8"), ("gguf", "$GGUF"), ("llama", "$LLAMA_BENCH"),
                        ("engine", "$ENGINE"), ("bandwidth", "$BANDWIDTH"), ("model_manifest", "$MODEL_MANIFEST")]:
        path = getattr(args, name)
        if path:
            locations[path.resolve()] = alias
    if args.llama:
        locations[args.llama.resolve().parent.parent.parent] = "$LLAMA_ROOT"
    if args.stage == "freeze":
        freeze(args, locations)
        return
    protocol = json.loads((args.output / "protocol.json").read_text())
    if digest({k: v for k, v in protocol.items() if k != "id"}) != protocol["id"]:
        raise ValueError("protocol was changed after freezing")
    if args.threads not in protocol["threads"]:
        parser.error("supply one --threads from the frozen matrix")
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        parser.error("timing commands must run at nice 19")
    thread, context = args.threads, args.contexts
    cpus = protocol["cpu_order"][:thread]
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("frozen CPU set is no longer available")
    deadline = time.monotonic() + 1740  # Leave a minute below the external 30-minute window limit.
    if args.stage == "bandwidth":
        identity = protocol["artifacts"]["bandwidth"]
        if stamp(args.bandwidth) != {k: identity[k] for k in ["bytes", "mtime_ns"]} or file_hash(args.bandwidth) != identity["sha256"]:
            raise ValueError("frozen bandwidth executable changed")
        path = args.output / f"bandwidth-t{thread}.json"
        if path.exists():
            raise FileExistsError(path)
        env = {**os.environ, "OMP_PLACES": ",".join(f"{{{x}}}" for x in cpus), "OMP_PROC_BIND": "true", "OMP_DYNAMIC": "false"}
        record = {"schema": "cpu-decode-v2-bandwidth", "protocol_id": protocol["id"], "threads": thread,
                  "cpu_set": cpus, "environment": environment(locations), "invocations": []}
        save(path, record)
        for kernel in ["simd256", "simd512"]:
            command = ["taskset", "-c", ",".join(map(str, cpus)), str(args.bandwidth),
                       "--threads", str(thread), "--kernel", kernel, "--repeats", str(protocol["repeats"])]
            record["invocations"].append(execute(command, path.with_name(f"bandwidth-t{thread}-{kernel}"), locations, deadline, env))
            save(path, record)
        if not all(r["success"] for r in record["invocations"]):
            raise RuntimeError("bandwidth window has unsuccessful invocations; raw records retained")
        return
    if context not in protocol["contexts"]:
        parser.error("supply one --contexts from the frozen matrix")
    check_artifacts(args, protocol, args.stage == "ablation")
    bundled = args.stage == "window"
    if bundled:
        matches = [c for c in protocol["candidates"] if args.candidate == "all" or c["id"] == args.candidate]
        if not matches:
            parser.error("--candidate must be all or name one frozen candidate")
        settings = protocol["native"]
        name = f"window-t{thread}-c{context}-{args.candidate}"
    else:
        if (thread, context) not in ABLATION_CELLS or not args.label or not args.label.replace("-", "").isalnum():
            parser.error("ablation needs a safe --label and cell 2/128 or 6/4096")
        matches = [None]
        settings = {"kernel": args.kernel, "kv_dtype": args.kv, "attention": args.attention,
                    "scheduler": args.scheduler, "rope": args.rope}
        name = f"ablation-t{thread}-c{context}-{args.label}"
    path = args.output / f"{name}.json"
    if path.exists():
        raise FileExistsError(path)
    record = {"schema": "cpu-decode-v2-window" if bundled else "cpu-decode-v2-ablation",
              "protocol_id": protocol["id"], "threads": thread, "context": context, "cpu_set": cpus,
              "candidates": matches if bundled else [], "native_settings": settings, "label": args.label,
              "environment": environment(locations), "invocations": []}
    if not bundled:
        record["weights_sha256"] = file_hash(args.model / "model.safetensors")
    save(path, portable(record, locations))
    command = engine_command(args.engine, args.model, protocol, thread, context, settings)
    native = InteractiveNative(command + ["--interactive", "1"], args.output / f"{name}-native-process", locations, deadline)
    try:
        for candidate in matches:
            candidate_id = candidate["id"] if candidate else args.label
            for round_id in range(protocol["rounds"]):
                for engine in (["native", "llama"] if bundled else ["native"]):
                    stem = args.output / f"{name}-{candidate_id}-r{round_id}-{engine}"
                    invocation = (native.run(stem) if engine == "native" else
                                  execute(llama_command(args.llama, args.gguf, protocol, thread, context, candidate),
                                          stem, locations, deadline))
                    record["invocations"].append({"engine": engine, "round": round_id,
                                                  "candidate_id": candidate_id, **invocation})
                    save(path, portable(record, locations))
    finally:
        record["native_process"] = native.close()
        # A nonzero process exit invalidates even previously emitted native samples.
        if not record["native_process"]["success"]:
            for invocation in record["invocations"]:
                if invocation["engine"] == "native":
                    invocation["success"] = False
                    invocation["error"] = "shared native process did not exit successfully"
        save(path, portable(record, locations))
    if not all(r["success"] for r in record["invocations"]) or not record["native_process"]["success"]:
        raise RuntimeError("window has unsuccessful invocations; raw records retained")


if __name__ == "__main__":
    main()
