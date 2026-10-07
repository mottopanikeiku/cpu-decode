"""Prepare pinned CPU artifacts and run the fixed cloud comparison in one container.

This is called by cloud_modal.py inside the running CPU container. Compilation
uses that container's CPU, not the image builder. No laptop timing is supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import statistics
import subprocess
import time


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def capture(command, cwd=None):
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout


def clone(url, commit, destination):
    run(["git", "init", str(destination)])
    run(["git", "remote", "add", "origin", url], destination)
    run(["git", "fetch", "--depth", "1", "origin", commit], destination)
    run(["git", "checkout", "--detach", commit], destination)
    if capture(["git", "rev-parse", "HEAD"], destination).strip() != commit:
        raise ValueError("Source checkout differs from fixed commit")


def environment():
    records = {"lscpu_text": capture(["lscpu"]), "lscpu_json": json.loads(capture(["lscpu", "--json"])),
               "compiler": capture(["c++", "--version"]), "cmake": capture(["cmake", "--version"]),
               "uname": capture(["uname", "-a"]), "allowed_cpus": sorted(os.sched_getaffinity(0))}
    records["collected_utc"] = capture(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).strip()
    records["exposed_cpu_topology"] = [
        dict(zip(("cpu", "core", "socket"), map(int, row.split(","))))
        for row in capture(["lscpu", "-p=CPU,CORE,SOCKET"]).splitlines() if row and not row.startswith("#")]
    for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        records[name] = path.read_text() if path.exists() else None
    flags = next((line.split(":", 1)[1].split() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("flags")), [])
    records["flags"] = flags
    return records


class Worker:
    def __init__(self, binary: Path, config: dict, log: Path):
        self.log = log.open("w")
        self.process = None
        self.stopped = False
        self.closed = False
        self.cooperative_idle = config.get("backend") == "llama"
        self.config = config
        try:
            self.process = subprocess.Popen(
                ["taskset", "-c", ",".join(map(str, config["cpu_set"])), str(binary)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, bufsize=1, start_new_session=True)
            self.send(config)
            self.ready = self.receive()
            if self.ready.get("event") != "ready":
                raise ValueError("Benchmark worker did not finish load/prefill/warmup")
            if self.cooperative_idle and self.ready["metadata"].get("idle_policy") != "ggml_pool_paused":
                raise ValueError("Baseline did not request its real GGML pool pause")
            self.stop()
        except BaseException:
            self.abort()
            raise

    def send(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout=1200):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=timeout):
                tasks = {}
                for task in Path(f"/proc/{self.process.pid}/task").glob("*"):
                    tasks[task.name] = {}
                    for name in ("status", "wchan", "stat"):
                        try:
                            tasks[task.name][name] = (task / name).read_text()
                        except OSError as error:
                            tasks[task.name][name] = {"error": str(error)}
                self.log.flush()
                detail = Path(self.log.name).read_text()[-6000:]
                raise TimeoutError(f"Benchmark worker response exceeded {timeout} seconds; "
                                   f"config={self.config}; tasks={json.dumps(tasks)}\n{detail}")
        line = self.process.stdout.readline()
        if not line:
            self.log.flush()
            detail = Path(self.log.name).read_text()[-3000:]
            raise RuntimeError(f"Benchmark worker exited: {self.process.poll()}\n{detail}")
        return json.loads(line)

    def stop(self):
        if self.cooperative_idle:
            return
        if not self.stopped:
            os.killpg(self.process.pid, signal.SIGSTOP)
            _, status = os.waitpid(self.process.pid, os.WUNTRACED)
            if not os.WIFSTOPPED(status) or os.WSTOPSIG(status) != signal.SIGSTOP:
                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    self.process.returncode = os.waitstatus_to_exitcode(status)
                raise RuntimeError("Benchmark worker exited before confirmed suspension")
            self.stopped = True

    def sample(self):
        if not self.cooperative_idle:
            os.killpg(self.process.pid, signal.SIGCONT)
            self.stopped = False
        self.send({"command": "run"})
        result = self.receive(timeout=120)
        if self.cooperative_idle and result.get("idle_pool_pause_requested") is not True:
            raise ValueError("Baseline did not request its GGML pool pause after decode")
        self.stop()
        if not isinstance(result.get("seconds"), (int, float)) or result["seconds"] <= 0:
            raise ValueError("Invalid worker timing result")
        return result

    def abort(self):
        if self.closed:
            return
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except BrokenPipeError:
                    pass
        self.log.close()
        self.closed = True

    def close(self):
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.send({"command": "exit"})
                except BrokenPipeError:
                    pass
                if self.stopped:
                    try:
                        os.killpg(self.process.pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                    self.stopped = False
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.abort()
        if self.process.returncode:
            raise RuntimeError(f"Benchmark worker failed with {self.process.returncode}")


def close_workers(workers):
    first_error = None
    for worker in workers:
        try:
            worker.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def prepare(source: Path, work: Path, output: Path, design: dict, *, assets: Path | None = None):
    python = source / ".venv/bin/python"
    cache = work / "artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    hf = assets / "hf" if assets is not None else work / "hf"
    if assets is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
    download = [str(python), "-m", "tools.download_model", "--hf-home", str(hf),
                "--output", str(output / "source-model.json")]
    if assets is not None:
        download.append("--offline")
    run(download, source)
    model = hf / "hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / design["model_revision"]
    build = source / "build-cloud"
    native_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCPU_DECODE_NATIVE=ON", "-DCMAKE_CXX_FLAGS=-march=native"]
    run(["cmake", "-S", str(source), "-B", str(build), *native_flags])
    run(["cmake", "--build", str(build), "--target", "cpu-decode", "-j4"])
    if assets is None:
        native = cache / "g64f16"
        run([str(python), "-m", "tools.quantize", "--source", str(model), "--output", str(native),
             "--engine", str(build / "cpu-decode"), "--group-size", "64", "--scale-dtype", "f16",
             "--manifest", str(output / "native-preparation.json")], source)
    else:
        native = assets / "artifacts/g64f16"
        shutil.copyfile(assets / "manifests/native-preparation.json", output / "native-preparation.json")
    if sha(native / "model.safetensors") != design["native_weights_sha256"]:
        raise ValueError("Cloud g64f16 weights differ from the existing v2 format")
    llama_root = cache / "llama.cpp" / design["llama_commit"]
    llama_source = llama_root / "source"
    if assets is None:
        clone(design["llama_repository"], design["llama_commit"], llama_source)
    else:
        staged_llama = assets / "artifacts/llama.cpp" / design["llama_commit"]
        shutil.copytree(staged_llama / "source", llama_source)
        if capture(["git", "rev-parse", "HEAD"], llama_source).strip() != design["llama_commit"]:
            raise ValueError("Cached upstream source differs from the fixed commit")
        revision = design["model_revision"]
        for name in (f"qwen-{revision}-bf16.gguf", f"qwen-{revision}-q8_0.gguf"):
            (llama_root / name).symlink_to(staged_llama / name)
        shutil.copyfile(staged_llama / f"preparation-{revision}.json",
                        llama_root / f"preparation-{revision}.json")
    llama_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_OPENMP=OFF",
                   "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_CPU_REPACK=ON", "-DLLAMA_CURL=OFF",
                   "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DCMAKE_CXX_FLAGS=-march=native"]
    run(["cmake", "-S", str(llama_source), "-B", str(llama_root / "build"), *llama_flags])
    if assets is None:
        run(["cmake", "--build", str(llama_root / "build"), "--target", "llama-bench", "llama-quantize", "-j4"])
        run([str(python), "-m", "tools.prepare_llama", "--model", str(model), "--cache", str(cache),
             "--jobs", "4", "--reuse-build", "--output", str(output / "llama-preparation.json")], source)
        preparation = json.loads((output / "llama-preparation.json").read_text())
    else:
        run(["cmake", "--build", str(llama_root / "build"), "--target", "llama", "-j4"])
        preparation = json.loads((llama_root / f"preparation-{revision}.json").read_text())
        for name in ("upstream_binaries", "converter_python", "bench_binary"):
            preparation.pop(name, None)
        preparation["build"] = {"type": "Release", "native_cpu": True, "gpu": False,
                                "jobs": 4, "reused": False, "targets": ["llama"]}
        for artifact in preparation["artifacts"].values():
            path = llama_root / Path(artifact["path"]).name
            artifact.update(sha256=sha(path), bytes=path.stat().st_size)
        save(output / "llama-preparation.json", preparation)
    expected = json.loads((source / "results/llama-preparation.json").read_text())
    comparison = {}
    for name in ("BF16", "Q8_0"):
        comparison[name] = {"expected_sha256": expected["artifacts"][name]["sha256"],
                            "actual_sha256": preparation["artifacts"][name]["sha256"],
                            "equal": expected["artifacts"][name]["sha256"] == preparation["artifacts"][name]["sha256"]}
        if not comparison[name]["equal"]:
            raise ValueError(f"Cloud {name} differs from pinned v2 artifact; inspect before timing")
    run(["cmake", "-S", str(source), "-B", str(build), *native_flags,
         "-DCPU_DECODE_LLAMA_ROOT=" + str(llama_root)])
    run(["cmake", "--build", str(build), "--target", "cloud-bench", "-j4"])
    return build / "cloud-bench", native, llama_root / f"qwen-{design['model_revision']}-q8_0.gguf", {
        "native_commit": design["native_commit"], "llama_commit": design["llama_commit"],
        "native_flags": native_flags, "llama_flags": llama_flags,
        "native_weights_sha256": sha(native / "model.safetensors"),
        "config_sha256": sha(native / "config.json"), "gguf_hash_comparison": comparison,
        "native_binary_sha256": sha(build / "cpu-decode"), "driver_sha256": sha(build / "cloud-bench"),
        "llama_binary_sha256": sha(llama_root / "build/bin/llama-bench") if assets is None else None,
        "shared_library_sha256": {path.name: sha(path) for path in sorted((llama_root / "build/bin").glob("lib*.so"))},
        "source_cpp_sha256": {str(path.relative_to(source)): sha(path) for path in sorted((source / "src").glob("*.cpp"))},
        "driver_source_sha256": sha(source / "tools/cloud_bench.cpp")}


def read_completed(path: Path, design_sha256: str, run_mode: str, require_vnni: bool):
    if not path.exists():
        return None
    from .cloud_summary import validate
    previous = json.loads(path.read_text())
    if previous["design_sha256"] != design_sha256 or previous.get("run_mode") != run_mode:
        raise ValueError("Stored cells belong to another design or runtime-pilot mode")
    validate(previous, allow_partial=True)
    if require_vnni and any(cell["native_settings"]["kernel"] != "vnni16"
                            or cell["native_settings"].get("activation_dtype") != "int16"
                            for cell in previous["cells"]):
        raise ValueError("Stored cells are not the actual vnni16/int16 path")
    if require_vnni:
        registry = {run["id"]: run for run in previous.get("container_runs", [])}
        if any(not cell.get("container_run_id") or cell["container_run_id"] not in registry
               for cell in previous["cells"]):
            raise ValueError("Stored cells are missing their measured container provenance")
    return previous


def compare(source: Path, work: Path, output: Path, design_path: Path, *,
            assets: Path | None = None, require_vnni: bool = False,
            runtime_pilot: bool = False, budget_minutes: int | None = None,
            requested_gpu: str = "none", resume: bool = False,
            container_run_id: str = ""):
    start = time.monotonic()
    design = json.loads(design_path.read_text())
    run_mode = "runtime-pilot" if runtime_pilot else "full-matrix"
    previous = read_completed(output / "raw.json", sha(design_path), run_mode, require_vnni) if resume else None
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"})
    env = environment()
    if require_vnni:
        from .cloud_host import snapshot
        env["startup_cpu"] = snapshot()
        if not env["startup_cpu"]["supports_vnni16"]:
            raise ValueError("Timing container does not expose every required vnni16 CPU flag")
    if require_vnni and not container_run_id:
        raise ValueError("VNNI measurements require a unique container run identifier")
    binary, native_model, gguf, artifacts = prepare(source, work, output, design, assets=assets)
    topology = json.loads(capture([str(source / "build-cloud/cpu-decode"), "cpus"]))
    env["native_cpu_discovery"] = topology
    if len(topology["preferred_cpu_ids"]) < max(design["threads"]):
        raise ValueError("Cloud CPU set does not cover the complete design")
    exposed = {row["cpu"]: (row["socket"], row["core"]) for row in env["exposed_cpu_topology"]}
    selected = topology["preferred_cpu_ids"][:max(design["threads"])]
    if len({exposed[cpu] for cpu in selected}) != len(selected):
        raise ValueError("Selected cloud CPUs are not distinct exposed physical cores")
    kernel = "vnni16" if require_vnni or {"avx512_vnni", "avx512bw", "f16c"}.issubset(env["flags"]) else "auto"
    resources = dict(design["resources"])
    resources.update(gpu=requested_gpu, cpu_cores=8, memory_gib=8)
    if budget_minutes is not None:
        resources["timeout_minutes"] = budget_minutes
    if not resources["timeout_minutes"]:
        raise ValueError("The comparison needs an explicit positive booked duration")
    result = {"design_sha256": sha(design_path), "design": design, "environment": env,
              "artifacts": artifacts, "resources": resources,
              "pilot": previous["pilot"] if previous else [],
              "cells": previous["cells"] if previous else [],
              "run_mode": run_mode, "offline_model_assets": assets is not None,
              "container_runs": previous.get("container_runs", []) if previous else []}
    result["container_runs"].append({"id": container_run_id, "environment": env,
                                     "artifacts": artifacts, "resources": resources})
    expected_tokens = {}
    cell_order = [design["runtime_pilot_cell"]] if runtime_pilot else design["cell_order"]
    completed = {(cell["threads"], cell["context"]) for cell in result["cells"]}
    for threads, context in cell_order:
        if (threads, context) in completed:
            continue
        workers = []
        cell_start = time.monotonic()
        try:
            cpus = topology["preferred_cpu_ids"][:threads]
            common = {"threads": threads, "context": context, "steps": design["steps"],
                      "tokens": design["seed_tokens"], "cpu_set": cpus, "poll": 50}
            print(f"CLOUD_PHASE native load/prefill/warmup t{threads} c{context}", flush=True)
            native = Worker(binary, {**common, "backend": "native", "model": str(native_model),
                                     "kernel": kernel, "flash": "auto"}, output / f"t{threads}-c{context}-native.stderr.txt")
            workers.append(native)
            if require_vnni and (native.ready["metadata"]["kernel"] != "vnni16"
                                 or native.ready["metadata"]["activation_dtype"] != "int16"):
                raise ValueError("Accepted CPU did not execute the actual vnni16/int16 path")
            baseline_choices = []
            pilot = {"threads": threads, "context": context, "native": [], "llama": {}}
            for _ in range(3):
                print(f"CLOUD_PHASE native pilot {_ + 1}/3 t{threads} c{context}", flush=True)
                pilot["native"].append(native.sample())
            for flash in ("on", "off", "auto"):
                print(f"CLOUD_PHASE llama {flash} load/prefill/warmup t{threads} c{context}", flush=True)
                baseline = Worker(binary, {**common, "backend": "llama", "model": str(gguf),
                                           "kernel": kernel, "flash": flash}, output / f"t{threads}-c{context}-llama-{flash}.stderr.txt")
                workers.append(baseline)
                samples = []
                for sample_index in range(3):
                    print(f"CLOUD_PHASE llama {flash} pilot {sample_index + 1}/3 t{threads} c{context}", flush=True)
                    samples.append(baseline.sample())
                pilot["llama"][flash] = {"metadata": baseline.ready["metadata"], "samples": samples}
                baseline_choices.append((statistics.median(sample["seconds"] for sample in samples), flash, baseline))
            _, selected, baseline = min(baseline_choices, key=lambda choice: choice[0])
            pilot["selected_flash"] = selected
            result["pilot"].append(pilot)
            for _, _, worker in baseline_choices:
                if worker is not baseline:
                    worker.close()
                    workers.remove(worker)
            cell = {"threads": threads, "context": context, "steps": design["steps"],
                    "native_settings": native.ready["metadata"], "llama_settings": baseline.ready["metadata"], "pairs": []}
            cell["native_settings"]["requested_kernel"] = kernel
            if not result["cells"]:
                median_native = statistics.median(sample["seconds"] for sample in pilot["native"])
                median_llama = statistics.median(sample["seconds"] for sample in pilot["llama"][selected]["samples"])
                setup_seconds = time.monotonic() - cell_start
                remaining_cells = design["cell_order"][1:]
                remaining_seconds = design["pairs_per_cell"] * (median_native + median_llama)
                remaining_seconds += sum((setup_seconds + design["pairs_per_cell"] * (median_native + median_llama))
                                         * max(1, threads / other_threads) for other_threads, _ in remaining_cells)
                pilot["runtime_projection"] = {
                    "estimated_remaining_minutes": remaining_seconds / 60,
                    "remaining_function_minutes": resources["timeout_minutes"] - (time.monotonic() - start) / 60,
                    "assumptions": "Use first long-context pilot and setup for all remaining cells; scale 1-thread work by 2 and assume no gain at 4 threads. This is a planning estimate, not a timing result or upper bound."}
            for block in range(design["pairs_per_cell"] // 2):
                first_native = native.sample()
                first_llama = baseline.sample()
                second_llama = baseline.sample()
                second_native = native.sample()
                for offset, a, b, order in ((0, first_native, first_llama, "AB"), (1, second_native, second_llama, "BA")):
                    for name, sample in (("native", a), ("llama", b)):
                        key = (threads, context, name)
                        if len(sample["tokens"]) != design["steps"]:
                            raise ValueError("Worker generated the wrong number of measured tokens")
                        trajectory = (sample["tokens"], sample["next_token"])
                        if key in expected_tokens and expected_tokens[key] != trajectory:
                            raise ValueError("Greedy trajectory changed between rewound repeats")
                        expected_tokens[key] = trajectory
                    cell["pairs"].append({"id": block * 2 + offset, "block": block, "order": order, "native": a, "llama": b})
            cell["elapsed_cell_seconds"] = time.monotonic() - cell_start
            cell["container_run_id"] = container_run_id
            result["cells"].append(cell)
            save(output / "raw.json", result)
            print("CLOUD_CELL_COMPLETE", flush=True)
        finally:
            close_workers(workers)
    result["cost"] = {"runner_wall_minutes": (time.monotonic() - start) / 60,
                      "requested_resources_hourly_usd": 8 * 0.047160 + 8 * 0.007992,
                      "note": "Runner wall time includes artifact preparation, builds, pilots and warmups, but excludes initial native checkout and environment installation. Function resource estimate is added by the launcher; neither estimate is an invoice."}
    save(output / "raw.json", result)


def cancel(_signum, _frame):
    raise InterruptedError("Cloud runner cancelled; release its active and stopped workers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Previously verified CPU-prepared model Volume; use models offline")
    parser.add_argument("--require-vnni", action="store_true")
    parser.add_argument("--runtime-pilot", action="store_true", help="Measure only the first complete cell; never pool it into final inference")
    parser.add_argument("--budget-minutes", type=int)
    parser.add_argument("--requested-gpu", default="none")
    parser.add_argument("--resume", action="store_true", help="Skip validated completed cells from this same design and mode")
    parser.add_argument("--container-run-id", default="")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, cancel)
    compare(args.source, args.work, args.output, args.design, assets=args.assets,
            require_vnni=args.require_vnni, runtime_pilot=args.runtime_pilot,
            budget_minutes=args.budget_minutes, requested_gpu=args.requested_gpu,
            resume=args.resume, container_run_id=args.container_run_id)


if __name__ == "__main__":
    main()
