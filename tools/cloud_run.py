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
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
        self.process = subprocess.Popen(
            ["taskset", "-c", ",".join(map(str, config["cpu_set"])), str(binary)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, start_new_session=True)
        self.stopped = False
        self.send(config)
        self.ready = self.receive()
        if self.ready.get("event") != "ready":
            raise ValueError("Benchmark worker did not finish load/prefill/warmup")
        self.stop()

    def send(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=1200):
                raise TimeoutError("Benchmark worker response exceeded 20 minutes")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Benchmark worker exited: {self.process.poll()}")
        return json.loads(line)

    def stop(self):
        if not self.stopped:
            os.killpg(self.process.pid, signal.SIGSTOP)
            self.stopped = True

    def sample(self):
        os.killpg(self.process.pid, signal.SIGCONT)
        self.stopped = False
        self.send({"command": "run"})
        result = self.receive()
        self.stop()
        if not isinstance(result.get("seconds"), (int, float)) or result["seconds"] <= 0:
            raise ValueError("Invalid worker timing result")
        return result

    def close(self):
        if self.process.poll() is None:
            self.send({"command": "exit"})
            if self.stopped:
                os.killpg(self.process.pid, signal.SIGCONT)
                self.stopped = False
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()
        if self.process.returncode:
            raise RuntimeError(f"Benchmark worker failed with {self.process.returncode}")


def prepare(source: Path, work: Path, output: Path, design: dict):
    python = source / ".venv/bin/python"
    cache = work / "artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    hf = work / "hf"
    run([str(python), "-m", "tools.download_model", "--hf-home", str(hf),
         "--output", str(output / "source-model.json")], source)
    model = hf / "hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / design["model_revision"]
    build = source / "build-cloud"
    native_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCPU_DECODE_NATIVE=ON", "-DCMAKE_CXX_FLAGS=-march=native"]
    run(["cmake", "-S", str(source), "-B", str(build), *native_flags])
    run(["cmake", "--build", str(build), "--target", "cpu-decode", "-j4"])
    native = cache / "g64f16"
    run([str(python), "-m", "tools.quantize", "--source", str(model), "--output", str(native),
         "--engine", str(build / "cpu-decode"), "--group-size", "64", "--scale-dtype", "f16",
         "--manifest", str(output / "native-preparation.json")], source)
    if sha(native / "model.safetensors") != design["native_weights_sha256"]:
        raise ValueError("Cloud g64f16 weights differ from the existing v2 format")
    llama_root = cache / "llama.cpp" / design["llama_commit"]
    llama_source = llama_root / "source"
    clone(design["llama_repository"], design["llama_commit"], llama_source)
    llama_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_OPENMP=OFF",
                   "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_CPU_REPACK=ON", "-DLLAMA_CURL=OFF",
                   "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DCMAKE_CXX_FLAGS=-march=native"]
    run(["cmake", "-S", str(llama_source), "-B", str(llama_root / "build"), *llama_flags])
    run(["cmake", "--build", str(llama_root / "build"), "--target", "llama-bench", "llama-quantize", "-j4"])
    run([str(python), "-m", "tools.prepare_llama", "--model", str(model), "--cache", str(cache),
         "--jobs", "4", "--reuse-build", "--output", str(output / "llama-preparation.json")], source)
    preparation = json.loads((output / "llama-preparation.json").read_text())
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
        "llama_binary_sha256": sha(llama_root / "build/bin/llama-bench"),
        "shared_library_sha256": {path.name: sha(path) for path in sorted((llama_root / "build/bin").glob("lib*.so"))},
        "source_cpp_sha256": {str(path.relative_to(source)): sha(path) for path in sorted((source / "src").glob("*.cpp"))},
        "driver_source_sha256": sha(source / "tools/cloud_bench.cpp")}


def compare(source: Path, work: Path, output: Path, design_path: Path):
    start = time.monotonic()
    design = json.loads(design_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"})
    env = environment()
    binary, native_model, gguf, artifacts = prepare(source, work, output, design)
    topology = json.loads(capture([str(source / "build-cloud/cpu-decode"), "cpus"]))
    env["native_cpu_discovery"] = topology
    if len(topology["preferred_cpu_ids"]) < max(design["threads"]):
        raise ValueError("Cloud CPU set does not cover the complete design")
    kernel = "vnni16" if {"avx512_vnni", "avx512bw", "f16c"}.issubset(env["flags"]) else "auto"
    result = {"design_sha256": sha(design_path), "design": design, "environment": env,
              "artifacts": artifacts, "resources": design["resources"], "pilot": [], "cells": []}
    expected_tokens = {}
    for threads, context in design["cell_order"]:
        workers = []
        cell_start = time.monotonic()
        try:
            cpus = topology["preferred_cpu_ids"][:threads]
            common = {"threads": threads, "context": context, "steps": design["steps"],
                      "tokens": design["seed_tokens"], "cpu_set": cpus, "poll": 50}
            native = Worker(binary, {**common, "backend": "native", "model": str(native_model),
                                     "kernel": kernel, "flash": "auto"}, output / f"t{threads}-c{context}-native.stderr.txt")
            workers.append(native)
            baseline_choices = []
            pilot = {"threads": threads, "context": context, "native": [], "llama": {}}
            for _ in range(3):
                pilot["native"].append(native.sample())
            for flash in ("on", "off", "auto"):
                baseline = Worker(binary, {**common, "backend": "llama", "model": str(gguf),
                                           "kernel": kernel, "flash": flash}, output / f"t{threads}-c{context}-llama-{flash}.stderr.txt")
                workers.append(baseline)
                samples = [baseline.sample() for _ in range(3)]
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
                    "remaining_function_minutes": design["resources"]["timeout_minutes"] - (time.monotonic() - start) / 60,
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
            result["cells"].append(cell)
            save(output / "raw.json", result)
            print(f"Completed cloud cell threads={threads} context={context}", flush=True)
        finally:
            for worker in workers:
                worker.close()
    result["cost"] = {"runner_wall_minutes": (time.monotonic() - start) / 60,
                      "requested_resources_hourly_usd": 8 * 0.047160 + 8 * 0.007992,
                      "note": "Runner wall time includes artifact preparation, builds, pilots and warmups, but excludes initial native checkout and environment installation. Function resource estimate is added by the launcher; neither estimate is an invoice."}
    save(output / "raw.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    compare(args.source, args.work, args.output, args.design)


if __name__ == "__main__":
    main()
