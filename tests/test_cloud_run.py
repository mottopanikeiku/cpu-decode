"""Real Linux process lifecycle checks; protocol fixtures are not model timings."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tools.cloud_run import Worker, close_workers


@pytest.fixture
def protocol_worker(tmp_path):
    def make(name="worker", ready="ready", exit_code=0):
        script = tmp_path / name
        script.write_text(f"#!{sys.executable}\n" + f'''
import json, sys, threading, time
for _ in range(3):
    threading.Thread(target=lambda: time.sleep(60), daemon=True).start()
json.loads(sys.stdin.readline())
print(json.dumps({{"event": {ready!r}, "metadata": {{}}}}), flush=True)
for line in sys.stdin:
    command = json.loads(line)["command"]
    if command == "exit":
        sys.exit({exit_code})
    print(json.dumps({{"seconds": 1, "tokens": [42] * 128, "next_token": 42}}), flush=True)
''')
        script.chmod(0o700)
        return script
    return make


def config():
    return {"cpu_set": [min(os.sched_getaffinity(0))]}


def assert_stopped(worker):
    tasks = list(Path(f"/proc/{worker.process.pid}/task").iterdir())
    assert len(tasks) >= 4
    for task in tasks:
        state = next(line for line in (task / "status").read_text().splitlines() if line.startswith("State:"))
        assert "T (stopped)" in state


def test_stop_acknowledges_every_thread_and_consumes_wait_event(protocol_worker, tmp_path):
    worker = Worker(protocol_worker(), config(), tmp_path / "worker.log")
    try:
        assert_stopped(worker)
        assert os.waitpid(worker.process.pid, os.WNOHANG | os.WUNTRACED) == (0, 0)
        first = worker.sample()
        assert_stopped(worker)
        assert os.waitpid(worker.process.pid, os.WNOHANG | os.WUNTRACED) == (0, 0)
        assert worker.sample() == first
        assert_stopped(worker)
    finally:
        worker.close()
    assert worker.process.returncode == 0
    assert worker.closed and worker.log.closed
    assert worker.process.stdin.closed and worker.process.stdout.closed
    worker.close()  # Idempotent after graceful cleanup.


def test_failed_readiness_kills_reaps_and_closes_process(protocol_worker, tmp_path, monkeypatch):
    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", record)
    with pytest.raises(ValueError, match="did not finish"):
        Worker(protocol_worker(ready="wrong-event"), config(), tmp_path / "bad.log")
    assert len(children) == 1
    child = children[0]
    assert child.returncode == -signal.SIGKILL
    assert child.stdin.closed and child.stdout.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(child.pid, 0)


def test_spawn_failure_closes_log(tmp_path, monkeypatch):
    logs = []
    original = Path.open

    def opened(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        logs.append(stream)
        return stream

    def fail(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failed"):
        Worker(tmp_path / "unused", config(), tmp_path / "spawn.log")
    assert len(logs) == 1 and logs[0].closed


def test_cleanup_releases_every_worker_after_nonzero_exit(protocol_worker, tmp_path):
    bad = Worker(protocol_worker("bad", exit_code=7), config(), tmp_path / "bad.log")
    good = Worker(protocol_worker("good"), config(), tmp_path / "good.log")
    with pytest.raises(RuntimeError, match="failed with 7"):
        close_workers([bad, good])
    assert bad.closed and good.closed
    assert bad.process.returncode == 7
    assert good.process.returncode == 0
    assert bad.log.closed and good.log.closed


def test_abort_reaps_a_stopped_worker(protocol_worker, tmp_path):
    worker = Worker(protocol_worker(), config(), tmp_path / "abort.log")
    assert_stopped(worker)
    worker.abort()
    assert worker.closed and worker.process.returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.killpg(worker.process.pid, 0)
    worker.abort()
