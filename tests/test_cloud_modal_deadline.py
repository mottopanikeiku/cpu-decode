"""Model-free launcher tests; Modal is an optional CLI dependency."""
import importlib.util
import ast
import json
import io
import zipfile
from pathlib import Path
import tempfile
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

SDK_AVAILABLE = importlib.util.find_spec("modal") is not None
if SDK_AVAILABLE:
    from tools.cloud_modal import ROOT, UPLOADS, enforce_deadline, guard_destination, parse_deadline, persist_outputs, remaining, unpack_outputs


@unittest.skipUnless(SDK_AVAILABLE, "install the optional Modal CLI to test its launcher")
class DeadlineTests(unittest.TestCase):
    def cleanup(self, process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        process.stdout.close()

    def test_alarm_escalates_while_stdout_reader_is_blocked(self):
        code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(2)"
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        alarm = None
        try:
            self.assertEqual(process.stdout.readline(), "ready\n")
            alarm = threading.Timer(0.01, enforce_deadline, args=(process, 0.05))
            alarm.start()
            self.assertEqual(process.stdout.read(), "")
            self.assertEqual(process.wait(timeout=2), -signal.SIGKILL)
            alarm.join(timeout=2)
            self.assertFalse(alarm.is_alive())
        finally:
            if alarm is not None:
                alarm.cancel()
                alarm.join(timeout=2)
            self.cleanup(process)

    def test_leader_exit_does_not_leave_descendant_holding_stdout(self):
        child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('child-ready',flush=True);time.sleep(2)"
        code = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(2)"
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline(), "child-ready\n")
            enforce_deadline(process, 0.05)
            self.assertEqual(process.stdout.read(), "")
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            self.cleanup(process)

    def test_deadline_requires_an_explicit_timezone(self):
        with self.assertRaises(ValueError):
            parse_deadline("2026-10-07T15:30:00")
        self.assertEqual(parse_deadline("2026-10-07T22:30:00Z"),
                         parse_deadline("2026-10-07T15:30:00-07:00"))

    def test_cleanup_reserve_is_not_available_for_more_work(self):
        with self.assertRaises(TimeoutError):
            remaining(time.monotonic(), 0, None)


@unittest.skipUnless(SDK_AVAILABLE, "install the optional Modal CLI to test its launcher")
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_is_atomic_and_committed_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            checkpoint = Path(directory) / "checkpoint"
            source.mkdir()
            (source / "raw.json").write_text('{"cells": []}')
            (source / "native-preparation.json").write_text('{"source": "synthetic"}')
            (source / "native.stderr.txt").write_text("excluded")
            (source / "unfinished.json.tmp").write_text("excluded")
            raw = {"cells": [{"kind": "synthetic"}]}
            with patch("tools.cloud_modal.modal.Volume.from_name") as volume:
                def committed():
                    self.assertEqual(json.loads((checkpoint / "raw.json").read_text()), raw)
                    self.assertEqual({path.name for path in checkpoint.iterdir()},
                                     {"raw.json", "native-preparation.json"})
                volume.return_value.commit.side_effect = committed
                persist_outputs(source, checkpoint, raw, "cpu-decode-day-test")
                volume.return_value.commit.assert_called_once_with()

    def test_image_helpers_include_relative_import_dependencies(self):
        uploaded = set(UPLOADS)
        for filename in UPLOADS:
            if filename.endswith(".py"):
                for node in ast.walk(ast.parse((ROOT / "tools" / filename).read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                        self.assertIn(node.module + ".py", uploaded)

    def test_archive_recovery_preserves_local_experiment_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "raw.json").write_text(json.dumps({
                "design_sha256": "a" * 64, "run_mode": "runtime-pilot"}))
            self.assertEqual(guard_destination(destination, True, True, True), "a" * 64)
            with self.assertRaisesRegex(ValueError, "another comparison"):
                guard_destination(destination, True, True, False)
            with self.assertRaisesRegex(ValueError, "another comparison"):
                guard_destination(destination, False, True, True)
            with self.assertRaisesRegex(ValueError, "explicit --resume"):
                guard_destination(destination, True, False, True)

    def test_archive_payload_cannot_overwrite_a_different_saved_experiment(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("raw.json", json.dumps({"design_sha256": "b" * 64}))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "raw.json").write_text("unchanged")
            with self.assertRaisesRegex(ValueError, "another archived experiment"):
                unpack_outputs(destination, payload.getvalue(), expected_design_sha="a" * 64)
            self.assertEqual((destination / "raw.json").read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
