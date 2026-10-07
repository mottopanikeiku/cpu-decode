"""Model-free launcher tests; Modal is an optional CLI dependency."""
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
import unittest

SDK_AVAILABLE = importlib.util.find_spec("modal") is not None
if SDK_AVAILABLE:
    from tools.cloud_modal import enforce_deadline, parse_deadline, remaining


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


if __name__ == "__main__":
    unittest.main()
