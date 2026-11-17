import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdc_research import telemetry


class GpuParsingTests(unittest.TestCase):
    def test_parses_reading_and_rejects_unsupported_fields(self):
        self.assertEqual(telemetry.parse_gpu_line("350, 12, 41.50, 35\n"),
                         {"vram_mib": 350.0, "gpu_util_pct": 12.0, "power_w": 41.5, "temp_c": 35.0})
        self.assertIsNone(telemetry.parse_gpu_line("350, [N/A], 41.50, 35"))
        self.assertIsNone(telemetry.parse_gpu_line("garbage"))

    def test_summary_uses_only_window_samples(self):
        samples = [(0.5, {"vram_mib": 300.0, "gpu_util_pct": 0.0, "power_w": 2.0, "temp_c": 30.0}),
                   (1.1, {"vram_mib": 5000.0, "gpu_util_pct": 80.0, "power_w": 40.0, "temp_c": 50.0}),
                   (1.9, {"vram_mib": 5200.0, "gpu_util_pct": 100.0, "power_w": 60.0, "temp_c": 54.0}),
                   (2.5, {"vram_mib": 9999.0, "gpu_util_pct": 1.0, "power_w": 1.0, "temp_c": 99.0})]
        summary = telemetry.summarize_gpu(samples, 1.0, 2.0)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["vram_max_mib"], 5200.0)
        self.assertEqual(summary["gpu_util_mean_pct"], 90.0)
        self.assertEqual(summary["power_mean_w"], 50.0)
        self.assertEqual(summary["energy_j"], 50.0)
        self.assertEqual(summary["temp_max_c"], 54.0)

    def test_short_window_falls_back_to_latest_earlier_sample(self):
        samples = [(0.5, {"vram_mib": 300.0, "gpu_util_pct": 5.0, "power_w": 10.0, "temp_c": 30.0})]
        summary = telemetry.summarize_gpu(samples, 1.0, 1.05)
        self.assertEqual(summary["samples"], 0)
        self.assertEqual(summary["vram_max_mib"], 300.0)
        self.assertTrue(summary["carried_from_earlier_sample"])

    def test_no_samples_is_an_error_not_zero(self):
        summary = telemetry.summarize_gpu([], 1.0, 2.0)
        self.assertEqual(summary, {"samples": 0, "error": "no GPU samples collected"})


class ProcessTests(unittest.TestCase):
    def test_reads_own_process(self):
        reading = telemetry.read_process(os.getpid())
        self.assertGreater(reading["cpu_seconds"], 0)
        self.assertGreater(reading["rss_mib"], 1)
        self.assertGreaterEqual(reading["peak_rss_mib"], reading["rss_mib"] - 1)
        self.assertGreaterEqual(reading["threads"], 1)

    def test_missing_process_is_reported(self):
        self.assertIn("error", telemetry.read_process(2 ** 22 + 12345))

    def test_process_delta(self):
        before = {"cpu_seconds": 1.0, "rss_mib": 100.0, "peak_rss_mib": 120.0, "threads": 9}
        after = {"cpu_seconds": 5.0, "rss_mib": 110.0, "peak_rss_mib": 130.0, "threads": 9}
        self.assertEqual(telemetry.process_delta(before, after, 2.0),
                         {"cpu_seconds": 4.0, "cpu_cores_mean": 2.0, "rss_mib": 110.0,
                          "peak_rss_mib": 130.0, "threads": 9})
        self.assertIn("error", telemetry.process_delta({"error": "gone"}, after, 2.0))


class SamplerTests(unittest.TestCase):
    def test_without_server_pid_file_sampler_is_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("pdc_research.telemetry.subprocess.Popen") as popen:
                with telemetry.sampler(Path(directory) / "server.log") as probe:
                    result = probe.since(probe.mark())
                popen.assert_not_called()
        self.assertEqual(result, {"error": "no server process recorded; telemetry skipped"})

    def test_missing_nvidia_smi_is_recorded_and_process_still_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            telemetry.pid_file(log).write_text(str(os.getpid()))
            with patch("pdc_research.telemetry.subprocess.Popen", side_effect=FileNotFoundError("nvidia-smi")):
                with telemetry.sampler(log) as probe:
                    mark = probe.mark()
                    sum(range(200000))
                    result = probe.since(mark)
        self.assertIn("nvidia-smi", result["gpu"]["error"])
        self.assertGreaterEqual(result["process"]["cpu_seconds"], 0)
        self.assertGreater(result["window_seconds"], 0)

    def test_streamed_samples_are_collected_for_one_gpu_and_a_process_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "nvidia-smi"
            fake.write_text("#!/bin/sh\n"
                            "case \"$*\" in *'-i 1'*) ;; *) echo 'missing -i 1' >&2; exit 1;; esac\n"
                            "while true; do echo '1000, 50, 200.0, 60'; sleep 0.05; done\n")
            fake.chmod(0o755)
            with patch.dict(os.environ, {"PATH": f"{directory}:{os.environ['PATH']}"}):
                with telemetry.Sampler(os.getpid(), gpu_index=1, tree=True) as probe:
                    mark = probe.mark()
                    import time
                    time.sleep(0.4)
                    result = probe.since(mark)
        self.assertGreater(result["gpu"]["samples"], 2)
        self.assertEqual(result["gpu"]["gpu_util_mean_pct"], 50.0)
        self.assertGreaterEqual(result["process"]["processes"], 1)

    def test_conditions_never_raise(self):
        conditions = telemetry.run_conditions()
        self.assertIn("recorded_utc", conditions)
        self.assertIn("cpu_governor", conditions)


if __name__ == "__main__":
    unittest.main()
