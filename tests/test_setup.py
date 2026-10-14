"""Build setup checks must not overwrite modified upstream sources."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("setup_llama", "scripts/setup_llama.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Local CUDA wheel layout targets Linux")
    def test_local_cuda_preparation_is_idempotent_and_preserves_original_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vendor/cuda/nvidia/cu13"
            (root / "include/crt").mkdir(parents=True)
            (root / "lib").mkdir()
            (root / "lib/libcudart.so.13").write_bytes(b"fake library")
            header = root / "include/crt/math_functions.h"
            original = "extern double rsqrt(double x);\nextern float rsqrtf(float x);\n"
            header.write_text(original)
            with patch("os.getcwd", return_value=directory):
                setup.prepare_cuda_toolkit(root, True)
                patched = header.read_text()
                setup.prepare_cuda_toolkit(root, True)
            self.assertEqual(header.read_text(), patched)
            self.assertEqual(patched.count("__THROW"), 2)
            self.assertEqual(header.with_suffix(".h.pdc-original").read_text(), original)
            self.assertTrue((root / "lib64").is_dir())
            self.assertEqual((root / "lib/libcudart.so").resolve(), root / "lib/libcudart.so.13")

    def test_dirty_checkout_is_rejected_before_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
                patch.object(setup.subprocess, "check_output", return_value=" M source.cpp\n"),
            ):
                with self.assertRaisesRegex(RuntimeError, "local changes"):
                    setup.checkout_source(root, {"llama_cpp_commit": "abc"})
                self.assertEqual(run.call_count, 1)  # Only check HEAD; don't mutate checkout.


if __name__ == "__main__":
    unittest.main()
