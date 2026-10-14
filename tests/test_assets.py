"""Asset setup must verify bytes and avoid downloading already complete files."""

import importlib.util
import io
import tarfile
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from pdc_research.common import sha256

spec = importlib.util.spec_from_file_location("download_assets", "scripts/download_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_archive_prefix_and_reused_extraction_are_verified(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            archive = Path("data/raw/flores200_dataset.tar.gz")
            archive.parent.mkdir(parents=True)
            with tarfile.open(archive, "w:gz") as stream:
                for split in ("dev", "devtest"):
                    for language in ("eng_Latn", "rus_Cyrl", "kaz_Cyrl", "zho_Hans"):
                        info = tarfile.TarInfo(f"./flores200_dataset/{split}/{language}.{split}")
                        info.size = 4
                        stream.addfile(info, io.BytesIO(b"one\n"))
            config = {"dataset": {
                "url": "https://example.invalid/data", "filename": archive.name,
                "size_bytes": archive.stat().st_size, "sha256": sha256(archive),
            }}
            assets.prepare_dataset(config)
            extracted = Path("data/raw/flores200_dataset/dev/eng_Latn.dev")
            self.assertEqual(extracted.read_bytes(), b"one\n")
            extracted.write_bytes(b"modified\n")
            with self.assertRaisesRegex(ValueError, "Extraction does not match"):
                assets.prepare_dataset(config)

    def test_complete_partial_file_is_verified_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(b"GGUF-example")
            asset = {"size_bytes": partial.stat().st_size, "sha256": sha256(partial)}
            with patch.object(assets.subprocess, "run") as run:
                assets.download("https://example.invalid/model", target, asset)
                run.assert_not_called()
            self.assertEqual(target.read_bytes(), b"GGUF-example")
            self.assertFalse(partial.exists())

    def test_corrupt_complete_download_is_never_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                assets.download("https://example.invalid/model", target,
                                {"size_bytes": 3, "sha256": "0" * 64})
            self.assertFalse(target.exists())
            self.assertTrue(partial.exists())

    def test_verified_existing_download_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset.tar.gz"
            target.write_bytes(b"archive")
            asset = {"size_bytes": target.stat().st_size, "sha256": sha256(target)}
            with patch.object(assets.subprocess, "run") as run:
                self.assertEqual(assets.download("https://example.invalid/data", target, asset), target)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
