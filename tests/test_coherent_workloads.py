"""Curated passage preparation must enforce actual FLORES article boundaries."""

import tempfile
import unittest
from pathlib import Path

from pdc_research.common import load_rows, read_json, write_json
from pdc_research.workloads import prepare


class CoherentWorkloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "dev").mkdir()
        for language in ("en", "kk"):
            (self.root / "dev" / f"{language}.dev").write_text(
                "\n".join(f"{language} sentence {i}" for i in range(1, 5)) + "\n"
            )
        (self.root / "metadata_dev.tsv").write_text("URL\narticle-a\narticle-a\narticle-b\narticle-b\n")
        self.selection = self.root / "selection.json"
        self.experiment = {"languages": ["en", "kk"], "split": "dev", "groups": 2,
                           "sentences_per_group": 2, "seed": 42, "selection_file": str(self.selection)}

    def test_preserves_source_text_and_records_metadata_provenance(self):
        write_json(self.selection, {"split": "dev", "groups": [
            {"sentence_ids": [1, 2], "source_url": "article-a"},
            {"sentence_ids": [3, 4], "source_url": "article-b"},
        ]})
        output = self.root / "work.jsonl"
        prepare(self.experiment, self.root, output)
        rows = load_rows(output)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["text"], "en sentence 1\nen sentence 2")
        self.assertEqual(rows[0]["sentence_ids"], rows[1]["sentence_ids"])
        self.assertEqual(rows[0]["source_url"], "article-a")
        metadata = read_json(str(output) + ".meta.json")
        self.assertIn("metadata_sha256", metadata)
        self.assertIn("selection_sha256", metadata)

    def test_rejects_mixed_article_passage_before_writing_output(self):
        write_json(self.selection, {"split": "dev", "groups": [
            {"sentence_ids": [2, 3], "source_url": "article-a"},
        ]})
        self.experiment["groups"] = 1
        output = self.root / "invalid.jsonl"
        with self.assertRaisesRegex(ValueError, "article boundary"):
            prepare(self.experiment, self.root, output)
        self.assertFalse(output.exists())

    def test_rejects_overlapping_passages(self):
        write_json(self.selection, {"split": "dev", "groups": [
            {"sentence_ids": [1, 2], "source_url": "article-a"},
            {"sentence_ids": [1, 2], "source_url": "article-a"},
        ]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(self.experiment, self.root, self.root / "invalid.jsonl")

    def test_rejects_incorrect_source_url(self):
        self.experiment["groups"] = 1
        write_json(self.selection, {"split": "dev", "groups": [
            {"sentence_ids": [1, 2], "source_url": "article-b"},
        ]})
        with self.assertRaisesRegex(ValueError, "wrong source URL"):
            prepare(self.experiment, self.root, self.root / "invalid.jsonl")


if __name__ == "__main__":
    unittest.main()
