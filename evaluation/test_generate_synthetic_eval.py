import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import generate_synthetic_eval as generator


class GenerateSyntheticEvalTests(unittest.TestCase):
    def test_resume_and_reject_changed_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "faq.csv"
            output = Path(temp_dir) / "cases.jsonl"
            with source.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["question", "answer"])
                writer.writeheader()
                writer.writerow({"question": "سؤال اصلی", "answer": "پاسخ اصلی"})
            items = [{"question": f"سؤال {i}", "answer": "پاسخ"} for i in range(3)]
            with patch.object(generator, "CSV_PATH", source), patch.object(generator, "OUTPUT_PATH", output), \
                    patch.object(generator, "generate_cases", return_value=items) as generate, \
                    patch.object(sys, "argv", ["generate_synthetic_eval.py"]):
                generator.main()
                generator.main()
                self.assertEqual(generate.call_count, 1)
                cases = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(cases), 3)
                self.assertEqual(cases[0]["expected_chunk_ids"], ["faq_001"])
                self.assertEqual(cases[0]["review_status"], "unreviewed")
                source.write_text("question,answer\nchanged,data\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    generator.main()


if __name__ == "__main__":
    unittest.main()
