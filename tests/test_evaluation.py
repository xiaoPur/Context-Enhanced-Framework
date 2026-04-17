import unittest
import uuid
from pathlib import Path

from evaluation import compute_report_metrics, write_report_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP_ROOT = REPO_ROOT / ".tmp_tests"
LOCAL_TMP_ROOT.mkdir(exist_ok=True)


class EvaluationTests(unittest.TestCase):
    def test_compute_report_metrics_returns_bleu_and_rouge(self):
        metrics = compute_report_metrics(
            references=["no acute cardiopulmonary abnormality ."],
            hypotheses=["no acute cardiopulmonary abnormality ."],
        )

        self.assertEqual(metrics["bleu_1"], 1.0)
        self.assertEqual(metrics["bleu_4"], 1.0)
        self.assertEqual(metrics["rouge_l"], 1.0)

    def test_write_report_outputs_creates_expected_files(self):
        output_dir = LOCAL_TMP_ROOT / f"evaluation_case_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True)
        metrics = {"bleu_1": 1.0, "bleu_2": 1.0, "bleu_3": 1.0, "bleu_4": 1.0, "rouge_l": 1.0}
        records = [{"uid": "1", "reference": "a", "hypothesis": "a"}]

        write_report_outputs(
            output_dir=str(output_dir),
            references=["a"],
            hypotheses=["a"],
            metrics=metrics,
            records=records,
        )

        self.assertTrue((output_dir / "references.txt").exists())
        self.assertTrue((output_dir / "hypotheses.txt").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
