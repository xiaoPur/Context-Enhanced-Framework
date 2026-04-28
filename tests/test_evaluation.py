import unittest
import uuid
from unittest import mock
from pathlib import Path

from evaluation import compute_report_metrics, write_report_outputs
from qwen_postprocess import normalize_report_text_for_metrics

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
        self.assertNotIn("meteor", metrics)
        self.assertNotIn("cider", metrics)

    def test_compute_report_metrics_can_include_paper_metrics_from_external_scorers(self):
        class FakeMeteor:
            def compute_score(self, references_by_id, hypotheses_by_id):
                assert references_by_id == {0: ["no acute cardiopulmonary abnormality ."]}
                assert hypotheses_by_id == {0: ["no acute cardiopulmonary abnormality ."]}
                return 0.8754321, [0.8754321]

        class FakeCider:
            def compute_score(self, references_by_id, hypotheses_by_id):
                assert references_by_id == {0: ["no acute cardiopulmonary abnormality ."]}
                assert hypotheses_by_id == {0: ["no acute cardiopulmonary abnormality ."]}
                return 1.2345678, [1.2345678]

        metrics = compute_report_metrics(
            references=["no acute cardiopulmonary abnormality ."],
            hypotheses=["no acute cardiopulmonary abnormality ."],
            include_paper_metrics=True,
            scorer_factories={"meteor": FakeMeteor, "cider": FakeCider},
        )

        self.assertEqual(metrics["meteor"], 0.875432)
        self.assertEqual(metrics["cider"], 1.234568)

    def test_compute_report_metrics_requires_external_dependency_for_paper_metrics(self):
        with mock.patch("evaluation._load_coco_scorer_factories", side_effect=ImportError("missing pycocoevalcap")):
            with self.assertRaises(ImportError) as context:
                compute_report_metrics(
                    references=["no acute cardiopulmonary abnormality ."],
                    hypotheses=["no acute cardiopulmonary abnormality ."],
                    include_paper_metrics=True,
                )

        self.assertIn("METEOR/CIDEr", str(context.exception))

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

    def test_normalized_qwen_text_restores_metric_alignment(self):
        references = ["lungs are clear . no pleural effusion ."]
        raw_qwen_hypotheses = ["Lungs are clear. No pleural effusion."]
        normalized_qwen_hypotheses = [normalize_report_text_for_metrics(raw_qwen_hypotheses[0])]

        raw_metrics = compute_report_metrics(references, raw_qwen_hypotheses)
        normalized_metrics = compute_report_metrics(references, normalized_qwen_hypotheses)

        self.assertLess(raw_metrics["bleu_4"], 1.0)
        self.assertEqual(normalized_metrics["bleu_4"], 1.0)
        self.assertEqual(normalized_metrics["rouge_l"], 1.0)


if __name__ == "__main__":
    unittest.main()
