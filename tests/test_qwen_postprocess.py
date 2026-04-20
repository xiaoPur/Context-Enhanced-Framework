import unittest

from qwen_postprocess import (
    build_qwen_output_record,
    build_qwen_rewrite_prompt,
    normalize_report_text_for_metrics,
)


class QwenPromptTests(unittest.TestCase):
    def test_build_qwen_rewrite_prompt_contains_constraints(self):
        prompt = build_qwen_rewrite_prompt(
            history="cough . no prior study .",
            draft="lungs are clear .",
        )

        self.assertIn("Only revise the report text", prompt)
        self.assertIn("Do not introduce new findings", prompt)
        self.assertIn("cough . no prior study .", prompt)
        self.assertIn("lungs are clear .", prompt)

    def test_normalize_report_text_for_metrics_aligns_case_and_punctuation(self):
        normalized = normalize_report_text_for_metrics("Lungs are clear. No pleural effusion!")

        self.assertEqual(normalized, "lungs are clear . no pleural effusion !")

    def test_build_qwen_output_record_preserves_raw_text_and_adds_normalized_text(self):
        updated_record = build_qwen_output_record(
            {"uid": "1", "hypothesis": "lungs are clear ."},
            "Lungs are clear.",
        )

        self.assertEqual(updated_record["qwen_hypothesis"], "Lungs are clear.")
        self.assertEqual(updated_record["qwen_hypothesis_normalized"], "lungs are clear .")


if __name__ == "__main__":
    unittest.main()
