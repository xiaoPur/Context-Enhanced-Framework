import unittest

from qwen_postprocess import build_qwen_rewrite_prompt


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


if __name__ == "__main__":
    unittest.main()
