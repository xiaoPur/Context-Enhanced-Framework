import json
import unittest
import uuid
from pathlib import Path

import pandas as pd
from PIL import Image

from datasets import IndianaRawIUXRAY


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP_ROOT = REPO_ROOT / ".tmp_tests"
LOCAL_TMP_ROOT.mkdir(exist_ok=True)


class IndianaRawDatasetTests(unittest.TestCase):
    def test_indiana_raw_dataset_builds_multiview_sample(self):
        root = LOCAL_TMP_ROOT / f"dataset_case_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        images_dir = root / "images_normalized"
        images_dir.mkdir()

        for image_name in ("0001_a.dcm.png", "0001_b.dcm.png"):
            Image.new("RGB", (12, 12), color=(255, 255, 255)).save(images_dir / image_name)

        reports = pd.DataFrame(
            [
                {
                    "uid": 1,
                    "MeSH": "",
                    "Problems": "",
                    "image": "",
                    "indication": "Cough",
                    "comparison": "No prior study",
                    "findings": "No acute cardiopulmonary abnormality.",
                    "impression": "Normal chest.",
                }
            ]
        )
        reports.to_csv(root / "indiana_reports.csv", index=False)

        projections = pd.DataFrame(
            [
                {"uid": 1, "filename": "0001_a.dcm.png", "projection": "PA"},
                {"uid": 1, "filename": "0001_b.dcm.png", "projection": "LATERAL"},
            ]
        )
        projections.to_csv(root / "indiana_projections.csv", index=False)

        label_path = root / "labels.json"
        label_path.write_text(json.dumps({"ecgen-radiology/1.xml": [1] + [0] * 13}), encoding="utf-8")

        nounphrase_path = root / "count_nounphrase.json"
        nounphrase_path.write_text(
            json.dumps({f"np_{idx}": 100 - idx for idx in range(100)}),
            encoding="utf-8",
        )

        dataset = IndianaRawIUXRAY(
            directory=str(root),
            reports_csv="indiana_reports.csv",
            projections_csv="indiana_projections.csv",
            images_dir="images_normalized",
            input_size=(32, 32),
            random_transform=False,
            max_views=2,
            max_len=32,
            vocab_file=str(REPO_ROOT / "iu_xray" / "nlmcxr_unigram_1000.model"),
            label_file=str(label_path),
            nounphrase_file=str(nounphrase_path),
            seed=123,
        )

        sample_source, sample_target = dataset[0]
        imgs, vpos = sample_source[0]

        self.assertEqual(imgs.shape, (2, 3, 32, 32))
        self.assertEqual(vpos.shape[0], 2)
        self.assertEqual(sample_source[3].shape[0], dataset.max_len)
        self.assertEqual(sample_target[0].shape[0], dataset.max_len)
        self.assertEqual(sample_target[1].shape[0], 114)


if __name__ == "__main__":
    unittest.main()
