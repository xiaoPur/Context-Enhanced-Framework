import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_report_text(text):
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\r", " ").strip().lower()
    return " ".join(text.split())


def legacy_uid_key(uid):
    return f"ecgen-radiology/{int(uid)}.xml"


def _resolve_path(value, search_roots):
    path = Path(value)
    if path.is_absolute():
        return path
    for root in search_roots:
        candidate = Path(root) / path
        if candidate.exists():
            return candidate
    return Path(search_roots[0]) / path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_label_mapping(label_file):
    raw_labels = _load_json(label_file)
    return {str(key): np.asarray(value, dtype=np.float32) for key, value in raw_labels.items()}


def _load_external_labels(label_file):
    if not label_file:
        return {}

    label_path = Path(label_file)
    if label_path.suffix.lower() == ".json":
        raw_labels = _load_json(label_path)
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in raw_labels.items()}

    label_frame = pd.read_csv(label_path)
    label_columns = [column for column in label_frame.columns if column != "uid"]
    if len(label_columns) < 14:
        raise ValueError("external label file must contain uid plus at least 14 label columns")

    labels = {}
    for row in label_frame.itertuples(index=False):
        uid = getattr(row, "uid")
        legacy_key = legacy_uid_key(uid)
        label_values = [getattr(row, column) for column in label_columns[:14]]
        labels[legacy_key] = np.asarray(label_values, dtype=np.float32)
    return labels


class IndianaRawIUXRAY(data.Dataset):
    def __init__(
        self,
        directory,
        reports_csv="indiana_reports.csv",
        projections_csv="indiana_projections.csv",
        images_dir="images_normalized",
        input_size=(256, 256),
        random_transform=True,
        view_pos=("AP", "PA", "LATERAL"),
        max_views=2,
        sources=("image", "caption", "label", "history"),
        targets=("caption", "label"),
        max_len=1000,
        vocab_file="iu_xray/nlmcxr_unigram_1000.model",
        label_file="iu_xray/file2label.json",
        nounphrase_file="tools/count_nounphrase.json",
        external_label_file=None,
        split_file=None,
        split=None,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        seed=123,
    ):
        super().__init__()

        self.directory = Path(directory)
        self.repo_root = Path(__file__).resolve().parent
        self.reports_path = _resolve_path(reports_csv, [self.directory, self.repo_root])
        self.projections_path = _resolve_path(projections_csv, [self.directory, self.repo_root])
        self.images_dir = _resolve_path(images_dir, [self.directory, self.repo_root])
        self.vocab_path = _resolve_path(vocab_file, [self.repo_root, self.directory])
        self.label_path = _resolve_path(label_file, [self.repo_root, self.directory])
        self.nounphrase_path = _resolve_path(nounphrase_file, [self.repo_root, self.directory])
        self.external_label_path = (
            _resolve_path(external_label_file, [self.directory, self.repo_root])
            if external_label_file
            else None
        )
        self.split_path = Path(split_file) if split_file else None
        if self.split_path and not self.split_path.is_absolute():
            self.split_path = self.directory / self.split_path

        self.sources = list(sources)
        self.targets = list(targets)
        self.max_views = max_views
        self.view_pos = list(view_pos)
        self.dict_positions = dict(zip(self.view_pos, range(len(self.view_pos))))
        self.max_len = max_len
        self.input_size = input_size
        self.random_transform = random_transform
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.split = split
        self.source_sections = ["INDICATION", "COMPARISON"]
        self.target_sections = ["FINDINGS"]
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.vocab_path))
        self.base_label_map = _load_label_mapping(self.label_path)
        self.external_label_map = _load_external_labels(self.external_label_path)
        self.top_np = self._load_nounphrases()
        self.records = self._build_records()
        self.records = self._apply_split(self.records)

        if random_transform:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(0.1, 0.1, 0.1),
                            transforms.RandomRotation(15, expand=True),
                        ]
                    ),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        sources = []
        targets = []

        if "image" in self.sources:
            imgs, vpos = self._load_images(record["images"])
        history_text = record["history"]
        findings_text = record["findings"]

        encoded_source = self._encode_text(history_text)
        encoded_target = self._encode_text(findings_text)
        extra_labels = self._compute_np_labels(findings_text)
        label_vector = np.concatenate([record["base_labels"], extra_labels]).astype(np.float32)

        for source_name in self.sources:
            if source_name == "image":
                sources.append((imgs, vpos))
            elif source_name == "history":
                sources.append(encoded_source)
            elif source_name == "label":
                sources.append(label_vector)
            elif source_name == "caption":
                sources.append(encoded_target)
            elif source_name == "caption_length":
                sources.append(min(record["target_length"], self.max_len))

        for target_name in self.targets:
            if target_name == "label":
                targets.append(label_vector)
            elif target_name == "caption":
                targets.append(encoded_target)
            elif target_name == "caption_length":
                targets.append(min(record["target_length"], self.max_len))

        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def _load_nounphrases(self, top_k=100):
        count_np = _load_json(self.nounphrase_path)
        sorted_count_np = sorted(count_np.items(), key=lambda item: item[1], reverse=True)
        return [key for key, _ in sorted_count_np[:top_k]]

    def _build_records(self):
        reports = pd.read_csv(self.reports_path, dtype=object).fillna("")
        projections = pd.read_csv(self.projections_path, dtype=object).fillna("")

        grouped_images = {}
        for row in projections.itertuples(index=False):
            uid = str(getattr(row, "uid"))
            projection = str(getattr(row, "projection", "")).upper()
            filename = str(getattr(row, "filename"))
            if self.view_pos and projection and projection not in self.view_pos:
                continue
            grouped_images.setdefault(uid, []).append(
                {
                    "filename": filename,
                    "projection": projection if projection in self.dict_positions else (self.view_pos[0] if self.view_pos else "PA"),
                }
            )

        records = []
        for row in reports.itertuples(index=False):
            uid = str(getattr(row, "uid"))
            findings = normalize_report_text(getattr(row, "findings", ""))
            if not findings:
                continue

            image_entries = []
            for image_info in grouped_images.get(uid, []):
                image_path = self.images_dir / image_info["filename"]
                if image_path.exists():
                    image_entries.append(image_info)
            if not image_entries:
                continue

            history_parts = [
                normalize_report_text(getattr(row, "indication", "")),
                normalize_report_text(getattr(row, "comparison", "")),
            ]
            history = " ".join(part for part in history_parts if part)
            key = legacy_uid_key(uid)
            base_labels = self.external_label_map.get(key, self.base_label_map.get(key, np.zeros(14, dtype=np.float32)))

            encoded_target = [self.vocab.bos_id()] + self.vocab.encode(findings) + [self.vocab.eos_id()]
            records.append(
                {
                    "uid": uid,
                    "legacy_key": key,
                    "history": history,
                    "findings": findings,
                    "images": image_entries,
                    "base_labels": np.asarray(base_labels, dtype=np.float32),
                    "target_length": len(encoded_target),
                }
            )
        return records

    def _apply_split(self, records):
        if self.split not in {"train", "val", "test"}:
            return records

        split_map = None
        if self.split_path and self.split_path.exists():
            split_map = _load_json(self.split_path)
        else:
            uids = [record["uid"] for record in records]
            rng = np.random.RandomState(self.seed)
            indices = rng.permutation(len(uids))
            train_pvt = int(len(indices) * self.train_size)
            val_pvt = int(len(indices) * (self.train_size + self.val_size))
            split_map = {
                "train": [uids[index] for index in indices[:train_pvt]],
                "val": [uids[index] for index in indices[train_pvt:val_pvt]],
                "test": [uids[index] for index in indices[val_pvt:]],
            }
            if self.split_path:
                self.split_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.split_path, "w", encoding="utf-8") as handle:
                    json.dump(split_map, handle, indent=2, ensure_ascii=False)

        split_uids = set(split_map[self.split])
        return [record for record in records if record["uid"] in split_uids]

    def _encode_text(self, text):
        encoded = [self.vocab.bos_id()] + self.vocab.encode(text) + [self.vocab.eos_id()]
        encoded_text = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        encoded_text[: min(len(encoded), self.max_len)] = encoded[: min(len(encoded), self.max_len)]
        return encoded_text

    def _compute_np_labels(self, findings_text):
        np_labels = np.zeros(len(self.top_np), dtype=np.float32)
        for idx, nounphrase in enumerate(self.top_np):
            if nounphrase in findings_text:
                np_labels[idx] = 1.0
        return np_labels

    def _load_images(self, image_entries):
        imgs = []
        vpos = []
        ordered_entries = list(image_entries)
        if self.random_transform:
            indices = np.random.permutation(len(ordered_entries))
            ordered_entries = [ordered_entries[index] for index in indices]

        for image_info in ordered_entries[: self.max_views]:
            image_path = self.images_dir / image_info["filename"]
            img = Image.open(image_path).convert("RGB")
            imgs.append(self.transform(img).unsqueeze(0))
            vpos.append(self.dict_positions.get(image_info["projection"], 0))

        if not imgs:
            raise ValueError("record must contain at least one image")

        for _ in range(len(imgs), self.max_views):
            imgs.append(torch.zeros_like(imgs[0]))
            vpos.append(-1)

        return torch.cat(imgs, dim=0), np.asarray(vpos, dtype=np.int64)
