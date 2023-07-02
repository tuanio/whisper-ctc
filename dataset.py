import os
import json
import torch
from typing import List
import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import load_and_resample, BASE_SAMPLE_RATE, make_feat_extractor_and_tokenizer


class SpeechDataset(Dataset):
    def __init__(self, base: str, file_path: str):
        super().__init__()
        self.data = []
        file_type = "indict_tts" if "indict_tts" in file_path else "cv"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                datum = json.loads(line)
                datum["audio_path"] = os.path.join(base[file_type], datum["audio_path"])
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class SpeechDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_extractor, tokenizer = make_feat_extractor_and_tokenizer(
            **cfg.dataset.processor
        )
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        base = cfg.dataset.path.base
        self.train_set = ConcatDataset(
            [SpeechDataset(base, data_files) for data_files in cfg.dataset.path.train]
        )
        self.test_set = ConcatDataset(
            [SpeechDataset(base, data_files) for data_files in cfg.dataset.path.test]
        )
        self.dev_set = ConcatDataset(
            [SpeechDataset(base, data_files) for data_files in cfg.dataset.path.dev]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            **self.cfg.dataloader,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_set,
            shuffle=False,
            **self.cfg.dataloader,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            **self.cfg.dataloader,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        feat_output = self.feat_extractor(
            [load_and_resample(i["audio_path"]) for i in batch],
            sampling_rate=BASE_SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
        )
        feat = feat_output.input_features

        label_output = self.tokenizer(
            [i["label"] for i in batch],
            pading=True,
            truncation=True,
            return_tensors="pt",
        )
        target = label_output.input_ids
        target_length = label_output.attention_mask.sum(axis=1)

        return feat, target, target_length
