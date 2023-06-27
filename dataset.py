import torch
from typing import List

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import load_and_resample, BASE_SAMPLE_RATE, make_feat_extractor_and_tokenizer


class SpeechDataset(Dataset):
    def __init__(self, data_files: List[str]):
        super().__init__()

        self.data = []
        for file in data_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.batch_size = batch_size
        feat_extractor, tokenizer = make_feat_extractor_and_tokenizer(
            **cfg.dataset.processor
        )
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        self.train_set = ConcatDataset(
            [SpeechDataset(path) for path in self.cfg.dataset.path.train]
        )
        self.test_set = ConcatDataset(
            [SpeechDataset(path) for path in self.cfg.dataset.path.test]
        )
        self.dev_set = ConcatDataset(
            [SpeechDataset(path) for path in self.cfg.dataset.path.dev]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.cfg.dataloader)

    def val_dataloader(self):
        return DataLoader(self.dev_set, shuffle=False, **self.cfg.dataloader)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, **self.cfg.dataloader)

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
