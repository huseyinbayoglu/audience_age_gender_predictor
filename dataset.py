"""PyTorch Dataset reading split CSVs and JPEGs from dataset/images/."""
import io
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "dataset" / "images"
SPLIT_DIR = ROOT / "dataset" / "splits"

GENDER_CLASSES = ["male", "female", "unknown"]
AGE_CLASSES = ["18-", "18-24", "25-34", "35-44", "45+", "unknown"]

LABEL_COL = {
    "gender": "vllm_gender_prediction",
    "age": "vllm_age_range_prediction",
}
CLASSES = {"gender": GENDER_CLASSES, "age": AGE_CLASSES}


class AvatarDataset(Dataset):
    def __init__(self, split: str, task: str, transform=None, preload: bool = False):
        assert split in ("train", "val", "test")
        assert task in ("gender", "age")
        self.task = task
        self.classes: Sequence[str] = CLASSES[task]
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        df = pd.read_csv(SPLIT_DIR / f"{split}.csv", dtype={"id": str})
        col = LABEL_COL[task]
        df = df[df[col].isin(self.classes)].reset_index(drop=True)
        self.ids = df["id"].tolist()
        self.labels = [self.cls2idx[v] for v in df[col].tolist()]

        # Optional in-memory cache of JPEG bytes — kills disk-I/O bottleneck
        # on slow filesystems (e.g. Colab /content). Decoding still happens in
        # __getitem__ (cheap, parallelized by DataLoader workers).
        self._bytes_cache = None
        if preload:
            self._bytes_cache = []
            for uid in tqdm(self.ids, desc=f"preload {split}", leave=False):
                with open(IMAGES_DIR / f"{uid}.jpg", "rb") as f:
                    self._bytes_cache.append(f.read())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self._bytes_cache is not None:
            img = Image.open(io.BytesIO(self._bytes_cache[i])).convert("RGB")
        else:
            img = Image.open(IMAGES_DIR / f"{self.ids[i]}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)
