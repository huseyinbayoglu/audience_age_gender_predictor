"""PyTorch Dataset reading split CSVs and JPEGs from dataset/images/."""
import io
from pathlib import Path
from typing import Sequence

import numpy as np
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
    """
    Modes (in order of speed):
      preload=False, decoded_cache=False  → read JPEG from disk every step (slow)
      preload=True                        → JPEGs in RAM as bytes; still decode each step
      decoded_cache=True, decoded_size=N  → fully decoded uint8 tensor [N,3,H,W] in RAM
                                            (no PIL/decode at runtime — fastest)
    """
    def __init__(self, split: str, task: str, transform=None, preload: bool = False,
                 decoded_cache: bool = False, decoded_size: int = 160):
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

        self._bytes_cache = None
        self._decoded_cache = None  # uint8 torch.Tensor [N, 3, H, W]

        if decoded_cache:
            n = len(self.ids)
            arr = np.empty((n, decoded_size, decoded_size, 3), dtype=np.uint8)
            for i, uid in enumerate(tqdm(self.ids, desc=f"decode→RAM {split}", leave=False)):
                with Image.open(IMAGES_DIR / f"{uid}.jpg") as im:
                    im = im.convert("RGB").resize((decoded_size, decoded_size),
                                                  Image.BILINEAR)
                    arr[i] = np.asarray(im)
            # to torch CHW uint8 (transpose once, contiguous)
            self._decoded_cache = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            mb = self._decoded_cache.numel() / (1024 ** 2)
            print(f"  decoded cache [{split}]: {self._decoded_cache.shape} uint8, {mb:.0f} MB")
        elif preload:
            self._bytes_cache = []
            for uid in tqdm(self.ids, desc=f"preload {split}", leave=False):
                with open(IMAGES_DIR / f"{uid}.jpg", "rb") as f:
                    self._bytes_cache.append(f.read())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self._decoded_cache is not None:
            # transform is expected to handle uint8 CHW tensors (use v2 transforms).
            img = self._decoded_cache[i]
        elif self._bytes_cache is not None:
            img = Image.open(io.BytesIO(self._bytes_cache[i])).convert("RGB")
        else:
            img = Image.open(IMAGES_DIR / f"{self.ids[i]}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)
