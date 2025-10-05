# dataset_lightcurve.py
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np

class LightCurveDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=2000, normalize=True, max_len=None):
        """
        Args:
            root_dir (str or Path): directory containing subfolders 0/, 1/, ...
            transform: optional torch transform
            seq_len: pad or truncate each light curve to this length
            normalize: whether to normalize LC_DETREND values
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.seq_len = seq_len
        self.normalize = normalize
        self.max_len = max_len

        self.samples = []  # list of (filepath, label)
        for label_dir in sorted(self.root_dir.iterdir()):
            if label_dir.is_dir():
                label = int(label_dir.name)
                for csv_file in label_dir.glob("*.csv"):
                    self.samples.append((csv_file, label))

        print(f"[INFO] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label = self.samples[idx]

        try:
            df = pd.read_csv(csv_path)
            x = torch.tensor(df["LC_DETREND"].values, dtype=torch.float32)
        except Exception as e:
            print(f"[WARN] Skipping file {csv_path}: {e}")
            x = torch.zeros(128)  # fallback

        if self.max_len:
            # Static padding/truncation
            if len(x) > self.max_len:
                x = x[:self.max_len]
            elif len(x) < self.max_len:
                pad = torch.zeros(self.max_len - len(x))
                x = torch.cat([x, pad])

        # DO NOT unsqueeze here if using pad_sequence later
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
