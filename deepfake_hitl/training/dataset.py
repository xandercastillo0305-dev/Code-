"""Face-crop datasets (Section 3.5).

Expected layout (already-extracted face crops, e.g. by training/extract_faces.py):

    data/train/real/*.png   data/train/fake/*.png
    data/val/real/...       data/val/fake/...
    data/test/real/...      data/test/fake/...

Labels: real = 0, fake (Deepfake) = 1. Works for FaceForensics++, Celeb-DF
and DFDC once their frames are converted to this layout.

Pair files (for the similarity/threshold method, used by
pipeline/calibrate_threshold.py and evaluation/evaluate_model.py):

    data/<split>/pairs.csv   columns: suspect,reference,label   (label = real|fake,
                             paths relative to the csv's folder)
"""
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

IMG_EXTS = (".jpg", ".jpeg", ".png")
CLASSES = {"real": 0, "fake": 1}


def train_transform(size=config.IMAGE_SIZE):
    """Augmentation from Section 3.5: horizontal flip, rotation, scaling."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(size=config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class FaceCropDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform or eval_transform()
        self.samples = []
        for cls, label in CLASSES.items():
            folder = os.path.join(self.root, cls)
            if not os.path.isdir(folder):
                continue
            for dirpath, _, files in os.walk(folder):
                for f in sorted(files):
                    if f.lower().endswith(IMG_EXTS):
                        self.samples.append((os.path.join(dirpath, f), label))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}/{{real,fake}}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(label, dtype=torch.float32)

    def class_counts(self):
        n_fake = sum(l for _, l in self.samples)
        return {"real": len(self.samples) - n_fake, "fake": n_fake}


def read_pairs(csv_path):
    """Return a list of (suspect_path, reference_path, label 0/1)."""
    base = os.path.dirname(os.path.abspath(csv_path))
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().lower()
            if label not in CLASSES:
                raise ValueError(f"Bad label {row['label']!r} in {csv_path}")
            out.append((os.path.join(base, row["suspect"]), os.path.join(base, row["reference"]),
                        CLASSES[label]))
    return out
