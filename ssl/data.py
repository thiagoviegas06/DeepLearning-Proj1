"""Data pipeline for the CIFAR-10 Kaggle competition.

- Loads the provided competition data (train_images/, train_labels.csv,
  test_images/, test.csv) fully into memory for fast training.
- Fixed-seed stratified 45k/5k train/val split (reproducibility rule).
- Three transform modes: supervised-train, contrastive (two SimCLR views),
  and eval.
"""
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

DATA_DIR = "data"
# normalization stats computed from the provided training images
MEAN = (0.4907, 0.4818, 0.4465)
STD = (0.2450, 0.2407, 0.2600)

sup_train_tf = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# SimCLR-style augmentations for contrastive pretraining (Chen et al., 2020)
contrastive_tf = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

eval_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class TwoCropsTransform:
    """Apply the contrastive transform twice -> two views of one image."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, x):
        return [self.tf(x), self.tf(x)]


class InMemoryImages(Dataset):
    """All images preloaded as uint8 HWC numpy arrays; transform applied on access."""

    def __init__(self, images, filenames, targets, tf):
        self.images = images          # (N, 32, 32, 3) uint8
        self.filenames = filenames
        self.targets = targets        # list[int] or None (test set)
        self.tf = tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        out = self.tf(img)
        if self.targets is None:
            return out, self.filenames[i]
        return out, self.targets[i]


def _load_split():
    """Read train_labels.csv and preload all train images."""
    with open(os.path.join(DATA_DIR, "train_labels.csv")) as f:
        rows = list(csv.DictReader(f))
    filenames = [r["filename"] for r in rows]
    targets = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    for i, fn in enumerate(filenames):
        with Image.open(os.path.join(DATA_DIR, "train_images", fn)) as im:
            images[i] = np.asarray(im.convert("RGB"))
    return images, filenames, targets


def stratified_split(targets, val_per_class=500, seed=0):
    """Return (train_idx, val_idx) with a fixed number per class in val."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = [i for i, t in enumerate(targets) if t == c]
        rng.shuffle(idx)
        val_idx += idx[:val_per_class]
        train_idx += idx[val_per_class:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_loaders(batch_size=256, val_batch_size=512, num_workers=4, seed=0):
    images, filenames, targets = _load_split()
    train_idx, val_idx = stratified_split(targets, seed=seed)

    train_ds = InMemoryImages(images[train_idx], [filenames[i] for i in train_idx],
                              targets[train_idx], sup_train_tf)
    val_ds = InMemoryImages(images[val_idx], [filenames[i] for i in val_idx],
                            targets[val_idx], eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_contrastive_loader(batch_size=256, num_workers=4):
    images, _, _ = _load_split()
    ds = InMemoryImages(images, [f"train_{i:05d}.png" for i in range(len(images))],
                        list(range(len(images))), TwoCropsTransform(contrastive_tf))
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def get_test_loader(batch_size=512, num_workers=4):
    with open(os.path.join(DATA_DIR, "test.csv")) as f:
        rows = list(csv.DictReader(f))
    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        with Image.open(os.path.join(DATA_DIR, "test_images", r["filename"])) as im:
            images[i] = np.asarray(im.convert("RGB"))
    ds = InMemoryImages(images, [int(r["id"]) for r in rows], None, eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    tr, va = get_loaders(batch_size=256, num_workers=0)
    print(f"train: {len(tr.dataset)} images, {len(tr)} batches")
    print(f"val:   {len(va.dataset)} images, {len(va)} batches")
    x, y = next(iter(tr))
    print("batch:", tuple(x.shape), "labels:", y[:8].tolist())
    cl = get_contrastive_loader(batch_size=64, num_workers=0)
    (v1, v2), _ = next(iter(cl))
    print("contrastive views:", tuple(v1.shape), tuple(v2.shape))
