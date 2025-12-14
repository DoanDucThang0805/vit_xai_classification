"""
Dataset Configuration Module.

This module sets up data pipelines for the plant disease classification task.
It loads the PlantVillage dataset and applies data augmentation transformations
using Albumentations library.

The module provides three augmented datasets:
    - train_dataset: Training split with aggressive augmentation
    - val_dataset: Validation split with moderate augmentation
    - test_dataset: Test split with only normalization (no augmentation)

Augmentations include rotation, flipping, brightness/contrast adjustments, and
color space transformations to improve model robustness.
"""

from pathlib import Path
from utils.utils import LoadDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


root_dir = Path(__file__).resolve().parents[2] / 'data' / 'PlantVillage'

# Training data augmentation pipeline
# Applies various transformations to increase dataset diversity and prevent overfitting
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=30,
        p=0.5
    ),
    A.RandomGamma(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.RGBShift(
        r_shift_limit=15,
        g_shift_limit=15,
        b_shift_limit=15,
        p=0.3
    ),
    A.CLAHE(
        clip_limit=4.0,
        p=0.3
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# Validation data augmentation pipeline
# Similar to training but used to monitor model performance during training
val_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=30,
        p=0.5
    ),
    A.RandomGamma(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.RGBShift(
        r_shift_limit=15,
        g_shift_limit=15,
        b_shift_limit=15,
        p=0.3
    ),
    A.CLAHE(
        clip_limit=4.0,
        p=0.3
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# Test data transformation pipeline
# Minimal transformations: only resizing and normalization, no augmentation
test_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


train_dataset = LoadDataset(root_dir=root_dir, split='train', transform=train_transform)
val_dataset = LoadDataset(root_dir=root_dir, split='val', transform=val_transform)
test_dataset = LoadDataset(root_dir=root_dir, split='test', transform=test_transform)
