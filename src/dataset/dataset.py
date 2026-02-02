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
from collections import Counter
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
    # A.HorizontalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate(
    #     shift_limit=0.05,
    #     scale_limit=0.05,
    #     rotate_limit=30,
    #     p=0.5
    # ),
    # A.RandomGamma(p=0.2),
    # A.RandomBrightnessContrast(p=0.3),
    # A.RGBShift(
    #     r_shift_limit=15,
    #     g_shift_limit=15,
    #     b_shift_limit=15,
    #     p=0.3
    # ),
    # A.CLAHE(
    #     clip_limit=4.0,
    #     p=0.3
    # ),
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

robust_test_transform = A.Compose([
    A.Resize(height=224, width=224),

    # --- NHÓM 1: ÁNH SÁNG (Nhẹ nhàng) ---
    # Luôn luôn thay đổi ánh sáng một chút (vì ngoài trời không bao giờ sáng chuẩn)
    A.RandomBrightnessContrast(
        brightness_limit=0.2, # Giảm xuống 0.2 cho an toàn
        contrast_limit=0.2,
        p=1.0 
    ),

    # --- NHÓM 2: NHIỄU NẶNG (Dùng OneOf để không bị chồng chéo) ---
    # Mỗi ảnh chỉ bị dính 1 trong 3 chiêu này thôi -> Không bị nát ảnh
    A.OneOf([
        # Chiêu 1: Rung tay (Motion Blur)
        A.MotionBlur(blur_limit=5, p=1.0), # Giảm limit từ 7 xuống 5 cho đỡ chóng mặt
        
        # Chiêu 2: Nhiễu hạt (Sensor Noise)
        A.GaussNoise(var_limit=(10.0, 30.0), p=1.0), # Giảm var_limit max xuống 30
        
        # Chiêu 3: Mất nét (Gaussian Blur)
        A.GaussianBlur(blur_limit=5, p=1.0),
        
        # Chiêu 4: Che khuất (Occlusion) - Nhẹ thôi
        A.CoarseDropout(
            max_holes=4,       # Giảm số lỗ xuống 4
            max_height=32,
            max_width=32,
            fill_value=0,
            p=1.0
        ),
    ], p=1.0), # <--- QUAN TRỌNG: p=1.0 nghĩa là chắc chắn sẽ dính 1 trong 4 chiêu trên

    # Chuẩn hóa
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Load datasets with respective transformations
train_dataset = LoadDataset(root_dir=root_dir, split='train', transform=train_transform)
validation_dataset = LoadDataset(root_dir=root_dir, split='validation', transform=val_transform)
test_dataset = LoadDataset(root_dir=root_dir, split='test', transform=test_transform)
robust_test_dataset = LoadDataset(root_dir=root_dir, split='test', transform=robust_test_transform)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(validation_dataset)}")
print(f"Test size: {len(test_dataset)}")
print(f"Robust Test size: {len(robust_test_dataset)}")
print(f"Numbers of train labels: {Counter(train_dataset.labels)}")
print(f"Numbers of validation labels: {Counter(validation_dataset.labels)}")
print(f"Numbers of test labels: {Counter(test_dataset.labels)}")
print(train_dataset.class_to_idx)
print(train_dataset[0][0].shape)