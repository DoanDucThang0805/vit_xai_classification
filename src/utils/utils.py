"""
Utility Functions and Dataset Loading Classes.

This module provides helper functions and dataset classes for loading and processing
plant disease images. It includes the LoadDataset class for structured data loading
with support for train/validation/test splits.

Key Features:
    - Automatic dataset splitting with stratification
    - Class to index mapping for label encoding
    - Image loading with PIL
    - Transform pipeline support via Albumentations
"""

import os
from typing import List, Tuple, Dict, Literal, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image
from sklearn.model_selection import train_test_split


class LoadDataset(Dataset):
    """
    Dataset loader for plant disease classification.
    
    Loads images from a structured directory of disease classes and provides
    automatic train/validation/test splitting with stratification to ensure
    balanced class distribution across splits.
    
    The dataset expects a directory structure like:
        root_dir/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_Early_blight/
        ├── Tomato_healthy/
        └── ... (other disease classes)
    
    Attributes:
        root_dir (Path): Root directory containing disease class folders
        split (str): Dataset split ('train', 'val', or 'test')
        train_ratio (float): Proportion of data for training
        image_paths (List[str]): List of image file paths for the selected split
        labels (List[int]): Class labels corresponding to image_paths
        class_to_idx (Dict[str, int]): Mapping from class name to class index
        idx_to_class (Dict[int, str]): Mapping from class index to class name
    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal['train', 'validation', 'test'],
        train_ratio: float = 0.8,
        transform: transforms.Compose = None
    ) -> None:
        """
        Initialize the dataset loader.
        
        Args:
            root_dir (Path): Root directory containing class subdirectories
            split (str, optional): Dataset split - 'train', 'val', or 'test'. Defaults to 'train'.
            train_ratio (float, optional): Proportion of data for training (0 to 1). Defaults to 0.8.
            transform (transforms.Compose, optional): Image transformation pipeline. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.image_paths, self.labels, self.class_to_idx, self.idx_to_class = self._split_dataset()

    def _load_image(self, root_dir: Path) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Load all images and labels from the root directory.
        
        Scans the root directory for subdirectories starting with "Tomato" (disease classes),
        collects all image files from each class, and creates class-to-index mappings.
        
        Args:
            root_dir (Path): Root directory containing class subdirectories
            
        Returns:
            Tuple containing:
                - image_paths (List[str]): Absolute paths to all image files
                - labels (List[int]): Class labels (0-indexed) corresponding to each image
                - class_to_idx (Dict[str, int]): Mapping from class name to class index
                - idx_to_class (Dict[int, str]): Mapping from class index to class name
        """
        class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))
             and d.startswith("Tomato")]
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        image_paths = []
        labels = []
        for class_name in class_names:
            dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(dir, fname))
                    labels.append(class_to_idx[class_name])
        return image_paths, labels, class_to_idx, idx_to_class

    def _split_dataset(self) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Split dataset into train, validation, and test sets.
        
        Uses stratified sampling to ensure balanced class distribution across all splits:
        - 80% training, 20% temporary (from train_ratio)
        - Temporary split: 50% validation, 50% test
        
        Returns:
            Tuple containing:
                - image_paths (List[str]): Image paths for the selected split
                - labels (List[int]): Labels for the selected split
                - class_to_idx (Dict[str, int]): Class name to index mapping
                - idx_to_class (Dict[int, str]): Index to class name mapping
                
        Raises:
            ValueError: If split is not 'train', 'val', or 'test'
        """
        image_paths, labels, class_to_idx, idx_to_class = self._load_image(self.root_dir)

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size= 1-self.train_ratio, stratify=labels, random_state=42, shuffle=True
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42, shuffle=True
        )

        if self.split == 'train':
            return train_paths, train_labels, class_to_idx, idx_to_class
        elif self.split == 'validation':
            return val_paths, val_labels, class_to_idx, idx_to_class
        elif self.split == 'test':
            return test_paths, test_labels, class_to_idx, idx_to_class
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")

    def __len__(self) -> int:
        """
        Return the total number of samples in this dataset split.
        
        Returns:
            int: Number of images in the current split
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset by index.
        
        Loads an image from disk, applies transformations if specified, and returns
        the transformed image tensor and its corresponding class label.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - image (torch.Tensor): Transformed image tensor
                - label (int): Class label (0-indexed)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            augumented = self.transform(image=image)
            image = augumented["image"]
        return image, label


class LoadPlantdocDataset(Dataset):
    """
    PyTorch Dataset class for loading PlantDoc plant disease images.
    
    Automatically loads images from train and test directories, splits them into
    train/validation/test sets, and provides image-label pairs for model training.
    
    Attributes:
        train_dir (Path): Path to the training data directory.
        test_dir (Path): Path to the test data directory.
        split (str): Dataset split type - "train", "validation", or "test".
        validation_ratio (float): Ratio of validation set from training data.
        transform: Optional data augmentation transforms to apply.
        images_paths (List[str]): List of image file paths.
        labels (List[int]): List of corresponding labels.
        class_to_idx (Dict[str, int]): Mapping from class name to index.
        idx_to_class (Dict[int, str]): Mapping from index to class name.
    """
    
    def __init__(
        self,
        train_dir: Path,
        test_dir: Path,
        split: Literal["train", "validation", "test"],
        validation_ratio: float,
        transform: None
    ) -> None:
        """
        Initialize the LoadPlantdocDataset.
        
        Args:
            train_dir (Path): Path to the training data directory.
            test_dir (Path): Path to the test data directory.
            split (Literal["train", "validation", "test"]): Dataset split to load.
            validation_ratio (float): Ratio of samples to use for validation (0.0 to 1.0).
            transform: Optional albumentations transform for data augmentation.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.split = split
        self.validation_ratio = validation_ratio
        self.transform = transform
        self.images_paths, self.labels, self.class_to_idx, self.idx_to_class = self._split_dataset()

    def _load_image(self, root_dir: Path) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Load image paths and labels from a directory.
        
        Recursively searches for images in subdirectories starting with "Tomato" and
        creates class-to-index mappings.
        
        Args:
            root_dir (Path): Root directory containing class subdirectories.
            
        Returns:
            Tuple containing:
                - image_paths (List[str]): List of full image file paths.
                - labels (List[int]): List of class indices corresponding to images.
                - class_to_idx (Dict[str, int]): Mapping from class name to index.
                - idx_to_class (Dict[int, str]): Mapping from index to class name.
        """
        class_names = sorted(
            [
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
                and d.startswith("Tomato")
            ]
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        image_paths = []
        labels = []
        for class_name in class_names:
            dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(dir, fname))
                    labels.append(class_to_idx[class_name])
        return image_paths, labels, class_to_idx, idx_to_class
    
    def _split_dataset(self) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Split dataset into train, validation, and test sets.
        
        Returns the appropriate split based on the initialized split parameter.
        Uses stratified train-test split to maintain class distribution.
        
        Returns:
            Tuple containing image paths, labels, and class mappings for the selected split.
        """
        train_val_image_paths, train_val_labels, train_val_class_to_idx, train_val_idx_to_class = self._load_image(self.train_dir)
        test_image_paths, test_labels, test_class_to_idx, test_idx_to_class = self._load_image(self.test_dir)
        train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(train_val_image_paths, train_val_labels, test_size=self.validation_ratio, random_state=42, stratify=train_val_labels)
        if self.split == "train":
            return train_image_paths, train_label_paths, train_val_class_to_idx, train_val_idx_to_class
        elif self.split == "validation":
            return val_image_paths, val_label_paths, train_val_class_to_idx, train_val_idx_to_class
        else:
            return test_image_paths, test_labels, test_class_to_idx, test_idx_to_class
        
    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.
        
        Returns:
            int: Number of images in the current split.
        """
        return len(self.images_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """
        Get a single image and its label by index.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the image and its label.
        """
        image_path = self.images_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            augumented = self.transform(image=image)
            image = augumented['image']
        return image, label

class LoadCroppedDataset(Dataset):
    """
    PyTorch Dataset class for loading cropped plant disease images.
    
    Loads images from train and test folders, splits into train/validation/test sets,
    and provides image-label pairs for model training and evaluation.
    
    Attributes:
        cropped_data_path (Path): Path to the cropped data directory containing 'train' and 'test' folders.
        split (str): Dataset split type - 'train', 'validation', or 'test'.
        validation_ratio (float): Ratio of validation set from training data.
        transform: Optional data augmentation transforms to apply.
        image_paths (List[str]): List of image file paths.
        labels (List[int]): List of corresponding labels.
        class_to_idx (Dict[str, int]): Mapping from class name to index.
        idx_to_class (Dict[int, str]): Mapping from index to class name.
    """
    def __init__(
        self,
        cropped_data_path: Path,
        split: Literal['train', 'validation', 'test'],
        validation_ratio: float,
        transform = None
    ):
        """
        Initialize the cropped plant disease dataset.
        
        Args:
            cropped_data_path (Path): Path to cropped data directory (contains 'train' and 'test' folders).
            split (str): Dataset split type ('train', 'validation', 'test').
            validation_ratio (float): Ratio of validation data from train set.
            transform: Data augmentation pipeline (optional).
        """
        super(LoadCroppedDataset, self).__init__()
        self.cropped_data_path = cropped_data_path
        self.split = split
        self.validation_ratio = validation_ratio
        self.transform = transform
        self.image_paths, self.labels, self.class_to_idx, self.idx_to_class = self._split_data()

    def _load_images(self, data_dir: Path) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Load all images and labels from the cropped data directory.
        
        Args:
            data_dir (Path): Path to cropped data directory ('train', 'test').
        
        Returns:
            Tuple containing:
                - image_paths (List[str]): Image file paths.
                - labels (List[int]): Corresponding labels.
                - class_to_idx (Dict[str, int]): Mapping from class name to index.
                - idx_to_class (Dict[int, str]): Mapping from index to class name.
        """
        train_path = data_dir / 'train'
        test_path = data_dir / 'test'
        class_names = sorted([
            d for d in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, d))
        ])
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        labels, image_paths = [], []
        for class_name in class_names:
            train_dir = os.path.join(train_path, class_name)
            test_dir = os.path.join(test_path, class_name)
            for fname in os.listdir(train_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(train_dir, fname))
                    labels.append(class_to_idx[class_name])
            for fname in os.listdir(test_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(test_dir, fname))
                    labels.append(class_to_idx[class_name])
        return image_paths, labels, class_to_idx, idx_to_class
    
    def _split_data(self) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            Tuple containing:
                - image_paths (List[str]): Image paths for the selected split.
                - labels (List[int]): Labels for the selected split.
                - class_to_idx (Dict[str, int]): Mapping from class name to index.
                - idx_to_class (Dict[int, str]): Mapping from index to class name.
        """
        image_paths, labels, class_to_idx, idx_to_class = self._load_images(self.cropped_data_path)
        train_image_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=self.validation_ratio, random_state=42, stratify=labels, shuffle=True)
        val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels, shuffle=True)
        if self.split == 'train':
            return train_image_paths, train_labels, class_to_idx, idx_to_class
        if self.split == 'validation':
            return val_image_paths, val_labels, class_to_idx, idx_to_class
        if self.split == 'test':
            return test_image_paths, test_labels, class_to_idx, idx_to_class
        
    def __len__(self) -> int:
        """
        Return the number of samples in the current split.
        
        Returns:
            int: Number of images.
        """
        return(len(self.image_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Tuple[torch.Tensor, int]: Transformed image and corresponding label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            augumented = self.transform(image=image)
            image = augumented["image"]
        return image, label
