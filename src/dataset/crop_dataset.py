"""
Crop and organize tomato disease images from COCO-style annotation to class folders.

This script reads COCO-format annotation files, crops images based on bounding boxes,
and saves them into class-specific folders for training and testing.
Functions:
    - slugify: Normalize and convert disease names to folder-friendly format.
    - crop_image: Crop an image using bounding box coordinates.
    - cropped_tomato_data: Main routine to process and save cropped images by class.
"""

import os
from pathlib import Path
import re
import unicodedata
from typing import Literal, List
import json
from PIL import Image


data_root_dir = Path(__file__).resolve().parents[2] / 'data' / 'cocoplantdoc'
train_dir = data_root_dir / 'train'
test_dir = data_root_dir / 'test'

os.makedirs(Path(__file__).resolve().parents[2] / 'data' / 'cropped_data', exist_ok=True)
cropped_dir = Path(__file__).resolve().parents[2] / 'data' / 'cropped_data'


def slugify(name: str) -> str:
    """
    Normalize and convert a string to a slug (folder-friendly name).
    Removes accents, keeps only alphanumeric and spaces, replaces spaces with underscores.
    Args:
        name (str): Input string (disease name).
    Returns:
        str: Slugified string.
    """
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = re.sub(r"\s+", "_", name).lower()
    return name


def crop_image(image_path: Path, bbox: List):
    """
    Crop an image using bounding box coordinates.
    Args:
        image_path (Path): Path to the image file.
        bbox (List): Bounding box [x, y, w, h].
    Returns:
        PIL.Image: Cropped image.
    """
    img = Image.open(image_path)
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    cropped = img.crop((x, y, x2, y2))
    return cropped


def cropped_tomato_data(data_dir_path: Path, type_data: Literal['train', 'test']):
    """
    Crop and save tomato disease images from COCO annotation to class folders.
    Args:
        data_dir_path (Path): Path to the data directory ('train' or 'test').
        type_data (Literal['train', 'test']): Data split type.
    """
    os.makedirs(cropped_dir / type_data, exist_ok=True)
    annotation_path = data_dir_path / '_annotations.coco.json'
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
    for data in annotation_data['annotations']:
        if not (20 <= data['category_id'] <= 27):
            continue
        disease_name = annotation_data['categories'][data['category_id']]['name']
        disease_slug = slugify(disease_name)
        disease_path = cropped_dir / type_data / disease_slug
        os.makedirs(disease_path, exist_ok=True)
        image_filename = annotation_data['images'][data['image_id']]['file_name']
        image_path = data_dir_path / image_filename
        bbox = data['bbox']
        cropped_image = crop_image(image_path, bbox)
        file_name = f"image_{data['image_id']}_{disease_slug}.jpg"
        save_path = disease_path / file_name
        cropped_image.save(save_path)
    return


if __name__ == "__main__":
    cropped_tomato_data(train_dir, 'train')
    cropped_tomato_data(test_dir, 'test')
