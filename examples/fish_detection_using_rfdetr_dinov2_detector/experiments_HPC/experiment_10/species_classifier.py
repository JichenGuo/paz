"""Crop-based FCN species classifier utilities for FathomNet-style COCO data."""

import json
import math
import random
from collections import Counter
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco_annotations(annotation_path):
    annotation_path = Path(annotation_path).expanduser().resolve()
    with annotation_path.open() as f:
        return json.load(f)


def read_class_names(coco):
    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    categories = sorted(categories, key=lambda category: category["id"])
    if not categories:
        raise ValueError("No usable categories found in annotation JSON")
    return [category["name"] for category in categories]


def build_category_mapping(coco):
    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    categories = sorted(categories, key=lambda category: category["id"])
    return {category["id"]: index for index, category in enumerate(categories)}


def find_image_path(image_root, file_name):
    image_root = Path(image_root)
    image_path = image_root / file_name
    if image_path.exists():
        return image_path

    stem = Path(file_name).stem
    matches = [
        path
        for suffix in IMAGE_EXTENSIONS
        for path in image_root.glob(f"{stem}{suffix}")
        if path.exists()
    ]
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Image referenced by annotations is missing: {image_path}")


def build_crop_records(coco, image_root, min_box_size=4, include_classes=None):
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    category_to_original_index = build_category_mapping(coco)
    all_class_names = read_class_names(coco)
    original_to_training_index = {
        index: index for index in range(len(all_class_names))
    }
    class_names = list(all_class_names)
    include_indices = None
    if include_classes:
        requested = {name.strip() for name in include_classes if name.strip()}
        missing = sorted(requested - set(all_class_names))
        if missing:
            raise ValueError(
                f"Classes not found in annotations: {missing}. "
                f"Available: {sorted(all_class_names)}"
            )
        include_indices = {
            index for index, name in enumerate(all_class_names) if name in requested
        }
        kept_indices = [
            index for index, name in enumerate(all_class_names) if name in requested
        ]
        class_names = [all_class_names[index] for index in kept_indices]
        original_to_training_index = {
            original_index: training_index
            for training_index, original_index in enumerate(kept_indices)
        }

    records = []
    skipped = 0
    for annotation in coco.get("annotations", []):
        category_id = annotation.get("category_id")
        if category_id not in category_to_original_index:
            skipped += 1
            continue

        original_label = category_to_original_index[category_id]
        if include_indices is not None and original_label not in include_indices:
            continue
        label = original_to_training_index[original_label]

        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            skipped += 1
            continue

        x, y, w, h = [float(value) for value in annotation.get("bbox", [])[:4]]
        if w < min_box_size or h < min_box_size:
            skipped += 1
            continue

        image_path = find_image_path(image_root, image["file_name"])
        records.append(
            {
                "image": str(image_path),
                "bbox_xywh": [x, y, w, h],
                "label": label,
                "class_name": class_names[label],
                "original_label": original_label,
                "annotation_id": annotation.get("id"),
                "image_id": image.get("id"),
            }
        )

    if not records:
        raise ValueError("No crop records were built from the annotation JSON")
    return records, class_names, skipped


def split_records(records, val_fraction=0.2, seed=42):
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    rng = random.Random(seed)
    train_records = []
    val_records = []
    for label_records in by_label.values():
        label_records = list(label_records)
        rng.shuffle(label_records)
        if len(label_records) == 1:
            train_records.extend(label_records)
            continue
        val_count = max(1, int(round(len(label_records) * val_fraction)))
        val_count = min(val_count, len(label_records) - 1)
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    if not val_records:
        raise ValueError("Validation split is empty; reduce --val-fraction")
    return train_records, val_records


def crop_with_padding(image, bbox_xywh, padding=0.12):
    width, height = image.size
    x, y, w, h = bbox_xywh
    pad_x = w * padding
    pad_y = h * padding
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(width, int(math.ceil(x + w + pad_x)))
    bottom = min(height, int(math.ceil(y + h + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box after clipping: {bbox_xywh}")
    return image.crop((left, top, right, bottom))


def resize_letterbox(image, image_size, fill=(0, 0, 0)):
    image = ImageOps.contain(image, (image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def augment_crop(image, rng, jitter=0.2, hflip_prob=0.5):
    if rng.random() < hflip_prob:
        image = ImageOps.mirror(image)
    if jitter > 0:
        for enhancer_cls in (
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
        ):
            factor = 1.0 + rng.uniform(-jitter, jitter)
            image = enhancer_cls(image).enhance(max(0.1, factor))
    return image


class FathomNetCropDataset(keras.utils.PyDataset):
    """Keras PyDataset that returns annotation crops and species labels."""

    def __init__(
        self,
        records,
        image_size=224,
        batch_size=32,
        shuffle=False,
        augment=False,
        crop_padding=0.12,
        seed=42,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.records = list(records)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.crop_padding = float(crop_padding)
        self.seed = int(seed)
        self.class_weights = class_weights
        self.rng = random.Random(seed)
        self.indices = list(range(len(self.records)))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images = []
        labels = []
        weights = []

        for record_index in batch_indices:
            record = self.records[record_index]
            image = Image.open(record["image"]).convert("RGB")
            crop = crop_with_padding(image, record["bbox_xywh"], self.crop_padding)
            if self.augment:
                crop = augment_crop(crop, self.rng)
            crop = resize_letterbox(crop, self.image_size)
            images.append(np.asarray(crop, dtype="float32"))
            labels.append(record["label"])
            if self.class_weights is not None:
                weights.append(float(self.class_weights.get(record["label"], 1.0)))

        x = np.stack(images, axis=0)
        y = np.asarray(labels, dtype="int32")
        if self.class_weights is None:
            return x, y
        return x, y, np.asarray(weights, dtype="float32")

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def compute_balanced_class_weights(records, num_classes):
    counts = Counter(record["label"] for record in records)
    total = sum(counts.values())
    weights = {}
    for label in range(num_classes):
        if counts[label] == 0:
            weights[label] = 0.0
        else:
            weights[label] = total / (num_classes * counts[label])
    return weights


def build_fcn_classifier(input_shape, num_classes, width=32, dropout=0.25):
    """Build a compact fully-convolutional image classifier."""

    inputs = keras.layers.Input(shape=input_shape, name="crop")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)

    for block_index, filters in enumerate((width, width * 2, width * 4, width * 4)):
        x = keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"block{block_index + 1}_conv1",
        )(x)
        x = keras.layers.BatchNormalization(name=f"block{block_index + 1}_bn1")(x)
        x = keras.layers.Activation("relu", name=f"block{block_index + 1}_relu1")(x)
        x = keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"block{block_index + 1}_conv2",
        )(x)
        x = keras.layers.BatchNormalization(name=f"block{block_index + 1}_bn2")(x)
        x = keras.layers.Activation("relu", name=f"block{block_index + 1}_relu2")(x)
        x = keras.layers.MaxPooling2D(pool_size=2, name=f"block{block_index + 1}_pool")(x)

    x = keras.layers.Conv2D(
        width * 8,
        1,
        padding="same",
        activation="relu",
        name="fcn_projection",
    )(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.Conv2D(num_classes, 1, padding="same", name="logits_map")(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_logits")(x)
    outputs = keras.layers.Activation("softmax", name="species")(x)
    return keras.Model(inputs, outputs, name="FathomNet_FCN_species_classifier")
