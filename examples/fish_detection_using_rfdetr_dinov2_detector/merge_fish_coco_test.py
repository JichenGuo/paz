#!/usr/bin/env python3
"""Merge the selected COCO datasets into a single collision-safe test set."""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "Combined_Fish_Crab_Lobster_coco" / "test"
SOURCES = [
    ("find_train", ROOT / "datasets" / "Find fish- crab and more.coco" / "train"),
    ("find_valid", ROOT / "datasets" / "Find fish- crab and more.coco" / "valid"),
    ("find_test", ROOT / "datasets" / "Find fish- crab and more.coco" / "test"),
    (
        "labelimage_test",
        ROOT / "datasets" / "Labelimage_Fish_coco_split_70_20_10" / "test",
    ),
]
CATEGORY_IDS = {"crab": 1, "fish": 2, "lobster": 3}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    merged = {
        "info": {"description": "Combined fish, crab, and lobster COCO test dataset"},
        "licenses": [],
        "categories": [
            {"id": category_id, "name": name, "supercategory": "animal"}
            for name, category_id in CATEGORY_IDS.items()
        ],
        "images": [],
        "annotations": [],
    }
    next_image_id = 1
    next_annotation_id = 1

    for source_name, source_dir in SOURCES:
        annotation_path = source_dir / "_annotations.coco.json"
        with annotation_path.open(encoding="utf-8") as stream:
            source = json.load(stream)

        source_categories = {
            category["id"]: category["name"] for category in source["categories"]
        }
        image_id_map = {}

        for image in source["images"]:
            source_image = source_dir / image["file_name"]
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing source image: {source_image}")

            output_name = f"{source_name}__{Path(image['file_name']).name}"
            if (OUTPUT / output_name).exists():
                raise FileExistsError(f"Output filename collision: {output_name}")
            shutil.copy2(source_image, OUTPUT / output_name)

            new_image = dict(image)
            image_id_map[image["id"]] = next_image_id
            new_image["id"] = next_image_id
            new_image["file_name"] = output_name
            merged["images"].append(new_image)
            next_image_id += 1

        for annotation in source["annotations"]:
            category_name = source_categories[annotation["category_id"]]
            if category_name not in CATEGORY_IDS:
                raise ValueError(
                    f"Annotation uses unsupported category {category_name!r} "
                    f"in {annotation_path}"
                )
            if annotation["image_id"] not in image_id_map:
                raise ValueError(
                    f"Annotation references unknown image ID {annotation['image_id']} "
                    f"in {annotation_path}"
                )

            new_annotation = dict(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[annotation["image_id"]]
            new_annotation["category_id"] = CATEGORY_IDS[category_name]
            merged["annotations"].append(new_annotation)
            next_annotation_id += 1

    with (OUTPUT / "_annotations.coco.json").open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2)
        stream.write("\n")

    print(f"Created {OUTPUT}")
    print(f"Images: {len(merged['images'])}")
    print(f"Annotations: {len(merged['annotations'])}")


if __name__ == "__main__":
    main()
