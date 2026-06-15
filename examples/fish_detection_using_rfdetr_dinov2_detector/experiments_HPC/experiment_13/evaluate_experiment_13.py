#!/usr/bin/env python
"""Evaluate Experiment 13 RF-DETR Nano checkpoints.

This script evaluates checkpoints produced by ``experiment_13.py`` with the
same validation metric used in ``experiment_13_2.py``.  The evaluation dataset
is DeepFish ``valid`` plus FathomNet validation data, with all labels remapped
to the single ``sea_animal`` class.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
_EXAMPLE_ROOT = _SCRIPT_DIR.parents[1]
_SRC_DIR = _EXAMPLE_ROOT / "src"
_EXP_13_2_DIR = _SCRIPT_DIR.parent / "experiment_13_2"

for path in (_PAZ_ROOT, _SRC_DIR, _EXP_13_2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)
from train_utils import setup_logging, validate_epoch_full

from experiment_13_2 import CLASS_NAME, _prepare_merged_coco

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    _SCRIPT_DIR / "checkpoints" / "rfdetr_nano_sea_animal_final.weights.h5"
)
DEFAULT_DEEPFISH_DIR = Path("/mnt/beegfs/home/jguo/datasets/Deepfish")
DEFAULT_FATHOMNET_DIR = Path("/mnt/beegfs/home/jguo/datasets/fathomnet")
DEFAULT_FATHOMNET_VALID_DIR = Path(
    "/mnt/beegfs/home/jguo/datasets/images_fathom_test"
)


class _TeeWriter:
    def __init__(self, original, log_path):
        self._original = original
        self._log_file = open(log_path, "a")

    def write(self, text):
        self._original.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _path_from_env(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return Path(default).expanduser().resolve()
    return Path(value).expanduser().resolve()


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Experiment 13 RF-DETR Nano checkpoint(s) with the same "
            "validate_epoch_full metrics used by Experiment 13_2."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint to evaluate when --all-checkpoints is not set.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every *.weights.h5 file in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_SCRIPT_DIR / "checkpoints",
        help="Directory scanned by --all-checkpoints.",
    )
    parser.add_argument(
        "--deepfish-dir",
        type=Path,
        default=_path_from_env("DEEPFISH_DIR", DEFAULT_DEEPFISH_DIR),
    )
    parser.add_argument(
        "--fathomnet-dir",
        type=Path,
        default=_path_from_env("FATHOMNET_DIR", DEFAULT_FATHOMNET_DIR),
        help="FathomNet train dataset directory, needed to rebuild merged COCO.",
    )
    parser.add_argument(
        "--fathomnet-valid-dir",
        type=Path,
        default=_path_from_env("FATHOMNET_VALID_DIR", DEFAULT_FATHOMNET_VALID_DIR),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "evaluation_results",
        help="Directory for JSON, CSV, and log files.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=int(os.environ.get("RFDETR_EVAL_BATCH_SIZE", "16")),
    )
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-batches",
        type=_nonnegative_int,
        default=0,
        help="Debug limit. 0 means evaluate all batches.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(int(os.environ.get("RFDETR_REBUILD_DATASET", "0"))),
        help="Rebuild the merged validation COCO dataset.",
    )
    return parser.parse_args()


def collect_checkpoints(args):
    if args.all_checkpoints:
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        checkpoints = sorted(checkpoint_dir.glob("*.weights.h5"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No *.weights.h5 checkpoints found in {checkpoint_dir}"
            )
        return checkpoints
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train experiment_13.py first or pass --checkpoint /path/to/file.weights.h5"
        )
    return [checkpoint]


def build_model(checkpoint_path):
    logger.info("Creating RFDETRNano (num_classes=1) ...")
    detector = RFDETRNano(num_classes=1)
    dummy = np.ones((1, 384, 384, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.class_names = [CLASS_NAME]
    logger.info("Loading checkpoint: %s", checkpoint_path)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    return detector


def make_train_config(coco_dir, output_dir, batch_size):
    return TrainConfig(
        dataset_dir=str(coco_dir),
        dataset_file="coco_json",
        output_dir=str(output_dir),
        epochs=1,
        batch_size=batch_size,
        grad_accum_steps=1,
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        lr_scheduler="cosine",
        lr_min_factor=0.0,
        warmup_epochs=0.0,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        checkpoint_interval=10,
        early_stopping=False,
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=0,
        run_test=False,
        class_names=[CLASS_NAME],
    )


def write_csv(summary_rows, csv_path):
    metric_keys = [
        "val_loss",
        "val_loss_ce",
        "val_loss_bbox",
        "val_loss_giou",
        "val_mAP_50",
        "val_mAP_50_95",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_accuracy",
        "val_num_gt_boxes",
        "val_num_pred_boxes",
        "elapsed_seconds",
    ]
    fieldnames = ["checkpoint", "checkpoint_path"] + metric_keys
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(args.output_dir))
    sys.stdout = _TeeWriter(sys.stdout, args.output_dir / "output.txt")

    import keras

    logger.info("=" * 68)
    logger.info("EXPERIMENT 13 EVALUATION")
    logger.info("=" * 68)
    logger.info("Keras backend: %s", keras.backend.backend())
    logger.info("DeepFish dir         : %s", args.deepfish_dir)
    logger.info("FathomNet train dir  : %s", args.fathomnet_dir)
    logger.info("FathomNet valid dir  : %s", args.fathomnet_valid_dir)
    logger.info("Output dir           : %s", args.output_dir)

    checkpoints = collect_checkpoints(args)
    coco_dir, _, val_ds = _prepare_merged_coco(
        args.deepfish_dir.expanduser().resolve(),
        args.fathomnet_dir.expanduser().resolve(),
        args.fathomnet_valid_dir.expanduser().resolve(),
        _SCRIPT_DIR,
        rebuild=args.rebuild_dataset,
    )
    val_ds.resolution = 384
    val_indices = list(range(len(val_ds)))
    max_batches = args.max_batches if args.max_batches > 0 else None

    logger.info("Validation dataset   : %d images -> %s", len(val_indices), coco_dir)
    logger.info("Checkpoints          : %d", len(checkpoints))

    summary_rows = []
    results = {
        "experiment": "experiment_13",
        "metric_source": "experiment_13_2.validate_epoch_full",
        "class_names": [CLASS_NAME],
        "validation_dataset": {
            "coco_dir": str(coco_dir),
            "num_images": len(val_indices),
            "deepfish_dir": str(args.deepfish_dir),
            "fathomnet_valid_dir": str(args.fathomnet_valid_dir),
        },
        "settings": {
            "batch_size": args.batch_size,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "max_batches": max_batches,
        },
        "checkpoints": [],
    }

    for checkpoint in checkpoints:
        logger.info("")
        logger.info("-" * 68)
        logger.info("Evaluating %s", checkpoint.name)
        start = time.time()
        detector = build_model(checkpoint)
        train_config = make_train_config(coco_dir, args.output_dir, args.batch_size)
        criterion, _ = build_criterion_from_config(
            detector.model_config, train_config
        )
        metrics = validate_epoch_full(
            model=detector.model.model,
            criterion=criterion,
            dataset=val_ds,
            indices=val_indices,
            batch_size=args.batch_size,
            num_classes=1,
            class_names=[CLASS_NAME],
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            max_batches=max_batches,
            logger=logger,
            prefix="val",
        )
        elapsed = time.time() - start
        logger.info("Completed %s in %.1fs", checkpoint.name, elapsed)
        logger.info(
            "mAP@50=%.4f mAP@50:95=%.4f precision=%.4f recall=%.4f f1=%.4f loss=%.4f",
            metrics["val_mAP_50"],
            metrics["val_mAP_50_95"],
            metrics["val_precision"],
            metrics["val_recall"],
            metrics["val_f1"],
            metrics["val_loss"],
        )

        checkpoint_result = {
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "elapsed_seconds": elapsed,
            "metrics": _jsonable(metrics),
        }
        results["checkpoints"].append(checkpoint_result)

        row = {
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "elapsed_seconds": elapsed,
        }
        row.update({
            key: _jsonable(value)
            for key, value in metrics.items()
            if not key.startswith("per_class_")
        })
        summary_rows.append(row)

    json_path = args.output_dir / "experiment_13_evaluation_results.json"
    csv_path = args.output_dir / "experiment_13_evaluation_summary.csv"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(summary_rows, csv_path)

    logger.info("")
    logger.info("=" * 68)
    logger.info("Evaluation results saved:")
    logger.info("  JSON: %s", json_path)
    logger.info("  CSV : %s", csv_path)
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
