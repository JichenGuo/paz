"""Visualize total and per-head losses from a training CSV.

Example:
    python -m paz.graphics.synthetic_data.visualize_training_losses \
        --experiment experiments/rgbd_cnn_1000_regularized
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_NAMES = (
    "loss",
    "object_translation_loss",
    "object_orientation_6d_loss",
    "object_scale_loss",
    "light_position_loss",
    "light_intensity_loss",
    "shape_loss",
    "material_loss",
)


def load_history(csv_path):
    """Loads numeric columns from a Keras CSVLogger output."""
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No epochs found in {csv_path}")
    if "epoch" not in rows[0]:
        raise ValueError(f"Missing epoch column in {csv_path}")
    available = {
        name for name in rows[0]
        if name == "epoch" or name == "loss" or name == "val_loss"
        or name.endswith("_loss")
    }
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in available
    }


def discover_loss_names(history):
    """Returns loss bases recorded by either standard or geometry training."""
    names = []
    for column in history:
        name = column[4:] if column.startswith("val_") else column
        if name == "loss" or name.endswith("_loss"):
            if name not in names:
                names.append(name)
    ordered = [name for name in LOSS_NAMES if name in names]
    ordered.extend(sorted(set(names).difference(ordered)))
    return ordered


def plot_losses(history, output_path):
    """Saves total and per-head train/validation losses in one figure."""
    epochs = history["epoch"].astype(int) + 1
    selection_name = "val_loss" if "val_loss" in history else "loss"
    best_arg = int(np.nanargmin(history[selection_name]))
    best_epoch = epochs[best_arg]
    loss_names = discover_loss_names(history)
    columns = 2
    rows = max(1, math.ceil(len(loss_names) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(14, 4 * rows), sharex=True, squeeze=False
    )
    for axis, name in zip(axes.flat, loss_names):
        validation_name = f"val_{name}"
        has_curve = False
        if name in history:
            axis.plot(epochs, history[name], label="Training")
            has_curve = True
        if validation_name in history:
            axis.plot(epochs, history[validation_name], label="Validation")
            has_curve = True
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.5,
                     label=f"Best total val: epoch {best_epoch}")
        title = "Total multi-task loss" if name == "loss" else name
        axis.set_title(title.replace("_", " ").title())
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        if has_curve:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5, 0.5, "Not recorded in selected CSV",
                ha="center", va="center", transform=axis.transAxes,
            )
    for axis in axes.flat[len(loss_names):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return best_epoch, history[selection_name][best_arg]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=Path,
        default=Path("experiments/rgbd_cnn_1000_regularized"),
        help="Directory containing training.csv or geometry_warmup.csv.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional explicit CSV path.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG; defaults to <experiment>/loss_from_csv.png.",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    csv_path = args.csv
    if csv_path is None:
        candidates = (
            args.experiment / "training.csv",
            args.experiment / "geometry_warmup.csv",
        )
        csv_path = next((path for path in candidates if path.is_file()), None)
        if csv_path is None:
            expected = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"No training CSV found; checked: {expected}")
    output_path = args.output or args.experiment / "loss_from_csv.png"
    history = load_history(csv_path)
    best_epoch, best_loss = plot_losses(history, output_path)
    missing = [] if csv_path.name == "geometry_warmup.csv" else [
        name for name in LOSS_NAMES[1:]
        if name not in history and f"val_{name}" not in history
    ]
    print(f"Loaded training history from {csv_path}")
    print(f"Saved loss visualization to {output_path}")
    print(f"Best total validation loss: {best_loss:.6f} at epoch {best_epoch}")
    if missing:
        print(
            "Per-head losses were not recorded and cannot be reconstructed: "
            + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
