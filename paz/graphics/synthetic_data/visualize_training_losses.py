"""Visualize total and per-head losses from a training CSV.

Example:
    python -m paz.graphics.synthetic_data.visualize_training_losses \
        --experiment experiments/rgbd_cnn_1000_regularized
"""

import argparse
import csv
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
    available = {"epoch"}
    for name in LOSS_NAMES:
        for column in (name, f"val_{name}"):
            if column in rows[0]:
                available.add(column)
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in available
    }


def plot_losses(history, output_path):
    """Saves total and per-head train/validation losses in one figure."""
    epochs = history["epoch"].astype(int) + 1
    selection_name = "val_loss" if "val_loss" in history else "loss"
    best_arg = int(np.nanargmin(history[selection_name]))
    best_epoch = epochs[best_arg]
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    for axis, name in zip(axes.flat, LOSS_NAMES):
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
                0.5, 0.5, "Not recorded in training.csv",
                ha="center", va="center", transform=axis.transAxes,
            )
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
        help="Directory containing training.csv.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG; defaults to <experiment>/loss_from_csv.png.",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    csv_path = args.experiment / "training.csv"
    output_path = args.output or args.experiment / "loss_from_csv.png"
    history = load_history(csv_path)
    best_epoch, best_loss = plot_losses(history, output_path)
    missing = [
        name for name in LOSS_NAMES[1:]
        if name not in history and f"val_{name}" not in history
    ]
    print(f"Saved loss visualization to {output_path}")
    print(f"Best total validation loss: {best_loss:.6f} at epoch {best_epoch}")
    if missing:
        print(
            "Per-head losses were not recorded and cannot be reconstructed: "
            + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
