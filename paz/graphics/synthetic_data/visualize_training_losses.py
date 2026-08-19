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
    required = {"epoch"}
    for name in LOSS_NAMES:
        required.update((name, f"val_{name}"))
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in required
    }


def plot_losses(history, output_path):
    """Saves total and per-head train/validation losses in one figure."""
    epochs = history["epoch"].astype(int) + 1
    best_arg = int(np.nanargmin(history["val_loss"]))
    best_epoch = epochs[best_arg]
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    for axis, name in zip(axes.flat, LOSS_NAMES):
        validation_name = f"val_{name}"
        axis.plot(epochs, history[name], label="Training")
        axis.plot(epochs, history[validation_name], label="Validation")
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.5,
                     label=f"Best total val: epoch {best_epoch}")
        title = "Total multi-task loss" if name == "loss" else name
        axis.set_title(title.replace("_", " ").title())
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return best_epoch, history["val_loss"][best_arg]


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
    print(f"Saved loss visualization to {output_path}")
    print(f"Best total validation loss: {best_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
