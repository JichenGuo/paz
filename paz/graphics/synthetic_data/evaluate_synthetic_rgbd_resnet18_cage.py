"""Evaluate and visualize held-out ResNet-18 cage predictions.

The evaluator loads the best joint checkpoint, predicts geometry and physical
parameters from each RGB-D test sample, renders the connected cage-deformed
mesh with the PAZ triangle renderer, and saves quantitative and visual
comparisons against the original RGB-D observation.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.evaluate_synthetic_rgbd_resnet18_cage \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/resnet18_rgbd_cage_joint
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from .render_inverse_graphics_cage_predictions import main


if __name__ == "__main__":
    main()
