"""Train a joint RGB-D ResNet-18 cage inverse-graphics model.

This replaces the compact encoder with a standard ResNet-18 while reusing the
category-independent cage decoder, physical heads, differentiable renderer,
losses, metrics, and evaluation artifacts. Joint optimization starts at the
first epoch; geometry warm-up is disabled.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_cage \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --output experiments/resnet18_rgbd_cage_joint \
      --epochs 100 --batch-size 8
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import sys

import keras

from .train_synthetic_rgb_resnet18 import basic_block
from . import train_synthetic_rgbd_inverse_graphics_cage as cage


def resnet18_encoder(rgb, depth, object_mask, regularizer, base_width):
    masked_rgb = keras.layers.Multiply(
        name="resnet18_masked_rgb"
    )([rgb, object_mask])
    masked_depth = keras.layers.Multiply(
        name="resnet18_masked_depth"
    )([depth, object_mask])
    inputs = [rgb, depth, object_mask, masked_rgb, masked_depth]
    features = keras.layers.Concatenate(
        name="resnet18_rgbd_concatenation"
    )(inputs)
    width = base_width * 4
    features = keras.layers.Conv2D(
        width, 7, strides=2, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name="cage_resnet18_stem_conv",
    )(features)
    features = keras.layers.BatchNormalization(
        name="cage_resnet18_stem_bn"
    )(features)
    features = keras.layers.Activation(
        "relu", name="cage_resnet18_stem_relu"
    )(features)
    features = keras.layers.MaxPooling2D(
        3, strides=2, padding="same", name="cage_resnet18_stem_pool"
    )(features)
    for stage, filters in enumerate((width, width * 2, width * 4, width * 8)):
        for block in range(2):
            stride = 2 if stage > 0 and block == 0 else 1
            name = f"cage_resnet18_stage{stage + 1}_block{block + 1}"
            features = basic_block(
                features, filters, stride, regularizer, name
            )
    spatial_features = features
    global_features = keras.layers.GlobalAveragePooling2D(
        name="cage_resnet18_global_feature"
    )(spatial_features)
    return global_features, spatial_features


def joint_arguments(argv):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--geometry-only" in arguments:
        raise ValueError("ResNet-18 cage training is joint-only")
    arguments.extend(["--geometry-warmup-epochs", "0"])
    return arguments


def main(argv=None):
    cage.main(
        joint_arguments(argv), encoder_builder=resnet18_encoder,
        encoder_name="RGB-D ResNet-18", model_name="resnet18_rgbd_cage_mesh",
    )


if __name__ == "__main__":
    main()
