#!/bin/bash
###############################################################################
# detect_image.sh
# Run RF-DETR image detection inside Singularity container
###############################################################################

# Exit immediately if a command fails
set -e
export RFDETR_JIT_GRAD=0

# -------------------------------
# Paths
# -------------------------------
SIF=~/images/paz_fish_jax.sif

SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/detect_image_large.py

#IMAGE=~/datasets/Labelimage_Fish_coco/train/0128_png.rf.IrJ0xuUvIvyEKXAiv8af.png
#IMAGE=/mnt/beegfs/home/jguo/datasets/Deepfish/7117/valid/
IMAGE=/mnt/beegfs/home/jguo/datasets/Labelimage_Fish_coco_split_70_20_10/test
#IMAGE=/mnt/beegfs/home/jguo/datasets/fathom_test_small/



#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/checkpoints/rfdetr_nano_best.weights.h5
#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_runs/from_experiment_10/checkpoint_best_total.weights.h5
#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_runs/crab_lobster_oversampled/rfdetr_nano_finetuned_final.weights.h5
#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_large_oversample_epoch100/checkpoint_best_total.weights.h5
CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_large_fathomnet/checkpoint_best_total.weights.h5

#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/valid_detections_7117
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/valid_detections_7117/detections.json
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/oversampled_valid_deepfish
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/oversampled_detections_valid_deepfish.json
OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_results/fathomnet_epoch10_reefshield_test
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_results/fathomnet_epoch10_reefshield_test.json


THRESHOLD=0.5

# -------------------------------
# Run detection
# -------------------------------
singularity exec --nv \
    --bind /mnt/beegfs:/mnt/beegfs \
    "$SIF" \
    python3 "$SCRIPT" \
        --image "$IMAGE" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --json-output "$JSON_OUTPUT" \
        --threshold "$THRESHOLD"

echo "Detection finished."
echo "Output image: $OUTPUT"
echo "JSON results: $JSON_OUTPUT"