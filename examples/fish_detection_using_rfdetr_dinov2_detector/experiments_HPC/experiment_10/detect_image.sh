#!/bin/bash
###############################################################################
# detect_image.sh
# Run RF-DETR image detection inside Singularity container
###############################################################################

# Exit immediately if a command fails
set -e

# -------------------------------
# Paths
# -------------------------------
SIF=~/images/paz_fish_jax.sif

SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/detect_image.py

IMAGE=~/datasets/Labelimage_Fish_coco/train/5500_png.rf.61ccZBWaiM43G5pmYXFr.png
#IMAGE=/mnt/beegfs/home/jguo/datasets/Deepfish/7117/valid/

#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/checkpoints/rfdetr_nano_best.weights.h5
CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_runs/from_experiment_10/rfdetr_nano_finetuned_final.weights.h5
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/valid_detections_7117
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/valid_detections_7117/detections.json
OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_image.png
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detections.json

THRESHOLD=0.8

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