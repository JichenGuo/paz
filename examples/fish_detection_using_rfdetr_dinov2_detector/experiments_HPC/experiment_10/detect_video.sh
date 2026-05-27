#!/bin/bash
###############################################################################
# detect_video.sh
# Run RF-DETR video detection inside Singularity container
###############################################################################

set -e

SIF=~/images/paz_fish_jax.sif

SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/detect_video.py

VIDEO=~/datasets/input_video.mp4

CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_runs/crab_lobster_oversampled/rfdetr_nano_finetuned_final.weights.h5

OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video.mp4
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video.json

THRESHOLD=0.8

singularity exec --nv \
    --bind /mnt/beegfs:/mnt/beegfs \
    "$SIF" \
    python3 "$SCRIPT" \
        --video "$VIDEO" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --json-output "$JSON_OUTPUT" \
        --threshold "$THRESHOLD"

echo "Detection finished."
echo "Output video: $OUTPUT"
echo "JSON results: $JSON_OUTPUT"
