#!/bin/bash
###############################################################################
# detect_video.sh
# Run RF-DETR video detection inside Singularity container
###############################################################################

set -e
export RFDETR_JIT_GRAD=0

SIF=~/images/paz_fish_jax.sif

SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/detect_video.py

VIDEO=~/videos/Trim_1.mp4

CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/checkpoint_best_total.weights.h5

OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video_1.mp4
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video_1.json

THRESHOLD=0.5

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
