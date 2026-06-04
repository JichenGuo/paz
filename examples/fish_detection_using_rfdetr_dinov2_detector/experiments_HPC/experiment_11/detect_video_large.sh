#!/bin/bash
###############################################################################
# detect_video.sh
# Run RF-DETR video detection inside Singularity container
###############################################################################

set -e
export RFDETR_JIT_GRAD=0

SIF=~/images/paz_fish_jax.sif

SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/detect_video_large.py

VIDEO=~/videos/Trim_3.mp4

#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversample_epoch100_v2/checkpoint_best_total.weights.h5
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video_3_v2_nanov2.mp4
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_results/detected_video_3_v2_nanov2.json


CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_large_oversample_epoch100/checkpoint_best_total.weights.h5
OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_results/detected_video_3_large_best_total.mp4
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_results/detected_video_3_large_best_total.json

THRESHOLD=0.5

singularity exec --nv \
    --bind /mnt/beegfs:/mnt/beegfs \
    "$SIF" \
    python3 "$SCRIPT" \
        --video "$VIDEO" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --json-output "$JSON_OUTPUT" \
        --threshold "$THRESHOLD"\
        --count-classes fish,crab,lobster


echo "Detection finished."
echo "Output video: $OUTPUT"
echo "JSON results: $JSON_OUTPUT"
