#!/bin/bash
###############################################################################
#track_count_deepsort_grid.sh
# Run RF-DETR video detection inside Singularity container
###############################################################################

set -e
export RFDETR_JIT_GRAD=0

SIF=~/images/paz_fish_jax.sif

#SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/track_count_deepsort.py
SCRIPT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/track_count_deepsort_grid.py

VIDEO=~/videos/cut26.mp4
#VIDEO=~/videos/Trim_3_short.mp4


#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/checkpoint_best_total.weights.h5
#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_11/finetune_large_oversample_epoch100/checkpoint_best_total.weights.h5
CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/finetune_large_merged_oversampled_epoch100/checkpoint_best_total.weights.h5
#CHECKPOINT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/new_finetune_large_merged_oversampled_epoch100/rfdetr_large_finetuned_final.weights.h5

#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/track_results/tracked_video_3_short_trial.mp4
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/track_results/tracked_video_3_short_trial.json
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/track_results/tracked_video_3_exp10_oversampled_th05_v3.mp4
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_10/finetune_oversampled_epoch100/track_results/tracked_video_3_exp10_oversampled_th05_v3.json
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/finetune_large_merged_oversampled_epoch100/track_results/tracked_trim_3_short_v3.mp4
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/finetune_large_merged_oversampled_epoch100/track_results/tracked_trim_3_short_v3.json
#OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/new_finetune_large_merged_oversampled_epoch100/grid_results/new_tracked_cut26_grid.mp4
#JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/new_finetune_large_merged_oversampled_epoch100/grid_results/new_tracked_cut26_grid.json

OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/finetune_large_merged_oversampled_epoch100/grid_results/tracked_cut26_grid.mp4
JSON_OUTPUT=/mnt/beegfs/home/jguo/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments_HPC/experiment_15/finetune_large_merged_oversampled_epoch100/grid_results/tracked_cut26_grid.json
    
    
THRESHOLD=0.5
TRACK_FRAMES=3
MAX_AGE=3
N_INIT=3
DRAW_MISS_FRAMES=1
FRAME_STRIDE=5
DETECTOR="LARGE"


singularity exec --nv \
    --bind /mnt/beegfs:/mnt/beegfs \
    "$SIF" \
    python3 "$SCRIPT" \
        --video "$VIDEO" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --json-output "$JSON_OUTPUT" \
        --threshold "$THRESHOLD"\
        --detector "$DETECTOR"\
        --min-track-frames "$TRACK_FRAMES"\
        --max-age "$MAX_AGE"\
        --n-init "$N_INIT"\
        --max-draw-missed-frames "$DRAW_MISS_FRAMES"\
        --frame-stride "$FRAME_STRIDE"\
        --count-classes crab,fish,lobster


echo "Detection & Tracking finished."
echo "Output video: $OUTPUT"
echo "JSON results: $JSON_OUTPUT"
