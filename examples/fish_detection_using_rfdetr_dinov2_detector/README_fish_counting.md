# Fish detection, tracking, and grid counting

Run [track_count_deepsort_grid.sh](track_count_deepsort_grid.sh) to detect fish,
crabs, and lobsters in a video using RF-DETR, track detections with DeepSORT,
and save a video with a 2×2 counting grid. The launcher runs
[track_count_deepsort_grid.py](track_count_deepsort_grid.py) inside a
GPU-enabled Singularity container.

## Requirements

- This PAZ repository, including the RF-DETR implementation and example scripts.
- Singularity and an NVIDIA GPU accessible through `singularity exec --nv`.
- The `paz_fish_jax.sif` container, or an equivalent environment containing the
  project's Keras/JAX detector dependencies, OpenCV, NumPy, Pillow, and
  `deep-sort-realtime`. The default DeepSORT MobileNet appearance embedder also
  requires its PyTorch/torchvision dependencies and model weights.
- An input video and compatible RF-DETR checkpoint.

## Download the container

Download the already-built `paz_fish_jax.sif` image from the
[DFKI fish detection environment folder](https://cloud.dfki.de/owncloud/index.php/apps/files/files/123412959?dir=/fish_detection_envs).
Open the folder in a browser and sign in if prompted. No container build is
required.

Save the image to `~/images/paz_fish_jax.sif`, as expected by the launcher, or
set `SIF` in the launcher to its downloaded location. You still need Singularity,
a compatible NVIDIA GPU and host driver for `--nv`, and this PAZ repository:
the launcher loads the Python script from the checkout outside the container.
Download the checkpoint separately as described below.

## Download the checkpoint

Download the checkpoint weights from the provided
[DFKI cloud folder](https://cloud.dfki.de/owncloud/index.php/apps/files/files/123412945?dir=/fish_detection_weights).
Open this folder in a browser and sign in if prompted. Set `CHECKPOINT` in the
launcher to the downloaded `.weights.h5` file.

Keep the matching `finetune_config.json` beside the checkpoint when available.
The Python script uses that file to resolve detector class names; without it,
the fallback is a single `fish` class. For a multiclass checkpoint without its
configuration, add `--class-names` to the launch command using the exact class
order used during training. `--count-classes` selects which classes to count;
it does not define the detector's class order.

The active launcher selects `DETECTOR="LARGE"`; use weights trained for that
architecture. Select `NANO` only when using a matching Nano checkpoint.

## Configure the launcher

Edit the active assignments in `track_count_deepsort_grid.sh` before running.
Its current paths refer to a specific HPC installation and must be adapted to
your machine.

| Variable | Purpose |
| --- | --- |
| `SIF` | Path to the Singularity container image. |
| `SCRIPT` | Absolute path to `track_count_deepsort_grid.py` in this checkout. |
| `VIDEO` | Input video path. |
| `CHECKPOINT` | Downloaded `.weights.h5` checkpoint path. |
| `OUTPUT` | Annotated output video path, normally ending in `.mp4`. |
| `JSON_OUTPUT` | Tracking and counting JSON output path. |
| `DETECTOR` | Detector architecture; currently `LARGE`. |

For example, replace the corresponding assignments with your own paths:

```bash
SIF="$HOME/images/paz_fish_jax.sif"
SCRIPT="/absolute/path/to/paz/examples/fish_detection_using_rfdetr_dinov2_detector/track_count_deepsort_grid.py"
VIDEO="$HOME/videos/input.mp4"
CHECKPOINT="/absolute/path/to/weights/checkpoint_best_total.weights.h5"
OUTPUT="$HOME/videos/results/input_grid.mp4"
JSON_OUTPUT="$HOME/videos/results/input_grid.json"
```

The launcher binds `/mnt/beegfs` into the container with
`--bind /mnt/beegfs:/mnt/beegfs`. Adapt this bind to your storage location if
needed, ensuring that the repository, checkpoint, input video, and output
directory are accessible inside the container at the configured paths.

Example videos can be downloaded from [DFKI cloud video folder](https://cloud.dfki.de/owncloud/index.php/apps/files/files/123413025?dir=/Reefshield_videos)

## Run

From the repository root:

```bash
bash examples/fish_detection_using_rfdetr_dinov2_detector/track_count_deepsort_grid.sh
```

The launcher uses these settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `THRESHOLD` | `0.5` | Detection confidence threshold. |
| `TRACK_FRAMES` | `3` | Minimum observed track length for cumulative counting. |
| `MAX_AGE` | `3` | Maximum missed-frame age before DeepSORT removes a track. |
| `N_INIT` | `3` | Detections needed before DeepSORT confirms a track. |
| `DRAW_MISS_FRAMES` | `1` | Maximum missed frames for drawing a confirmed track. |
| `--count-classes` | `crab,fish,lobster` | Classes included in tracking and counting. |

For fish-only counting, change the final argument to `--count-classes fish`.
To process a short sample, add `--max-frames 100` to the Python command inside
the launcher. Add `--start-frame N` to begin at a particular frame index.
Preserve the trailing backslashes between continued command lines.

### Frame downsampling and processing time

Add `--frame-stride 5` to the Python command in the launcher to process every
10th frame. The default is `1` (all frames). Sampling starts at `--start-frame`:
for example, a start frame of 5 and stride of 10 process frames 5, 15, 25, etc.
Skipped frames are not passed to the detector or tracker. `--max-frames` limits
the number of processed frames. The output video uses input FPS divided by the
stride to preserve playback speed, with fewer frames and unchanged resolution.
DeepSORT age, confirmation, drawing, and minimum-track-length settings are
measured in processed frames; larger strides can make tracking harder when
objects move significantly between observations.

Every processed frame prints its source frame index and pipeline processing
time in milliseconds. When `--json-output` is supplied (as in the launcher),
each frame also records `pipeline_inference_time_ms`, `frame_index`, and
`source_time_seconds`. The timing includes preprocessing, detection, tracking,
counting, and drawing; it excludes video decoding and writing. The first frame
can include model warm-up or compilation overhead. JSON metadata records
`frame_stride`, input `fps`, and `output_fps`, along with aggregate timing.

## Read the output

The output video contains:

- Confirmed DeepSORT tracks with class labels and IDs once eligible for
  cumulative counting. Predicted boxes may remain visible for up to
  `DRAW_MISS_FRAMES` missed frames.
- A 2×2 grid. Each cell shows its current-frame detection total and per-class
  counts. A detection belongs to the cell containing its bounding-box center;
  centers exactly on a divider belong to the right or bottom cell.
- A bottom-right summary box showing **Current frame total** and per-class
  detection counts across the entire frame.

Displayed detection counts reset every frame and include selected-class
detections before track confirmation. Consequently, a detection can contribute
to a displayed count before its tracked box appears. Track confirmation and
minimum track length do not delay these current-frame counts.

The JSON file contains frame-level `detections`, `tracks`, and `grid_counts`
under `top_left`, `top_right`, `bottom_left`, and `bottom_right`. Sum a class's
four cell counts to obtain its current-frame total.

The JSON `counts` and `final_counts`, console counts, and displayed track IDs
still use cumulative track counting. Each confirmed track is counted once after
meeting `TRACK_FRAMES`. These totals count qualifying tracks; an object that
loses its identity and receives a new track can be counted again.

The JSON also includes processing metadata and pipeline timing. Timing covers
preprocessing, detection, tracking, counting, and annotation, and excludes
video decoding and writing.

## Troubleshooting

- **Paths are missing inside the container:** check the active launcher paths
  and Singularity bind mounts.
- **Checkpoint shape mismatch or incorrect labels:** verify `DETECTOR` and the
  checkpoint's training class names and order.
- **Missing DeepSORT module:** install `deep-sort-realtime` in the environment
  used inside the container, together with the default embedder dependencies.
- **Video cannot be written:** check output-directory permissions and available
  OpenCV codecs. The Python script defaults to `mp4v`; use `--fourcc` in the
  launch command to select another supported codec.
