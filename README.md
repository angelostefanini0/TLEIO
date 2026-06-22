# TLEIO: Tight Learned Events-Inertial Odometry

TLEIO is a tight learned event-inertial odometry pipeline for estimating camera motion from event streams and IMU measurements. It combines EventsFormer, a transformer-based learned front-end for short-window event-camera motion regression, with a stochastic-cloning EKF that tightly fuses learned relative-motion constraints and high-rate inertial propagation.


## Demo Video

TLEIO in action: learned event-camera motion constraints are fused with IMU measurements to produce accurate odometry trajectories.

https://github.com/user-attachments/assets/04fa2e40-7d03-445a-8c5b-53c3609a9f07


## Method Overview



The full pipeline transforms asynchronous event streams into voxel clips, estimates short-window camera displacements with EventsFormer, and tightly fuses the learned constraints with high-rate IMU propagation in a stochastic-cloning EKF.

![TLEIO model architecture](figures/method/model_architecture.png)

EventsFormer is the learned event front-end of TLEIO. It processes precomputed event voxel clips and predicts relative translation constraints, optionally with uncertainty estimates, for the filter back-end.

![TLEIO tokenizer](figures/method/tokenizer.png)

The tokenizer converts each event voxel clip into spatio-temporal patch tokens suitable for transformer processing.

![EventsFormer encoder](figures/method/eventsformer_encoder.png)

The EventsFormer encoder applies divided space-time attention to capture spatial event structure and temporal motion cues before the prediction head regresses consecutive relative motions.
## Repository Layout

```text
cfg/                  YAML defaults for the command-line scripts
scripts/download/     Dataset download helpers
scripts/processing/   Ground-truth processing and voxel precomputation
scripts/testing/      EventsFormer inference scripts
scripts/viz/          Optional visualization utilities
src/learning/         Dataloaders and EventsFormer implementation
src/filter/           Filter implementation
src/main_network.py   Training entry point
src/main_filter.py    Filter entry point
```



## Setup

```bash
conda env create -f environment.yaml
conda activate tleio
git submodule update --init --recursive
```

Run all pipeline commands from the activated `tleio` environment. The EventsFormer model imports `einops`; it is listed in `environment.yaml`, and this quick check should pass before inference:

```bash
conda activate tleio
python -c "import einops; print('einops OK')"
```

If that import fails in an existing environment, update it with `conda env update -f environment.yaml` or install the missing package with `conda install -c conda-forge einops`.

Most scripts read defaults from `cfg/*.yaml`; command-line arguments override the YAML values.

## Download Data

### EDS

```bash
python scripts/download/download_eds.py --seq 0,1,2,3,4,5
```

This downloads EDS data under `data/eds`. The default sequence list is in `cfg/download_eds.yaml`.

### TartanAir + TartanEvent

```bash
python scripts/download/download_tartanair.py --env office --difficulty easy hard
```

Training data is downloaded under `data/tartanair`. The script combines TartanAir pose data with TartanEvent event streams.

### TartanAir + TartanEvent Competition Split

```bash
python scripts/download/download_tartanair_competition.py
```

Competition data is written under `data/tartanair/competition`.

To download and extract only one competition sequence, pass `--seq`. For example:

```bash
python scripts/download/download_tartanair_competition.py --seq MH001
```

## Process Data

Process EDS into train, validation, and test folders:

```bash
python scripts/processing/processing_eds.py --overwrite
```

Process TartanAir/TartanEvent:

```bash
python scripts/processing/processing_tartan.py --overwrite
```

Processed sequence folders contain the files used by the rest of the pipeline:

```text
events.h5
anchor_poses.txt
relative_motions.txt
stamped_groundtruth.txt
imu.csv
```

If a Tartan sequence does not include an IMU file, synthesize one from the processed ground truth:

```bash
SEQ=TartanEvent_competition_mono_MH001
PROCESSED_ROOT=data/tartanair/processed_testing

python scripts/processing/imu_synthesizer.py \
  --sequence_dir $PROCESSED_ROOT/$SEQ \
  --overwrite
```

The generated file is `$PROCESSED_ROOT/$SEQ/imu.csv` with columns:

```text
timestamp_us,gx,gy,gz,ax,ay,az
```

## Precompute Event Voxels

EventsFormer inference and training use precomputed voxel clips by default.

```bash
python scripts/processing/precompute_derotated_voxels.py \
  --root_dir data/eds/processed_testing \
  --output_dir data/eds/precomputed_testing \
  --denoising true \
  --overwrite
```

Each precomputed sequence contains `derotated_voxels.npy`, `relative_motions.txt`, and `metadata.json`.

## Train EventsFormer



```bash
python src/main_network.py \
  --root_dir data/eds/precomputed_train \
  --val_root_dir data/eds/precomputed_validation \
  --checkpoint_path checkpoints/eds_eventsformer
```

Checkpoints and the matching `args.txt` are saved in the selected checkpoint directory. 

## Run EventsFormer Inference

Run inference on one precomputed sequence and write the predicted relative motions into the matching processed sequence folder. This is the format expected by the filter.

```bash
DATASET=eds
SEQ=03_rocket_earth_dark
PRECOMPUTED_ROOT=data/$DATASET/precomputed_testing
PROCESSED_ROOT=data/$DATASET/processed_testing
CKPT=checkpoints/eds_eventsformer/checkpoint_best.pth

python scripts/testing/test.py \
  --sequence_dir $PRECOMPUTED_ROOT/$SEQ \
  --checkpoint_file $CKPT \
  --output_file $PROCESSED_ROOT/$SEQ/$SEQ.txt \
  --average_overlaps
```

The prediction file has columns:

```text
t0_us t1_us px py pz
```

If `--save_covariance` is used with a covariance checkpoint, the file also contains `sigma_x sigma_y sigma_z`.

To run inference on every sequence in a precomputed folder:

```bash
python scripts/testing/batch_test.py \
  --batch_root $PRECOMPUTED_ROOT \
  --checkpoint_file $CKPT \
  --output_dir data/$DATASET/predicted_relative_motions \
  --average_overlaps
```

## Run the Filter

Run the EKF on one processed sequence after writing the EventsFormer prediction file into that same sequence folder.

```bash
DATASET=eds
SEQ=03_rocket_earth_dark
PROCESSED_ROOT=data/$DATASET/processed_testing

python src/main_filter.py \
  --dataset $DATASET \
  --processed_root $PROCESSED_ROOT \
  --sequence $SEQ \
  --plot_transformer \
  --plot_projections
```

Filter outputs are saved under:

```text
outputs/main_filter/<dataset>/<sequence>/
```

The main files are `stamped_traj_estimate.txt` and the trajectory/error plots generated by `scripts/filter_diagnostics.py`.

## Inspect Network Trajectories

To reconstruct and plot a trajectory directly from relative-motion predictions:

```bash
DATASET=eds
SEQ=03_rocket_earth_dark
PROCESSED_ROOT=data/$DATASET/processed_testing

python scripts/plot_trajectories.py \
  --gt $PROCESSED_ROOT/$SEQ/stamped_groundtruth.txt \
  --rel $PROCESSED_ROOT/$SEQ/$SEQ.txt \
  --gt_rel $PROCESSED_ROOT/$SEQ/relative_motions.txt \
  --save_dir plots/${DATASET}_${SEQ}
```

## Reproduce the Main Pipeline Results

1. Create the environment and initialize submodules.
2. Download the target dataset split.
3. Run the matching processing script.
4. Generate `imu.csv` if the processed sequence does not already include IMU data.
5. Precompute event voxels for the processed split.
6. Run EventsFormer inference with the trained checkpoint.
7. Run `src/main_filter.py` on each processed sequence.
8. Use the saved files in `outputs/main_filter/<dataset>/<sequence>/` for trajectory plots and metrics.

### Sequential Tartan Competition Example

This is the explicit step-by-step path for one Tartan competition sequence. `RAW_SEQ` is the competition sequence name from the archive. `SEQ` is the processed sequence name produced by `processing_tartan.py` from the raw `<environment>/<difficulty>/<sequence>` layout.

```bash
conda activate tleio

RAW_SEQ=MH001
SEQ=TartanEvent_competition_mono_${RAW_SEQ}
RAW_COMP_ROOT=data/tartanair/competition/data/storage/pellerito
RAW_SEQUENCE_DIR=$RAW_COMP_ROOT/TartanEvent_competition/mono/$RAW_SEQ
PROCESSED_ROOT=data/tartanair/processed_testing
PRECOMPUTED_ROOT=data/tartanair/precomputed_testing
CKPT=checkpoints/checkpoint_last.pth.zip

python scripts/download/download_tartanair_competition.py \
  --seq $RAW_SEQ \
  --skip-air

python scripts/processing/processing_tartan.py \
  $RAW_COMP_ROOT \
  --save-path data/tartanair/processed_train \
  --save_path_testing $PROCESSED_ROOT \
  --test-seq $SEQ \
  --process_gt pose_lcam_front.txt \
  --overwrite

python scripts/processing/imu_synthesizer.py \
  --sequence_dir $PROCESSED_ROOT/$SEQ \
  --overwrite

python scripts/processing/precompute_derotated_voxels.py \
  --root_dir $PROCESSED_ROOT \
  --output_dir $PRECOMPUTED_ROOT \
  --denoising true \
  --overwrite

python scripts/testing/test.py \
  --sequence_dir $PRECOMPUTED_ROOT/$SEQ \
  --checkpoint_file $CKPT \
  --output_file $PROCESSED_ROOT/$SEQ/$SEQ.txt \
  --average_overlaps

python src/main_filter.py \
  --dataset tartanair \
  --processed_root $PROCESSED_ROOT \
  --sequence $SEQ \
  --plot_transformer \
  --plot_projections

python scripts/plot_trajectories.py \
  --gt $PROCESSED_ROOT/$SEQ/stamped_groundtruth.txt \
  --rel $PROCESSED_ROOT/$SEQ/$SEQ.txt \
  --gt_rel $PROCESSED_ROOT/$SEQ/relative_motions.txt \
  --save_dir plots/tartanair_${RAW_SEQ}_sequential
```

If the checkpoint predicts covariance and you want the filter to consume those per-axis sigmas, add `--save_covariance` to `scripts/testing/test.py`. The output file then contains:

```text
t0_us t1_us px py pz sigma_x sigma_y sigma_z
```

### One-Pass Raw Tartan Pipeline

`src/main.py` performs online voxelization, network inference, and EKF fusion directly from the raw event stream and an IMU CSV. Use the same raw sequence folder and checkpoint:

```bash
conda activate tleio

RAW_SEQ=MH001
SEQ=TartanEvent_competition_mono_${RAW_SEQ}
RAW_COMP_ROOT=data/tartanair/competition/data/storage/pellerito
RAW_SEQUENCE_DIR=$RAW_COMP_ROOT/TartanEvent_competition/mono/$RAW_SEQ
PROCESSED_ROOT=data/tartanair/processed_testing
CKPT=checkpoints/checkpoint_last.pth.zip

python src/main.py \
  --raw_sequence_dir $RAW_SEQUENCE_DIR \
  --checkpoint_file $CKPT \
  --imu_file $PROCESSED_ROOT/$SEQ/imu.csv \
  --output_dir outputs/main_online/tartanair/$SEQ
```

## Notes

The public repository does not include downloaded datasets or trained checkpoint binaries. Place released checkpoints under `checkpoints/` and keep their `args.txt` files next to the `.pth` files, because inference loads the training-time model configuration from that file. A missing or stale `args.txt` can make inference fail or silently use preprocessing/model settings that do not match the checkpoint.
The checkpoints can be downloaded from https://drive.google.com/drive/folders/1RnAKGuD_6BHSWama648qtUKyGo524Bw3?usp=sharing alongside the txt folder