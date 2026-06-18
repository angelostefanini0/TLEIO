# TLEIO Setup and Quickstart

## 1. Environment Setup

Create and activate the conda environment to ensure all dependencies are met:

```bash
conda env create -f environment.yaml
conda activate tleio
```

To properly activate the rpg_trajectory_evaluation toolbox, imported in this repo as a git submodule, run the following command:

```bash
git submodule update --init --recursive
```

## 1.1 Script Configs

Launch defaults live in `cfg/*.yaml`. Supported scripts load their matching config automatically, and any CLI argument you pass overrides the YAML value.

For example:

```bash
python scripts/processing/precompute_derotated_voxels.py
python scripts/testing/test.py --sequence_dir data/eds/precomputed_testing/03_rocket_earth_dark
python src/main_network.py --config cfg/train.yaml --b_size 8
```

Edit the corresponding YAML file when you want a reusable launch setup without typing every argument each time.

For a normal run, the short commands below are enough as long as the matching YAML file contains the paths and options you want. Use CLI arguments only for one-off overrides.

## 2. Data Download

We provide example download scripts to download the synthetic data that has been used to train the network (TartanAir and TartanEvent), and real event-camera data from Event-aided Direct Sparse (EDS) Odometry paper. The download scripts are configured through `cfg/download_*.yaml` and can also be overridden from the CLI.

### 2.1 EDS

Download sequences from the EDS dataset:

- EDS project page: <https://rpg.ifi.uzh.ch/eds.html>
- EDS download root used by the script: <https://download.ifi.uzh.ch/rpg/eds/dataset>

```bash
python scripts/download/download_eds.py
```

By default, the script uses `cfg/download_eds.yaml`. To select sequences without editing the YAML:

```bash
python scripts/download/download_eds.py --seq 0,1,2,3
```

### 2.2 TartanAir + TartanEvent Training Data

The training downloader combines TartanEvent event streams with the matching TartanAir pose metadata. TartanEvent archives are downloaded from the RAMP-VO data resources, while TartanAir archives are downloaded from Hugging Face using the official TartanAir file list.

- TartanAir dataset website: <https://theairlab.org/tartanair-dataset/>
- TartanAir Hugging Face dataset used by the script: <https://huggingface.co/datasets/theairlabcmu/tartanair>
- TartanAir training file list used by the script to check for available sequences: <https://raw.githubusercontent.com/castacks/tartanair_tools/master/download_training_zipfiles.txt>
- RAMP-VO paper: <https://arxiv.org/abs/2309.09947>
- RAMP-VO/TartanEvent data root used by the script: <https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/>

Run the training downloader with the defaults in `cfg/download_tartanair.yaml`:

```bash
python scripts/download/download_tartanair.py
```

To download specific training environments:

```bash
python scripts/download/download_tartanair.py \
  --env office carwelding endofworld \
  --difficulty easy hard
```
The scripts also support partial downloads of ground-truth/camera data or event data only. 

### 2.3 TartanAir + TartanEvent Competition Data

Competition data is handled separately to keep the training downloader simple. The script downloads the TartanEvent competition event archive from the RAMP-VO data root and the TartanAir monocular test release from the Google Drive link used by the original download script.

```bash
python scripts/download/download_tartanair_competition.py
```

To download only one side of the competition data:

```bash
python scripts/download/download_tartanair_competition.py --skip-air
python scripts/download/download_tartanair_competition.py --skip-event
```

The default output root is `data/tartanair/competition`.

## 3. Data pre-processing

We provide example scripts to generate supervision data for the network, and to pre-compute the event voxels for efficient GPU training, storing them in npy files.   

### 3.1 EDS Data pre-processing

Run the `scripts/processing/processing_eds.py` script to process and the ground truth data to get supervision for the network using the data from the EDS dataset. The script generates a ms_to_idx mapping for efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency.

```bash
python scripts/processing/processing_eds.py
```

### 3.2 Tartan Data pre-processing

Run the `scripts/processing/processing_tartan.py` script to process the ground truth data to get supervision for the network.

The script builds `stamped_groundtruth.txt` from `pose_lcam_front.txt` and `imu/cam_time.txt`

```bash
python scripts/processing/processing_tartan.py
```

### 3.3 Precomputing event voxels

Training can either build event voxel grids online in the dataloader or read
precomputed voxels from disk. For faster training, we recommend precomputing the
voxel grids once and storing them as `.npy` files. This moves denoising,
downsampling, voxelization, and optional event de-rotation out of the training
loop, reducing CPU dataloader overhead and helping the GPU stay busy.

Run:

```bash
python scripts/processing/precompute_derotated_voxels.py
```

The script reads processed sequences from `root_dir` and writes one output
folder per sequence under `output_dir`. Each output sequence contains:

- `derotated_voxels.npy`: voxel tensor with shape `[N, C, H, W]`, where `N` is
  the number of anchor timestamps and `C` is the number of temporal bins.
- `relative_motions.txt`: copied supervision targets used by the training
  dataset.
- `metadata.json`: preprocessing settings used to produce the voxel file.

Defaults live in `cfg/precompute_derotated_voxels.yaml`. Typical overrides are:

```bash
python scripts/processing/precompute_derotated_voxels.py \
  --root_dir data/tartanair/processed_train \
  --output_dir data/tartanair/precomputed_train \
  --denoising true \
  --derotate true \
  --overwrite
```

The training script can then consume the precomputed folders directly, avoiding
event slicing and voxel construction during each epoch.

## 4. Training the model:
Run the `main_network.py` script to train the model. A bunch of arguments can be passed for general data handling, optimization strategies and model parameters.

Single GPU training with precomputed voxelization:

```bash
python src/main_network.py
```

### 4.1 Training Profiling

Training profiling is still built into `src/main_network.py`. To run a short profiling pass, edit `cfg/train_profile.yaml` if needed and run:

```bash
python src/main_network.py --config cfg/train_profile.yaml
```
Compare the printed timing line at the end of the epoch:
- If `avg_data_wait` is still larger than `avg_compute`, try more workers before increasing batch size further.
- If `avg_compute` is dominant and GPU memory is still comfortable, try a larger `--b_size`.
- If throughput stops improving, that configuration is already near the sweet spot.

## 5. Testing the model: 
Run the `scripts/testing/test.py` script to test the model and save the motions into a file.

The script reads `args.txt` from the checkpoint directory, so the same training-time settings for downsampling and denoising are reused automatically during inference. You can also enable `--average_overlaps` to average multiple predictions that correspond to the same relative motion. Translation-only checkpoints now save `t0_us t1_us px py pz`.

```bash
python scripts/testing/test.py
```

## 5.1 Batch testing precomputed sequences:
Use `scripts/testing/batch_test.py` to run `scripts/testing/test.py` on every valid sequence folder inside a precomputed root. A valid sequence folder must contain `derotated_voxels.npy` and `relative_motions.txt`.

Example used for the Office precomputed dataset:

```bash
python scripts/testing/batch_test.py
```

The main prediction files are saved as:

```text
data/tartanair/predicted_relative_motions/precomputed_office_integer/<sequence>.txt
```

Files ending in `_raw.txt` are raw model outputs for debugging and are not needed for trajectory plots.

To plot all Office sequences against ground truth:

```bash
for seq_dir in data/tartanair/precomputed_office_integer/*; do
  seq=$(basename "$seq_dir")

  python scripts/inspect_relative_motions.py \
    --gt "data/tartanair/processed_train/$seq/stamped_groundtruth.txt" \
    --rel "data/tartanair/predicted_relative_motions/precomputed_office_integer/$seq.txt" \
    --gt_rel "$seq_dir/relative_motions.txt" \
    --save_dir "plots/precomputed_office_integer/$seq"
done
```

## 6. Inspection of model output: 
Run the `inspect_relative_motions.py` script to see how the model predicition compares to the GT. `gt` expects the stamped groundtruth, `rel` expects translation-only predictions `[t0_us t1_us px py pz]`, and `gt_rel` expects the groundtruth relative motions used for reference translations and trajectory rotations.

```bash
python scripts/inspect_relative_motions.py
```

## 7. Live visualization and inspection of results: 
Run the `scripts/viz/test_trajectory_with_events.py` script with trajectory arguments to playback the input video with events overlayed onto RGB frames, and get the corresponding trajectory plot live (gt against predicted from the model). It supports the same optional background-activity denoising controls as the event-only mode and denoising during live playback
```bash
python scripts/viz/test_trajectory_with_events.py
```

## 8. Running the filter
Run the `src/main_filter.py` script to run the filter and save the results in the output directory 'outputs', alongside 3D trajectory, position and rotation comparison plots. Use '--h' for a complete overview of the parsers.
```bash
python src/main_filter.py \
--dataset DATASET \
--sequence SEQUENCE \
--plot_transformer \    # To visualize the trajectory estimated by the network-
--plot_projections \    # To save the 2D projections plots.
```
