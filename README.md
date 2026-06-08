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
### 2.1 EDS 

Download the eds dataset sequence from the Event Camera Dataset (EDS):

```bash
python scripts/download/download_eds.py
```

### 2.2 TartanAir + TartanEvent

Run the `download_tartanair.py` script to download sequences from one or more TartanEvent/TartanAir environments. TartanAir is downloaded from HuggingFace and TartanEvent is taken from UZH RPG resources found in the RAMPVO github repo. HuggingFace uses the same environment names as TartanEvent, so each environment is passed once with `--env`.

```bash
python scripts/download/download_tartanair.py
```

Multiple environments can be downloaded and merged in one run:

```bash
python scripts/download/download_tartanair.py \
--env office carwelding endofworld \
```

If the IMU data is failing to download then only the events will be in the folder, so later run 
```bash
python scripts/download/download_tartanair.py \
--skip-event
```
If the first partial download of events lives in a different folder than the one specified in the second one, then the second command should add `--merge-root /path/to/that/partial/folder`.

## 3. EDS Data pre-processing

Run the `scripts/processing/processing_eds.py` script to process the event stream and the ground truth data to get supervision for the network. The script generates a ms_to_idx mapping for efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency.

```bash
python scripts/processing/processing_eds.py
```
## 4. Tartan Data pre-processing

Run the `scripts/processing/processing_tartan.py` script to process the event stream and the ground truth data to get supervision for the network.

The script builds `stamped_groundtruth.txt` from `pose_lcam_front.txt` and `imu/cam_time.txt`

```bash
python scripts/processing/processing_tartan.py
```
When running on the server, add argument `--materialize-events-file`, so that the event file is not symlinked from the raw folder and `--remove-raw-after-materialize`, so as not to use space on the disk in the cluster. The training now runs on already voxelized (denoised and/or derotated) events to take advantage of full GPU compute. 

```bash
python scripts/processing/precompute_derotated_voxels.py
```

## 5. Training the model:
Run the `main_network.py` script to train the model. A bunch of arguments can be passed for general data handling, optimization strategies and model parameters.

Single GPU training with precomputed voxelization:

```bash
python src/main_network.py
```

### 5.1 Training Profiling

Training profiling is still built into `src/main_network.py`. To run a short profiling pass, edit `cfg/train_profile.yaml` if needed and run:

```bash
python src/main_network.py --config cfg/train_profile.yaml
```
Compare the printed timing line at the end of the epoch:
- If `avg_data_wait` is still larger than `avg_compute`, try more workers before increasing batch size further.
- If `avg_compute` is dominant and GPU memory is still comfortable, try a larger `--b_size`.
- If throughput stops improving, that configuration is already near the sweet spot.

## 6. Testing the model: 
Run the `scripts/testing/test.py` script to test the model and save the motions into a file.

The script reads `args.txt` from the checkpoint directory, so the same training-time settings for downsampling and denoising are reused automatically during inference. You can also enable `--average_overlaps` to average multiple predictions that correspond to the same relative motion. Translation-only checkpoints now save `t0_us t1_us px py pz`.

```bash
python scripts/testing/test.py
```

## 6.1 Batch testing precomputed sequences:
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

## 7. Inspection of model output: 
Run the `inspect_relative_motions.py` script to see how the model predicition compares to the GT. `gt` expects the stamped groundtruth, `rel` expects translation-only predictions `[t0_us t1_us px py pz]`, and `gt_rel` expects the groundtruth relative motions used for reference translations and trajectory rotations.

```bash
python scripts/inspect_relative_motions.py
```

## 8. Live visualization and inspection of results: 
Run the `scripts/viz/test_trajectory_with_events.py` script with trajectory arguments to playback the input video with events overlayed onto RGB frames, and get the corresponding trajectory plot live (gt against predicted from the model). It supports the same optional background-activity denoising controls as the event-only mode and denoising during live playback
```bash
python scripts/viz/test_trajectory_with_events.py
```

## 10. Running the filter
Run the `src/main_filter.py` script to run the filter and save the results in the output directory 'outputs', alongside 3D trajectory, position and rotation comparison plots. Use '--h' for a complete overview of the parsers.
```bash
python src/main_filter.py \
--dataset DATASET \
--sequence SEQUENCE \
--plot_transformer \    # To visualize the trajectory estimated by the network-
--plot_projections \    # To save the 2D projections plots.
```
