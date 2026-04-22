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

## 2. Data Generation
### 2.1 EDS 

Download the eds dataset sequence from the Event Camera Dataset (EDS):

```bash
python scripts/download/download_eds.py data/eds --seq 0,1,2,3,4,5
```

Usage: write the directory to save it into first, then pass the exact sequence numbers you want to download with `--seq` as a comma-separated list.

### 2.2 TartanAir + TartanEvent

Run the `download_tartanair.py` script to download sequences from a specified environment in the TartanAir dataset. TartanAir is downloaded using an API, but does not come with events (or doesn't look to come with it by default), and TartanEvent is taken from UZH RPG resources found in the RAMPVO github repo. Argument  `env-event` asks for the environment of TartanEvent (see list here: https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent/), while `env-air` asks for the environment of TartanAir (see list at: https://github.com/castacks/tartanair_tools/blob/master/download_training_zipfiles.txt). All environments have two difficulties: easy and hard, which must be passed as an argument. 
The python API installed by `pip install` is not updated with the code from the current github repo of the official dataset, so to make the script work do the following: 

If already installed, otherwise skip: 

```bash
pip uninstall tartanair -y
```

```bash
git clone https://github.com/castacks/tartanairpy.git
cd tartanairpy
pip install -e .
git submodule update --init --recursive
```

Then when installed go to `tartanairpy/tartanair/tartanair_module.py`, line 51 and add `'pose'` to the modalities to download the ground truth pose. 


```bash
python scripts/download/download_tartanair.py \
--root data/tartanair \
--env-event office \
--env-air Office \
--difficulty easy hard
```

## 3. EDS Data pre-processing

Run the `processing_eds.py` script to process the event stream and the ground truth data to get supervision for the network. The script generates a ms_to_idx mapping for efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency. 

CURRENTLY WORKING FOR EDS ONLY, NEEDS MINOR FIXES TO WORK WITH THE TARTAN AIR DATASET AS WELL

```bash
python scripts/processing_eds.py data/eds/raw \
--save-path data/eds/processed_train \
--save_path_validation data/eds/processed_validation \
--validation-seq 3 \
--save_path_testing data/eds/processed_testing \
--test-seq 0,6 \
--overwrite \
--timestamps-key t \
--process_gt imu.csv stamped_groundtruth.txt \
--delta_t_ms 50 \
--anchor_t_ms 50

```
## 4. Tartan Data pre-processing

Run the `processing_tartan.py` script to process the event stream and the ground truth data to get supervision for the network. The script generates a ms_to_idx mapping for efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency. 

Pass the top-level Tartan root, for example `data/tartanair`, not a single environment folder. The script builds `stamped_groundtruth.txt` from `pose_lcam_front.txt` and `imu/cam_time.txt`, and can also synthesize an `imu.csv` from the Tartan IMU files.

```bash
python scripts/processing_tartan.py data/tartanair \
--save-path data/tartanair/processed_train \
--overwrite \
--timestamps-key events/t \
--process_gt pose_lcam_front.txt \
--generate_imu_csv true \
--delta_t_ms 50 \
--anchor_t_ms 50

```

## 5. Visualization of event data: 
Run the `scripts/viz/play_events_on_rgb.py` script to playback the input video with events overlayed onto RGB frames. `root` expects the absolute path to the dataset root, `sequence` expects the name of the sequence to inspect, and `height` / `width` are the image dimensions to display. To have the playback uncapped, set `fps` to `0`.

```bash
python scripts/viz/play_events_on_rgb.py \
--root /home/alessandro/Desktop/TLEIO/data/eds/raw \
--sequence 01_peanuts_light \
--height 480 \
--width 640 \
--start-img 1 \
--num-frames 30 \
--fps 12.5
```

The script also supports visualizing the shared background-activity denoiser used by the training pipeline:

```bash
python scripts/viz/play_events_on_rgb.py \
--root /home/alessandro/Desktop/TLEIO/data/eds/raw \
--sequence 01_peanuts_light \
--height 480 \
--width 640 \
--start-img 1 \
--num-frames 300 \
--fps 0 \
--denoising true \
--denoise-dt-us 1000 \
--denoise-radius 1 \
--denoise-min-supporters 1
```

When denoising is enabled, the overlay shows `kept/raw` event counts so it is easier to check whether the filter is doing what you expect.

## 6. Training the model: 
Run the `main_network.py` script to train the model. A bunch of arguments can be passed for general data handling, optimization strategies and model parameters.

Important parameters:
- `downsampling_factor` decreases the event image resolution before voxelization.
- The downsampled spatial size must stay divisible by `patch_size`. For example, with `patch_size=16`, `downsampling_factor=0.7` gives `336x448`, which is valid.
- `denoising` enables the shared background-activity filter before voxelization.

The model build step now prints the effective image size, patch grid, and resulting token count, so you can verify the downsampling is doing what you expect.

Example training command:

 ```bash
python src/main_network.py \
--root_dir data/eds/processed_train \
--val_root_dir data/eds/processed_validation \
--checkpoint_path checkpoints/eds_ds07_denoise_wl0 \
--b_size 2 \
--depth 12 \
--heads 6 \
--num_workers 8 \
--downsampling_factor 0.7 \
--denoising true \
--denoise_dt_us 2000 \
--denoise_radius 1 \
--denoise_min_supporters 2 \
--derotate true

```

## 7. Testing the model: 
Run the `test.py` script to test the model and save the motions into a file.

The script reads `args.txt` from the checkpoint directory, so the same training-time settings for downsampling and denoising are reused automatically during inference. You can also enable `--average_overlaps` to average multiple predictions that correspond to the same relative motion. Translation-only checkpoints now save `t0_us t1_us px py pz`.

```bash
python test.py \
--sequence_dir data/eds/processed_testing \
--checkpoint_file checkpoints/checkpoint_best.pth \
--output_file data/eds/path/to/save/outputs.txt \
--average_overlaps
```

## 8. Inspection of model output: 
Run the `inspect_relative_motions.py` script to see how the model predicition compares to the GT. `gt` expects the stamped groundtruth, `rel` accepts either translation-only predictions `[t0_us t1_us px py pz]` or full relative motions `[t0_us t1_us px py pz rx ry rz]`, `gt_rel` expects the groundtruth relative motions, and `gt_rel_mode` expects one of `[rotation, translation, both]`. With translation-only predictions, `rotation` is usually the most useful mode because it fuses predicted translation with GT rotation.

```bash
python inspect_functions/inspect_relative_motions.py \
--gt data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt \
--rel path/to/predicted_relative_motions.txt \
--gt_rel data/eds/processed/00_peanuts_dark/relative_motions.txt \
--gt_rel_mode rotation
```

## 9. Live visualization and inspection of results: 
Run the `scripts/viz/test_trajectory_with_events.py` script to playback the input video with events overlayed onto RGB frames, and get the corresponding trajectory plot live (gt against predicted from the model). It supports the same optional background-activity denoising controls as `play_events_on_rgb.py`.
```bash
python scripts/viz/test_trajectory_with_events.py \
--root /home/alessandro/Desktop/TLEIO/data/eds/raw \
--sequence 03_rocket_earth_dark \
--height 480 \
--width 640 \
--fps 0 \
--rel-model data/eds/predicted_relative_motions/sequence_03/big_relative_motions_averaged.txt \
--rel-gt data/eds/processed_testing/03_rocket_earth_dark/relative_motions.txt \
--gt data/eds/processed_testing/03_rocket_earth_dark/stamped_groundtruth.txt
```

To enable denoising during live playback, add for example:

```bash
--denoising true \
--denoise-dt-us 1000 \
--denoise-radius 1 \
--denoise-min-supporters 1
```
