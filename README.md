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
Then when installed run: 

```bash
python scripts/download/download_tartanair.py \
--root data/tartanair \
--env-event office \
--env-air Office \
--difficulty easy hard
```

## 3. Event Visualization

Run the viewer to inspect the event stream and verify the correctness of data:

```bash
python scripts/view_events.py --h5 data/eds/00_peanuts_dark/events.h5
```



## 4. Data pre-processing

Run the `processing.py` script to process the event stream and the ground truth data to get supervision for the network. The script generates a ms_to_idx mapping for efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency. 

CURRENTLY WORKING FOR EDS ONLY, NEEDS MINOR FIXES TO WORK WITH THE TARTAN AIR DATASET AS WELL

```bash
python scripts/processing.py data/eds/raw   \
--save-path data/eds/processed   \
--save_path_testing data/eds/processed_testing \
--test-seq 0,6 \
--overwrite  \
--timestamps-key t \
--process_gt imu.csv stamped_groundtruth.txt \
--delta_t_ms 50 \
--anchor_hz 20
```

## 5. Inspection of model output: 
Run the `inspect_relative_motions.py` script to see how the model predicition compares to the GT. `gt` argument expects the stamped groundtruth, `rel` expects the predicted motions from the network, `gt_rel` expects the groundtruth relative motions, `gt_rel_mode` expects one of `[rotation, translation, both,]`. If `rotation` is used, the output will be the model predicted translation with gt rotation, if `translation` is used, the output will be the model predicted rotation, with gt translation, if `both` is used, the output will be the full gt relative motion.

```bash
python inspect_functions/inspect_relative_motions.py \
--gt data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt \
--rel path/to/predicted_relative_motions.txt \
--gt_rel data/eds/processed/00_peanuts_dark/relative_motions.txt \
--gt_rel_mode rotation
```

## 6. Visualization of event data: 
Run the `play_events_on_rgb.py` script to playback the input video with events overlayed onto RGB frames. `root` argument expects the absolute path to the root folder, `sequence` expects the name of the sequence to inspect, `height` and `width` are the input dimensions of the images to display. To have the playback uncapped, set `fps` to 0.

```bash
python scripts/play_events_on_rgb.py \
--root /home/alessandro/Desktop/TLEIO/data/eds/raw \
--sequence 01_peanuts_light \
--height 480 \
--width 640 \
--start-img 1 \
--num-frames 30 \
--fps 12.5
```
