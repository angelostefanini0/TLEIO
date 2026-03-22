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

Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):

```bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz', '00_peanuts_dark.tgz')"
```

Extract the archive into data/eds/raw/00_peanuts_dark directory:

```bash
tar -xzf 00_peanuts_dark.tgz
```

## 3. Event Visualization

Run the viewer to inspect the event stream and verify the correctness of data:

```bash
python scripts/view_events.py --h5 data/eds/00_peanuts_dark/events.h5
```

## 4. Data pre-processing

Run the processing.py script to process the event stream and the ground truth data to get supervision for the network. The script generates a ms_to_idx mapping fro efficient event retrieval in the dataloader, and the relative transforms between ground truth poses downsampled at the target frequency. Remember to change sequence_name to the actual sequence name (ex. 01_peanuts_light) in the raw folder

```bash
python scripts/processing.py data/eds/raw   \
--save-path data/eds/processed   \
--overwrite  \
--timestamps-key t \
--process_gt imu.csv stamped_groundtruth.txt\
--delta_t_ms 50 \
--anchor_hz 20
```

