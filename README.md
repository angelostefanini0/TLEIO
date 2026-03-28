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

Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):

```bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz', '00_peanuts_dark.tgz')"
```

Extract the archive into data/eds/raw/00_peanuts_dark directory:

```bash
tar -xzf 00_peanuts_dark.tgz
```
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
python scripts/download_tartanair.py \
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
--overwrite  \
--timestamps-key t \
--process_gt imu.csv stamped_groundtruth.txt \
--delta_t_ms 50 \
--anchor_hz 20
```


