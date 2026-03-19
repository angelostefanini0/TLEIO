# TLEIO Setup and Quickstart

## 1. Environment Setup

Create and activate the conda environment to ensure all dependencies are met:

```bash
conda env create -f environment.yaml
conda activate tleio
```

To properly activate the rpg_trajectory_evaluation toolbox, imported in this repo as a git submodule, run the followingo command:

```bash
git submodule update --init --recursive
```

## 2. Data Generation

Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):

```bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz', '00_peanuts_dark.tgz')"
```

Extract the archive into a new directory:

```bash
tar -xzf 00_peanuts_dark.tgz
```

## 3. Event Visualization

Run the viewer to inspect the event stream and verify the correctness of data:

```bash
python scripts/view_events.py --h5 data/eds/00_peanuts_dark/events.h5
```
