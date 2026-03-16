# TLEIO Setup and Quickstart

## 1. Environment Setup

Create and activate the conda environment to ensure all dependencies are met:

```bash
conda env create -f environment.yaml
conda activate tleio```

## 2. Data Generation
Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):
```bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('[https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz](https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz)', '00_peanuts_dark.tgz')".
```

## 2. Dataset Preparation
Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):

```
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('[https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz](https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz)', '00_peanuts_dark.tgz')"
```
Extract the archive into a new directory:

```
mkdir peanuts
python -c "import tarfile; tar = tarfile.open('00_peanuts_dark.tgz'); tar.extractall('peanuts'); tar.close()"
```
Your data folder should now look like this:

### Directory Structure

```text
data/
└── eds/
    └── peanuts/
        ├── events.h5
        ├── imu.csv
        ├── stamped_groundtruth
        ├── images/
        └── images_timestamps ```
## 3. Event Visualization
Run the viewer to inspect the event stream and verify the data:

```
python scripts/view_events.py --h5 data/eds/peanuts/events.h5
```

