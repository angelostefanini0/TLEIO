# TLEIO Setup and Quickstart

## 1. Environment Setup

Create and activate the conda environment to ensure all dependencies are met:

```bash
conda env create -f environment.yaml
conda activate tleio

## 2. Data Generation
Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):
```bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('[https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz](https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz)', '00_peanuts_dark.tgz')"
Gemini ha detto
Here is the cleaned-up and properly formatted Markdown code for your TLEIO Setup and Quickstart guide. I've corrected the syntax errors in your Python snippets and improved the visual hierarchy.

Markdown
# TLEIO Setup and Quickstart

## 1. Environment Setup

Create and activate the conda environment to ensure all dependencies are met:

```bash
conda env create -f environment.yaml
conda activate tleio
2. Dataset Preparation
Download the "00_peanuts_dark" sequence from the Event Camera Dataset (EDS):

Bash
cd data/eds
python -c "import urllib.request; urllib.request.urlretrieve('[https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz](https://download.ifi.uzh.ch/rpg/eds/dataset/00_peanuts_dark/00_peanuts_dark.tgz)', '00_peanuts_dark.tgz')"
Extract the archive into a new directory:

Bash
mkdir peanuts
python -c "import tarfile; tar = tarfile.open('00_peanuts_dark.tgz'); tar.extractall('peanuts'); tar.close()"
Directory Structure
Your data folder should now look like this:

Plaintext
data/
└── eds/
    └── peanuts/
        ├── events.h5
        ├── imu.csv
        ├── stamped_groundtruth
        ├── images/
        └── images_timestamps
3. Event Visualization
Run the viewer to inspect the event stream and verify the data:

Bash
python scripts/view_events.py --h5 data/eds/peanuts/events.h5
Visualization Legend:

<span style="color:red">Red</span>: Positive polarity events

<span style="color:blue">Blue</span>: Negative polarity events