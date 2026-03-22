# Setup TLIO pipeline with EDS dataset

## Steps
*From this directory*
1. Clone [TLIO](https://github.com/CathIAS/TLIO) repo locally, **not in this repo**. Follow instructions to setup the conda environment - be sure to remove torch3d from `environment.yaml`.
2. Uncomment links in `links.txt` to select which datasets to download
3. Run `python pull_eds.py links.txt *path to tlio*\local_data\eds_raw` to download datasets. They are large, so this could take ~15 minutes for some sequences.
4. Run `python convert_eds.py -i *path to tlio*\local_data\eds_raw -o *path to tlio*\local_data\eds_processed`
5. Create train_list.txt, val_list.txt, and test_list.txt where each line includes a sequence to include in each split
6. Follow instructions in TLIO readme (and below) for training, testing, and running the filter

## Notes on running TLIO
1. *TODO* Angelo here! Commands, any notes about where to run commands, blah blah blah for training, testing, and running filter