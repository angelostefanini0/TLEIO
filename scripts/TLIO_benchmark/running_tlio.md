uncomment links in links.txt for datasets to download
run pull_eds and put data in local_data\eds_raw folder in TLIO repo
run convert eds with input dir being the localdata\eds_raw and output being local_data\eds_processed
create train_list.txt, val_list.txt, and test_list.txt with sequence names to be used listed on new lines
remove torch3d from requirements and install repo
make sure you have tqdm: python -m pip install tqdm
remove verbose=True from line 331 of train.py
train with this command from TLIO dir:
python src/main_net.py --mode train --root_dir .\local_data\eds_prepped\ --out_dir models/resnet_v1 --epochs 100