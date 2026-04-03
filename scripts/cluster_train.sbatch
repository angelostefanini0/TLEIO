#!/bin/bash

#SBATCH --job-name="tleio_train"
#SBATCH --output=logs/%j_stdout.txt
#SBATCH --error=logs/%j_stderr.txt

#SBATCH --time=24:00
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1

. /etc/profile.d/modules.sh
module add cuda/12.6

. ~/env-tleio/bin/activate

python3 src/main_network.py --root_dir /work/scratch/astefanini/data/eds/processed_train --b_size 4 --clip_len 3 --delta_t_ms 50 --num_bins 5 --lr 2e-5 --epoch 20 --weighted_loss 0 --depth 6 --heads 6 --dim_head 64 --num_workers 3
