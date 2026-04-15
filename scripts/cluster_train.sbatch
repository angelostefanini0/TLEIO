#!/bin/bash

#SBATCH --job-name="tleio_train"
#SBATCH --output=logs/%j_stdout.txt
#SBATCH --error=logs/%j_stderr.txt

#SBATCH --time=60:00
#SBATCH --account=3dv
#SBATCH --gpus=1080ti:4

. /etc/profile.d/modules.sh
module add cuda/12.6

. ~/env-tleio/bin/activate

# Clear any manual overrides
unset CUDA_VISIBLE_DEVICES

# Or explicitly set it to what Slurm provided
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 src/main_network.py --root_dir /work/scratch/astefanini/data/eds/processed_train --b_size 4 --clip_len 3 --delta_t_ms 50 --num_bins 5 --lr 2e-5 --epoch 20 --depth 6 --heads 6 --dim_head 64 --num_workers 2
