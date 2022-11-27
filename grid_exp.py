import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obs_ord', type=int, default=-1)
args = parser.parse_args()
assert args.obs_ord > 0
# task_id == obs_ord
    
if __name__ == '__main__':
    for hidden_size in (16, 32):
        # for wptype in [0.25, 0.5, 0.75]:
        for num_layers in (1, 2, 3):
            subprocess.call(f'python train.py \
                --obs_ord {args.obs_ord} \
                --hidden_size {hidden_size} \
                --num_layers {num_layers}', \
                shell=True
            )