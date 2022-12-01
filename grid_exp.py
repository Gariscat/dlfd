import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obs_ord', type=int, default=-1)
args = parser.parse_args()
assert args.obs_ord > 0
# task_id == obs_ord
    
if __name__ == '__main__':
    log_dir = './runs/'
    ckpt_dir = './ckpt/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for spec_aug in (True, False,):
        for hidden_size in (32,):
            # for wptype in [0.25, 0.5, 0.75]:
            for num_layers in (1, 2,):
                subprocess.call(f'python train.py \
                    --obs_ord {args.obs_ord} \
                    --hidden_size {hidden_size} \
                    --num_layers {num_layers} \
                    --spec_aug {spec_aug}', \
                    shell=True
                )