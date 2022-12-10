import subprocess
import os
import argparse

    
if __name__ == '__main__':
    log_dir = './runs/'
    ckpt_dir = './ckpt/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for model in ('Chen', 'RNNBase', 'RNNSpecAug',):
        if 'RNN' not in model:
            # Chen's baseline
            # for alpha in (100, 500, 1000,):
            subprocess.call(f'python train.py --model Chen', shell=True)
            continue
            
        for obs_ord in (1, 2, 3):
            for hidden_size in (32,):
                # for wptype in [0.25, 0.5, 0.75]:
                for num_layers in (1, 2,):
                    subprocess.call(f'python train.py \
                        --obs_ord {obs_ord} \
                        --hidden_size {hidden_size} \
                        --num_layers {num_layers} \
                        --model {model}', \
                        shell=True
                    )