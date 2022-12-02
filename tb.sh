#!/bin/bash
#SBATCH -J tb
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3133@nyu.edu
#SBATCH --output=runs/tb.out
#SBATCH --error=runs/tb.err

conda init bash
source ~/.bashrc 

module load anaconda3 cuda/11.3.1
source activate mlfd


cd ~/dlfd/

tensorboard --logdir=runs --host=127.0.0.1