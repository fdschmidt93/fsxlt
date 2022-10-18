#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
python run.py experiment=pawsx_zs seed=${1}
