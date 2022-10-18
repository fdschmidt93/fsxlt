#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# PRETRAIN MNLI checkpoints
python run.py experiment=mnli seed=${1}
