#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme

# python run.py experiment=anli_mixup seed=${1} lang=${2} shots=${3} module.alpha=0.4 logger.wandb.project="anli-mixup-${2}-${5}" +module.module_cfg.weights_from_checkpoint.ckpt_path="your_mnli_checkpoint.ckpt"

# seed shots lang 
python run.py experiment=anli_mixup seed=${1} shots=${2} lang=${3} module.alpha=0.4 logger.wandb.project="anli-mixup-${3}-joint" test_after_training=false
