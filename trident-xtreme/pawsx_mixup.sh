#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# lang shots seed ckpt
# requires MNLI xlm-roberta-base checkpoint as trained with our framework!
# remove weights_from_checkpoint config and train pawsx-mixup-lm
python run.py experiment=pawsx_mixup seed=${3} lang=${1} shots=${2} module.alpha=0.4 logger.wandb.project="pawsx-mixup-${1}-last" test_after_training=false +module.module_cfg.weights_from_checkpoint.ckpt_path=your_pawsx_checkpoint.ckpt
