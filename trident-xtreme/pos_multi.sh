#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme

# seed shots lang ckpt
python run.py experiment=pos_multi seed=${1} shots=${2} logger.wandb.project="pos-multi-last" test_after_training=false module.module_cfg.weights_from_checkpoint.ckpt_path=your_pos_checkpoint.ckpt datamodule.dataloader_cfg.train.batch_size=8 
