#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme

# seed shots lang ckpt last|oracle
python run.py experiment=pos_mixup seed=${1} lang=${3} shots=${2} module.alpha=-1 logger.wandb.project="pos-macro-${3}-${5}" module._target_="src.projects.mixup.module.MultiTaskTokenClassification" datamodule.dataloader_cfg.train.batch_size=16 test_after_training=false +module.module_cfg.weights_from_checkpoint.ckpt_path=your_pos_checkpoint.ckpt
