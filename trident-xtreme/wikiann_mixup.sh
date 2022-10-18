#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# Positional args: seed[int] shot[int] lang[str] ckpt[int] last|oracle[str]
# requires wikiann-en-train pretraining!
# remove weights_from_checkpoint config and train wikiann-mixup-lm
python run.py experiment=wikiann_mixup seed=${1} lang=${3} shots=${2} module.alpha=0.4 logger.wandb.project="wikiann-mixup-${3}-${5}" module._target_="src.projects.mixup.module.MultiTaskTokenClassification" datamodule.dataloader_cfg.train.batch_size=16 +module.module_cfg.weights_from_checkpoint.ckpt_path=your_checkpoint.ckpt test_after_training=false 
