#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# LM configuration
python run.py experiment=anli_mixup seed=${1} lang=${2} shots=${3} module.alpha=0.1 logger.wandb.project="anli-macro-${2}-lm" module._target_="src.projects.mixup.module.MultiTaskSequenceClassification" datamodule.dataloader_cfg.batch_size=16
# replace last with oracle and ckpt number after mnli-zs pre-training
# python run.py experiment=anli_mixup seed=${1} lang=${2} shots=${3} module.alpha=0.1 logger.wandb.project="anli-macro-${2}-last" +module.module_cfg.weights_from_checkpoint.ckpt_path="your_mnli_checkpoint.ckpt" module._target_="src.projects.mixup.module.MultiTaskSequenceClassification" datamodule.dataloader_cfg.batch_size=16
