#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46
# do
# done
python run.py experiment=wikiann_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 test_after_training=false
