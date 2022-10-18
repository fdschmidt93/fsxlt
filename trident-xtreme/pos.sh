#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
python run.py experiment=pos_zs seed=${1} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 # logger.wandb.project="pos-zs-robust-base" 'logger.wandb.name="seed=${seed}-lr=${module.optimizer.lr}-epochs=${trainer.max_epochs}-dim=plain"' hydra.run.dir='logs/pos_robust_base/${seed}/${module.optimizer.lr}/plain'
