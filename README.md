# Don't Stop Fine-Tuning: On Training Regimes for Few-Shot Cross-Lingual Transfer with Multilingual Language Models

This is the code for our experiments as part of our paper `Don't Stop Fine-Tuning: On Training Regimes For Few-Shot Cross-Linugal Transfer with Multilingual Language Models`, a detailed study on the few-shot cross-lingual transfer learning setup in which we propose a simple framework of joint finetuning of the source and target language to overcome the instability and improve performance upon the conventional sequential fine-tuning


You can install the required dependencies in two steps:

1. `conda env create -f environment.yaml`
2. Activate the conda environment `conda env activate trident_xtreme`
3. Change your working directory to `trident`
4. `pip install -e ./`

Then switch to `trident-xtreme` and

1. `conda activate trident_xtreme`
2. `bash $YOUR_TASK_REGIME.sh`

Note that, for the time being, `last` and `oracle` regimes would require fine-tuning on the source language task. You should be able to train `lm`-variants out-of-the-box after appropriate setup.

# Contact

**Name:** Fabian David Schmidt\
**Mail:** fabian.schmidt@uni-wuerzburg.de\
**Affiliation:** Center For Artificial Intelligence and Data Science (CAIDAS), University of Würzburg

# TODO

- [ ] Link paper from ACL anthology
- [ ] Citation to be added once proceedings are released
- [ ] Make checkpoints by task available
