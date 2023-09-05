#!/bin/bash
#SBATCH -J TextAttack
#SBATCH -o TextAttack.o%j
#SBATCH --mail-user=pmehdigholampour@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 24:0:0
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=32GB

textattack attack --model-from-huggingface /project/verma/TextAttack_Parisa/roberta_adv_half/roberta_model --dataset-from-file /project/verma/TextAttack_Parisa/roberta_adv_half/textAttack_dataset.py --recipe deepwordbug --num-examples 418
