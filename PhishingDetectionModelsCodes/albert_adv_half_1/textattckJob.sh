#!/bin/bash
#SBATCH -J TextAttack
#SBATCH -o TextAttack.o%j
#SBATCH --mail-user=pmehdigholampour@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 40:0:0
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=32GB

textattack attack --model-from-huggingface /project/verma/TextAttack_Parisa/albert_adv_half_1/albert_model --dataset-from-file /project/verma/TextAttack_Parisa/albert_adv_half_1/textAttack_dataset.py --recipe bae --num-examples 604
