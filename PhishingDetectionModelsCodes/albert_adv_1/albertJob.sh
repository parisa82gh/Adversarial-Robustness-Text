#!/bin/bash
#SBATCH -J TrainALBERT
#SBATCH -o TrainALBERT.o%j
#SBATCH --mail-user=pmehdigholampour@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 4:0:0
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8GB

python albert.py --train /project/verma/TextAttack_Parisa/adv_data/train_f_adv.csv --valid /project/verma/TextAttack_Parisa/adv_data/valid_f_adv.csv --test /project/verma/TextAttack_Parisa/adv_data/test_f_adv.csv

