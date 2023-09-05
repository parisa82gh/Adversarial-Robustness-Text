#!/bin/bash
#SBATCH -J TrainALBERT
#SBATCH -o TrainALBERT.o%j
#SBATCH --mail-user=pmehdigholampour@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 4:0:0
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8GB

python albert.py --train /project/verma/TextAttack_Parisa/augmented_data/train_1_4.csv  --test /project/verma/TextAttack_Parisa/data/test_emails.csv

