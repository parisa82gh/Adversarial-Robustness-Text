#!/bin/bash
#SBATCH -J TrainSqueezBert
#SBATCH -o TrainSqueezBert.o%j
#SBATCH --mail-user=pmehdigholampour@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 4:0:0
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8GB

python squeezbert.py --train /project/verma/TextAttack_Parisa/data/train_emails.csv --valid /project/verma/TextAttack_Parisa/data/valid_emails.csv --test /project/verma/TextAttack_Parisa/data/test_emails.csv

