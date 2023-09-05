## For training and attacking models, first activate conda enviroment:

conda activate ../../envs

## To train a model exacute the following command: 
sbatch distilbertJob.sh

## To change training entry data points open albertJob.sh file and change --train, --valid and --test data path
python albert.py --train /project/verma/TextAttack_Parisa/data/train_emails.csv --valid /project/verma/TextAttack_Parisa/data/valid_emails.csv --test /project/verma/TextAttack_Parisa/data/test_emails.csv

## To exacute an attack exacute the following command:
textattckJob.sh

### To change the attack methods, the model under attack and entry data open textattckJob.sh file and change accordingly
### --model-from-huggingface : set the path to the folder that contains pytorch model and its tokenizer files
### --dataset-from-file: set the path to textAttack_dataset.py that contains the path to your dataset
### --recipe: select from one of theses text attack method, textfooler, deepwordbug, pwws and bae
### --num-examples: choose any number of examples you desire
textattack attack --model-from-huggingface /project/verma/TextAttack_Parisa/distillbert/distilbert_model --dataset-from-file /project/verma/TextAttack_Parisa/distillbert/textAttack_dataset.py --recipe deepwordbug --num-examples 518

## To change your input data in the text attack platform, open textAttack_dataset.py and change the data path in the following line of code which is line 6
df = pd.read_csv('/project/verma/TextAttack_Parisa/data/test_emails.csv', delimiter=',')

## The results of running albertJob.sh and textattckJob.sh save in the related log files.