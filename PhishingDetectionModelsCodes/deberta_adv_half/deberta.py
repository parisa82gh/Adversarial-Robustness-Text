
# Import initial modules
import os
import argparse
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate

from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix

from datasets import load_dataset, load_metric

from transformers import TrainingArguments, Trainer, DebertaTokenizer, DebertaForSequenceClassification


# Define a function that tokenize the input examples
def main():
    # emails = 'clean_phish_legit_emails.csv'
    # df_email = pd.read_csv(emails)
    # phish = df_email.loc[df_email['label'] == 1]
    # legit = df_email.loc[df_email['label'] == 0]
    # print('Number of original dataset phishing emails: {}'.format(phish.shape[0]))
    # print('Number of original dataset legitimate emails: {}'.format(legit.shape[0]))

    # df_email = df_email.sample(frac=1)

    # """#Dataset Distribution"""
    # # Split dataset dataframe into train and test dataframe using stratify to keep the original dataset distribution in them 
    # df_train, df_rest = train_test_split(df_email,random_state=42,train_size = 0.8, stratify=df_email.label.values)
    # df_valid, df_test = train_test_split(df_rest,random_state=42,train_size = 0.5, stratify=df_rest.label.values)

    # # Save the two dataframes into two seperate csv files 
    # df_train.to_csv('train_emails.csv',index=False)
    # df_valid.to_csv('valid_emails.csv',index=False)
    # df_test.to_csv('test_emails.csv',index=False)

    # """Build a dataframe with short length text"""

    # Transform .csv files to hugginface dataset format


    parser = argparse.ArgumentParser()
    parser.add_argument('--train',metavar='train',help='.csv file to read')
    parser.add_argument('--valid',metavar='valid',help='.csv file to write')
    parser.add_argument('--test',metavar='test',help='.csv file to write')
    args = parser.parse_args()

    dataset = load_dataset('csv', data_files= {'train':args.train, 'valid':args.valid, 'test':args.test})

    """HuggingFace dataset format"""

    """Set up the Tokenizer"""
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer = DebertaTokenizer.from_pretrained("/project/verma/TextAttack_Parisa/deberta/deberta_model")

    # Map the dataset using tokenizer to create a tokenized dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Based on the train or test label in the dataset, we can generate train and test tokenized dataset
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["valid"]
    test_dataset = tokenized_dataset["test"]

    model = DebertaForSequenceClassification.from_pretrained("/project/verma/TextAttack_Parisa/deberta/deberta_model", num_labels=2)

    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )

    # By using train function in trainer, we can fine_tune the model
    trainer.train()

    # To evaluate the model, use load_metric class to calculate accuracy parameters. 
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())

    #Prediction
    predictions = trainer.predict(tokenized_dataset["test"])
    
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    print(metric.compute(predictions=preds, references=predictions.label_ids))

    # Save the fine tuned model using save_pretrained function
    model.save_pretrained("deberta_model")
    tokenizer.save_pretrained("deberta_model")

if __name__ == "__main__":
    main()




