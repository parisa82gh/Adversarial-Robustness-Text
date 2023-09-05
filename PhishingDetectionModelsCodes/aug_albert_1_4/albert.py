
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

from transformers import TrainingArguments, Trainer, AlbertTokenizer, AlbertForSequenceClassification


# Define a function that tokenize the input examples

def main():
    
    # Transform .csv files to hugginface dataset format
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',metavar='train',help='.csv file to read')
    parser.add_argument('--test',metavar='test',help='.csv file to write')
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    df_train, df_valid = train_test_split(df,random_state=42,train_size = 0.9, stratify=df.label.values)
    df_train.to_csv('train_emails_1_4.csv',index=False)
    df_valid.to_csv('valid_emails_1_4.csv',index=False)

    dataset = load_dataset('csv', data_files= {'train':'train_emails_1_4.csv', 'valid':'valid_emails_1_4.csv', 'test':args.test})

    """HuggingFace dataset format"""

    """Set up the Tokenizer"""
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
        
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Map the dataset using tokenizer to create a tokenized dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Based on the train or test label in the dataset, we can generate train and test tokenized dataset
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["valid"]
    test_dataset = tokenized_dataset["test"]

    model = model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )

    # By using train function in trainer, we can fine_tune the model
    trainer.train()

    # To evaluate the model, use load_metric class to calculate accuracy parameters. 
    metric = load_metric("accuracy")

    # Evaluation

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

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
    print(predictions.predictions.shape, predictions.label_ids.shape)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    print(metric.compute(predictions=preds, references=predictions.label_ids))

    accuracy = evaluate.load("accuracy")
    f1= evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    matthews = evaluate.load("matthews_correlation")

    print(matthews.compute(predictions=preds, references=predictions.label_ids))
    print(accuracy.compute(predictions=preds, references=predictions.label_ids))
    print(f1.compute(predictions=preds, references=predictions.label_ids))
    print(precision.compute(predictions=preds, references=predictions.label_ids))
    print(recall.compute(predictions=preds, references=predictions.label_ids))

    # Save the fine tuned model using save_pretrained function
    model.save_pretrained("albert_model_1_4")
    tokenizer.save_pretrained("albert_model_1_4")

if __name__ == "__main__":
    main()




