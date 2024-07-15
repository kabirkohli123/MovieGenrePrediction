import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn import metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

class ClassificationDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,item):
        text = str(self.data[item]['synopsis'])
        target = str(self.data[item]['genre'])
        inputs = self.tokenizer(text,max_length = 20, padding = 'max_length',truncation=True)

        ids = inputs("input_ids")
        mask = inputs("attention_mask")

        return{
            "input_ids":torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = metrics.accuracy_score(labels,predictions)
    return{
        "accuracy": accuracy
    }


def train():
    df = load_dataset("test_data.txt")
    df = df.class_encode_column('genre')

    df_train = df['train']
    df_test = df['test']

    temp_df = df_train.train_test_split(test_size=0.2,stratify_by_column="genre")

    df_train = temp_df['train']
    df_val = temp_df['test']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels= len(df_train.features['genre']._int2str))

    train_dataset = ClassificationDataset(df_train, tokenizer)
    val_dataset = ClassificationDataset(df_val, tokenizer)
    test_dataset = ClassificationDataset(df_test, tokenizer)

    args = TrainingArguments(
        "model",
        evaluation_strategy = "epoch",
        save_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=1
       
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    preds = trainer.predict(test_dataset).predictions
    preds = np.argmax(preds,axis=1)

    #generate submission file

    submission = pd.DataFrame()







