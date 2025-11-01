# scripts/train_model.py
"""
Train a text classification model (example with Hugging Face).
Input CSV should have: report_text, urgency (labels as strings)
This script is a template — you must adapt label mapping and hyperparams.
"""
import os
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/triage-distilbert"

def prepare_dataset(csv_path="data/raw/patient_reports.csv", label_column="urgency"):
    df = pd.read_csv(csv_path)
    # Drop rows without label
    df = df.dropna(subset=[label_column])
    # Map labels to ints
    labels = sorted(df[label_column].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    df["label"] = df[label_column].map(label2id)
    # Save mapping
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.Series(label2id).to_json(os.path.join(OUTPUT_DIR, "label_map.json"))
    # Use datasets
    ds = load_dataset("csv", data_files=csv_path)["train"]
    def filter_fn(example):
        return example.get(label_column) is not None
    ds = ds.filter(lambda x: x[label_column] is not None)
    return ds, labels, label2id

def tokenize_batch(examples, tokenizer):
    return tokenizer(examples["report_text"], truncation=True, padding="max_length", max_length=256)

def main():
    ds, labels, label2id = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.map(lambda x: tokenizer(x["report_text"], truncation=True, padding="max_length", max_length=256), batched=True)
    # convert labels according to label2id if needed - depends on dataset structure
    # Create model
    num_labels = len(labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        fp16=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()





























##scripts/train_model.py
#
#"""
#Train a text classification model (example with hugging face).
#Input CSV should have : report_text, urgency (labels as strings)
#This script is a template — you must adapt label mapping and hyperparams.
#"""
#
#import os
#from datasets import load_dataset, ClassLabel
#from transformers import AutoTokenizer,  AutoModelForSequenceClassification, TrainingArguments, Trainer
#import numpy as np
#import pandas as pd
#
#MODEL_NAME = "distilbert-base-uncased"
#OUTPUT_DIR = "model/triage-distilbert"
#
#def prepare_dataset(csv_path="data/raw/patient.csv", label_column="urgency"):
#    df = pd.read_csv(csv_path)
#    #Drop rows without label
#    df= df.dropna(subset=[label_column])
#    #map labels to ints
#    labels = sorted(df[label_column].unique().tolist())
#    label2id = {l:i for i,l in enumerate(labels)}
#    df["label"] = df[label_column. map(label2id)]
#    # save mapping
#    os.makedirs(OUTPUT_DIR,exist_ok=True)
#    pd.Series(label2id).to_json(os.path.join(OUTPUT_DIR, "label_map.json"))
#    #Use datasets
#    ds = load_datasets("csv", data_files=csv_path)["train"]
#    def filter_fn(example):
#        return example.get(label_column) is not None
#    ds = ds.filter(lambda x: x[label_column] is not None)
#    return ds, labels, label2id