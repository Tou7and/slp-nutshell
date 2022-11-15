""" Training pipeline using Pytorch.

Refer: https://huggingface.co/docs/transformers/training

Results:
    {'accuracy': 0.9062}

2022.11.14, JamesH.
"""
import os
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(data_folder="exp"):
    print("Loading dataset...")
    datafiles = {
        'train': os.path.join(data_folder, 'train.json'), 
        'test': os.path.join(data_folder, 'test.json')}
    dataset = load_dataset("json", data_files=datafiles, cache_dir="exp/cache", field="data")

    print("Prepare tokenizer and tokenized dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Modification for Pytorch training
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    print("preparing dataset...")
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]


    print("preparing data loader...")
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
    
    print("prepare training...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    print("training...")
    model.to(DEVICE)
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print("evaluating...")
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print("eval results:")
    res = metric.compute()
    print(data_folder, res)
    return 

if __name__ == "__main__":
    main(data_folder="../exp/few_n0016")
    main(data_folder="../exp/few_n0080")
    main(data_folder="../exp/normal_n0400")
