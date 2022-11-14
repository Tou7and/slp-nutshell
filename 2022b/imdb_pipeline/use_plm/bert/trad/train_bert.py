""" A simple PLM finetuning pipeline using Transformer and Trainer.

Reference: https://huggingface.co/docs/transformers/training

Results:
    {
      "epoch": 1.0,
      "eval_accuracy": 0.9066,
      "eval_loss": 0.24325241148471832,
      "eval_runtime": 28.69,
      "eval_samples_per_second": 174.277,
      "eval_steps_per_second": 5.472,
      "step": 157
    },
    {
      "epoch": 2.0,
      "eval_accuracy": 0.9114,
      "eval_loss": 0.23791977763175964,
      "eval_runtime": 28.5082,
      "eval_samples_per_second": 175.388,
      "eval_steps_per_second": 5.507,
      "step": 314
    }


2022.11.14, JamesH.
"""
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed

METRIC = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return METRIC.compute(predictions=predictions, references=labels)

def main():
    set_seed(123) # Helper function for reproducible behavior by fixing random

    print("Loading dataset...")
    datafiles = {'train': 'exp/train.json', 'test': 'exp/test.json'}
    dataset = load_dataset("json", data_files=datafiles, cache_dir="exp/cache", field="data")

    print("Prepare tokenizer and tokenized dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_train_dataset = tokenized_datasets["train"]
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"]

    print("Training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2
    )
    # training_args = TrainingArguments(output_dir="exp")
    training_args = TrainingArguments(
        output_dir="exp/bert_imdb5000_mk3",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # 3, 5
        evaluation_strategy="epoch",
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model()
    return model

if __name__ == "__main__":
    main()
