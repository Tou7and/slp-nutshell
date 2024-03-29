"""
MLM tuning from existing PLM.

Reference:
    https://huggingface.co/course/chapter7/3?fw=pt

DataCollator source codes:
    https://agithub.com/huggingface/transformers/blob/v4.25.1/src/transformers/data/data_collator.py

2022.12.8, JamesH.
"""
import collections
import math
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# model_checkpoint = "distilbert-base-uncased"
model_checkpoint = "bert-base-uncased"
model_output_path = "exp/bert-mlm-imdb"
imdb_dataset = load_dataset("imdb")

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def show_sample():
    sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
# features: ['attention_mask', 'input_ids', 'word_ids']
# print(tokenizer.model_max_length)

def group_texts(examples, chunk_size=128):
    """
    Note that using a small chunk size can be detrimental in real-world scenarios.
    One should use a size that corresponds to the use case to apply the model to.
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Now, it's time to insert the MASK into training data
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Use a smaller set for super quick experiment!
# train_size = 10000 # 10000
# test_size = int(0.1 * train_size)

# downsampled_dataset = lm_datasets["train"].train_test_split(
#    train_size=train_size, test_size=test_size, seed=42)

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(lm_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=model_output_path,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
)

print("Check perp before training, then do the training...")
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

print("Check perp after training:")
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# trainer.save_model()
model.save_pretrained(model_output_path)
tokenizer.save_pretrained(model_output_path)

