"""
https://huggingface.co/docs/transformers/tasks/language_modeling

"""
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

print("Preparing dataset...")
eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()
eli5["train"][0]

# For MLM, train from roberta family. And for CLM, train from gpt family.
# tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
plm_path_or_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(plm_path_or_name)
model = AutoModelForCausalLM.from_pretrained(plm_path_or_name)

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Initializing the trainer, and start training...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

#save a copy of history models according to the output directory
trainer.save_model() 

