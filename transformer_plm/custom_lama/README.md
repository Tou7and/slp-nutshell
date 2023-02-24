# Custom LAMA
Try to write a custom LAMA for probing the knowledge of LLM.

# Snippet
```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def ask(text):
    # inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=150)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

