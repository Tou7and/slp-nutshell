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

# Dialogue Records 
TODO: 
- 留下幾個比較有鑑別力的問題即可
- 或許可以先拿 Google Search 的第一個答案作為 Ground-truth, 算 ROGUE 分數作為Benchmark指標

## "Who is the greatest emperor in the history, Shi Huangdi or Tiberius? Give the rationale before answering."
- FLAN-T5-XL: ['Tiberius was the Roman Emperor from AD 79 to AD 89. Shi Huangdi was the Emperor of the Qin Dynasty from AD 279 to AD 286. So the final answer is Tiberius.']

## "Who is the greatest emperor in the history? Give the rationale before answering."
- FLAN-T5-XL: ['Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the']

## "What is the biggest threat to America, and what kind of damage can it make?"
- FLAN-T5-XL: ['a nuclear attack can cause tens of thousands of deaths and billions of dollars in damage.']

## "What is the differnece between CPU and GPU? Give the rationale before answering."
- FLAN-T5-XL: ['The difference between CPU and GPU is the amount of processing power. The difference between CPU and GPU is the amount of memory. The difference between CPU and GPU is the amount of memory. So the final answer is memory.']

## "Which one is more delicious, beef or chicken? Give the rationale before answering."
- FLAN-T5-XL: ['Beef is a protein that is high in iron and zinc. Chicken is a protein that is low in iron and zinc. So the final answer is chicken.']

## "Which one is more nutritious, beef or chicken? Give the rationale before answering."
- FLAN-T5-XL: ['Beef is a source of protein and iron. Chicken is a source of protein and iron. So the final answer is chicken.']


