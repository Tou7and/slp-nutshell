from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B").to("cuda")

def ask(text):
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs, num_beams=10, 
        num_return_sequences=3, min_length=30, max_length=100)
    
    print(tokenizer.batch_decode(outputs))


