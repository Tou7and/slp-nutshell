from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
# from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained("microsoft/biogpt").to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/biogpt")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)

def ask(text):
    results = generator(text, max_length=20, num_return_sequences=5, do_sample=True)
    print(results)

