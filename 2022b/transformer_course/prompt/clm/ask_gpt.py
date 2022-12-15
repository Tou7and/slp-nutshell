"""
Ask GPT-2, pytorch version.

Reference:
    https://huggingface.co/docs/transformers/model_doc/gpt2
    https://huggingface.co/gpt2

Get Features in pytorch:
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
"""
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model_checkpoint = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
# model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
# 
# def ask_plm_gg(text="Hello, my dog is cute"):
#     inputs = tokenizer(text, return_tensors="pt")
#     outputs = model(**inputs, labels=inputs["input_ids"])
#     output_ids = torch.argmax(outputs.logits[0], dim=1)
#     text = tokenizer.decode(output_ids)
#     print(text)

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
def ask_plm(text="Hello, my dog is cute"):
    set_seed(42)
    topk_text = generator(text, max_length=30, num_return_sequences=5)
    for tt in topk_text:
        print(tt)

if __name__ == "__main__":
    # ask_plm()
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    # ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
    # ask_plm("Dante was born in [MASK].")
    # ask_plm("Dante Alighieri was born in [MASK].")
    # ask_plm("Taiwan is a [MASK].")
    # ask_plm("Spiderman's lover is [MASK].")
    ask_plm("When you catch a cold, you should ")
    print("-"*30)
    ask_plm("My name is Julien and I like to")
