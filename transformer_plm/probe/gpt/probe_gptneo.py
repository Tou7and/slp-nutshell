"""
Model card:
    https://huggingface.co/EleutherAI/gpt-neo-125M

"""

from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

def ask(text):
    x = generator(text, do_sample=True, min_length=20, max_length=300)
    print(x)

