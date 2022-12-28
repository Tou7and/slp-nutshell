from transformers import pipeline

generator = pipeline(
    'text-generation',
    model='/home/t36668/projects/icd-transformers/recipes/cmai_gpt2/v1a/exp/cmai-gpt2-v1a',
    device="cuda:0")

def ask(text):
    x = generator(text, do_sample=True, min_length=20, max_length=500)
    print(x)

