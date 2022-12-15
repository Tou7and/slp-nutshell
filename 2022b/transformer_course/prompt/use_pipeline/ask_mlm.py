from transformers import pipeline

# model_checkpoint = "distilbert-base-uncased"
model_checkpoint = "bert-base-uncased"
# model_checkpoint = "exp/bert-mlm-imdb"
# model_checkpoint = "exp/bert-wwm-imdb"

mask_filler = pipeline("fill-mask", model=model_checkpoint)

def ask_plm(text="This is a great [MASK]."):
    preds = mask_filler(text)
    for pred in preds:
        print(f">>> {pred['sequence']}")
    return

if __name__ == "__main__":
    # ask_plm()
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    # ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
    # ask_plm("Dante was born in [MASK].")
    # ask_plm("Dante Alighieri was born in [MASK].")
    # ask_plm("Taiwan is a [MASK].")
    # ask_plm("Spiderman's lover is [MASK].")
    # ask_plm("When you catch a cold, you should [MASK].")
    ask_plm("When you catch a cold, you should see a [MASK].")
