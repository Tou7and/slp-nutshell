"""
Ask RoBerta questions in a MLM style.

RoBerta use different special tokens compared to BERT.
(For example, Roberta use <mask> and BERT use [MASK])

Clinical-Longformer is trained upon Longformer,
and Longformer is a mutation of RoBerta.

Reference:
    https://huggingface.co/course/chapter7/3?fw=pt
"""
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

model_checkpoint = "yikuan8/Clinical-Longformer"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def ask_plm(text="This is a great <mask>."):
    """
    Try to predict the mask in a sentence.

    Returns:
        complete_texts(list): list of output sentences.
    """
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits

    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    complete_texts = []
    for token in top_5_tokens:
        complete_t = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
        complete_texts.append(complete_t)
    return complete_texts

if __name__ == "__main__":
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    # ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
    # ask_plm("Dante was born in [MASK].")
    # ask_plm("Dante Alighieri was born in [MASK].")
    # ask_plm("Taiwan is a [MASK].")
    # ask_plm("Spiderman's lover is [MASK].")
    # results = ask_plm("When you catch a cold, you should <mask>.")
    # results = ask_plm("When you catch a cold, you should see a <mask>.")
    results = ask_plm("When you catch a cold, the drugs you should use are <mask>.")
    for rr in results:
        print(rr)
