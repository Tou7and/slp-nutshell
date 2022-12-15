"""
Ask BERT questions in a MLM style.

Can this work as a quick sanity check: 
    - to see if the model already contain the knowledge we want it to know
    - to see if the domain adaption or knowledge injection work

Reference:
    https://huggingface.co/course/chapter7/3?fw=pt
"""
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

# model_checkpoint = "distilbert-base-uncased"
model_checkpoint = "bert-base-uncased"
# model_checkpoint = "exp/bert-mlm-imdb"
# model_checkpoint = "exp/bert-wwm-imdb"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def ask_plm(text="This is a great [MASK]."):
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

def ask_multimask(text="This is a great [MASK] [MASK]."):
    """
    Try to predict the mask in a sentence.

    Returns:
        complete_texts(list): list of output sentences.
    """
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs[0]
    # print(predictions)
    sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
    for k in range(10):
        predicted_index = [sorted_idx[i, k].item() for i in range(0,24)]
        predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1,24)]
        print(predicted_token)


if __name__ == "__main__":
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    # ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
    # ask_plm("Dante was born in [MASK].")
    # ask_plm("Dante Alighieri was born in [MASK].")
    # ask_plm("Taiwan is a [MASK].")
    # ask_plm("Spiderman's lover is [MASK].")
    results = ask_plm("When you catch a cold, you should [MASK].")
    # results = ask_plm("When you catch a cold, you should see a [MASK].")
    for rr in results:
        print(rr)
