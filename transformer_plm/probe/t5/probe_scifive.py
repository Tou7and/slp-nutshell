"""
Model card:
    https://huggingface.co/razent/SciFive-large-Pubmed_PMC

2022.12.23
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC").to("cuda")

def ask(sentence):
    # sentence = "Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor ."
    text =  sentence + " </s>"

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        early_stopping=True
    )

    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(line)

