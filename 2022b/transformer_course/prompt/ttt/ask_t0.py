"""
Ask T0.

https://huggingface.co/bigscience/T0pp
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
# model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B").to("cuda")

def ask_plm(text="Is this review positive or negative? Review: this is the best cast you will ever buy"):
    """
    Try to ask T0.

    Returns:
        complete_texts(list): list of output sentences.
    """
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)

    text_0 = tokenizer.decode(outputs[0])

    # to get 5 results:
    # outputs = model.generate(input_ids.to("cuda"), num_return_sequences=5, num_beams=5)
    # results = tokenizer.batch_decode(outputs)
    # TODO: more testing and API survey
    return [text_0]
    
if __name__ == "__main__":
    # ask_plm()
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    # ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
    # ask_plm("Dante was born in [MASK].")
    # ask_plm("Dante Alighieri was born in [MASK].")
    # ask_plm("Taiwan is a [MASK].")
    # ask_plm("Spiderman's lover is [MASK].")
    # results = ask_plm("When you catch a cold, you should <mask>.")
    # results = ask_plm("When you catch a cold, you should see a <mask>.")
    results = ask_plm("When you catch a cold, what should you do?")
    print(results)
    results = ask_plm("When you catch a cold, the drug to use is what?")
    print(results)
