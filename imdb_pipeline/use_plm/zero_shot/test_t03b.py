from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from tqdm import tqdm
from anwer_map import answer_mapping

def test_zero_shot():
    # Load model
    print("Loading model into memory...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B").to("cuda")

    # Load function for creating prompt template
    print("Loading dataset into memory...")
    imdb_prompts = DatasetTemplates("imdb")
    prompt_fuc = imdb_prompts.templates['02ff2949-0f45-4d97-941e-6fa4c0afbc2d']

    imdb = load_dataset("imdb")
    test_set = imdb['test']

    ref_list = []
    hyp_list = []
    # Convert example into prompt template
    for example in tqdm(test_set):
        # print(example)
        example_pt = prompt_fuc.apply(example)
        # print(example_pt[0])
        # print(example_pt[1])
        inputs = tokenizer.encode(example_pt[0], return_tensors="pt").to("cuda")
        
        outputs = model.generate(inputs)
        output_text = tokenizer.decode(outputs[0])
        output_label = answer_mapping(output_text)

        # print("text:", example_pt[0])
        ref_list.append(example_pt[1])
        hyp_list.append(output_label)

    print(ref_list)
    print(hyp_list)
    acc = sum([int(i==j) for i,j in zip(ref_list, hyp_list)])/len(ref_list)
    print("ACC:", acc)
    return acc

if __name__ == "__main__":
    test_zero_shot()
