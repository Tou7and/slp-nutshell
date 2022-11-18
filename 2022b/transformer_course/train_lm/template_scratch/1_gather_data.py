"""
In the course, they use a filter function for extracting wanted text.
Another good way to do so is use regular expression.
"""
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset

def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False

def quick_demo():
    filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
    example_1 = "import numpy as np"
    example_2 = "import pandas as pd"

    print(
        any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters)
    )
    return 0

def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)

def main():
    split = "train"  # "valid"
    filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

    data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
    filtered_data = filter_streaming_dataset(data, filters)

if __name__ == "__main__":
    main()
