""" Prepare IMDb dataset as HuggingFace dataset format.

Reference: https://huggingface.co/docs/datasets/loading

JSON: 
{
  "version": "0.1.0",
   "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
           {"a": 4, "b": -5.5, "c": null, "d": true}]
}

LOAD:
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_file.json", field="data")

2022.11.14, JamesH.
"""
import os
import random
import json
import numpy as np
from tqdm import tqdm

def prepare_text_as_json(imdb_data_path, n_subset=5000):
    # imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_data = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in tqdm(sorted(os.listdir(train_path))):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    tmp_text = f.read()
                tmp_label = 0 if category == 'neg' else 1
                train_data.append({'text': tmp_text, 'label': tmp_label})

    # Load the validation data.
    test_data = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in tqdm(sorted(os.listdir(test_path))):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    tmp_text = f.read()
                tmp_label = 0 if category == 'neg' else 1
                test_data.append({'text': tmp_text, 'label': tmp_label})

    seed = 123
    random.seed(seed)
    random.shuffle(train_data)
    random.seed(seed)
    random.shuffle(test_data)

    # Subset to reduce computation cost
    train_data = train_data[:n_subset]
    test_data = test_data[:n_subset]

    json_body_train = {"version": "IMDb_sub5000_train", "data": train_data}
    json_body_test = {"version": "IMDb_sub5000_test", "data": test_data}
    return json_body_train, json_body_test

def dump_json_files():
    body_train, body_test = prepare_text_as_json("../../../exp/aclImdb")

    with open('exp/train.json', 'w') as writer:
        json.dump(body_train, writer, indent=4)

    with open('exp/test.json', 'w') as writer:
        json.dump(body_test, writer, indent=4)
    return 0

def load_from_json():
    from datasets import load_dataset
    datafiles = {'train': 'exp/train.json', 'test': 'exp/test.json'}
    dataset = load_dataset("json", data_files=datafiles, cache_dir="exp/cache", field="data")
    print(dataset['train'][10])
    print(dataset['test'][10])
    return dataset

if __name__ == "__main__":
    # dump_json_files()
    load_from_json()

