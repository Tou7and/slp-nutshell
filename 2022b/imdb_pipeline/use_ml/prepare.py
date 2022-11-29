import os
import random
import numpy as np

def load_imdb_sentiment_analysis_dataset(data_path, n_subset=5000):
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.

    # Returns
        A tuple of training(25k) and validation(25k) data.

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    seed = 123
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)
    random.seed(seed)
    random.shuffle(test_texts)
    random.seed(seed)
    random.shuffle(test_labels)

    # Subset to reduce computation cost
    train_texts = train_texts[:n_subset]
    train_labels = train_labels[:n_subset]
    test_texts = test_texts[:n_subset]
    test_labels = test_labels[:n_subset]

    return train_texts, np.array(train_labels), test_texts, np.array(test_labels)

if __name__ == "__main__":
    train_texts, train_labels, test_texts, text_labels = load_imdb_sentiment_analysis_dataset("./exp/")
    print(train_texts[:2])
    print(train_labels[:10])
    print(train_labels.shape)
