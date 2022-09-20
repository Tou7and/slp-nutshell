from random import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from prepare import load_imdb_sentiment_analysis_dataset

def test_with_different_metrics():
    print("Loading dataset into memory...")
    train_examples, test_examples = load_imdb_sentiment_analysis_dataset("./exp/")
    # The dataset contain 25000 samples, with 12500 positive and 12500 negative labels samples

    # Take the first 90 samples, which is all label 1(positive)
    # Then take the last 10 samples, which is all label 0(negative)
    test_texts = test_examples[0][:90] + test_examples[0][-10:]
    test_labels = list(test_examples[1][:90]) + list(test_examples[1][-10:])

    # Suppose the model tend to predict negative
    test_preds = [1]*1 + [0]*99
    shuffle(test_preds)

    print("Micro F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
    print("Precision: ", precision_score(test_labels, test_preds, average="micro"))
    print("Recall: ", recall_score(test_labels, test_preds, average="micro"))

    print("\nMacro F1 Score: ", f1_score(test_labels, test_preds, average="macro", zero_division=1))
    print("Precision: ", precision_score(test_labels, test_preds, average="macro", zero_division=1))
    print("Recall: ", recall_score(test_labels, test_preds, average="macro", zero_division=1))
    # print("Macro F1: ", f1_score(test_labels, test_preds, average="macro"))

    print("\nF1 per label: ", f1_score(test_labels, test_preds, average=None, zero_division=1))
    print("Precision per label: ", precision_score(test_labels, test_preds, average=None, zero_division=1))
    print("Recall per label: ", recall_score(test_labels, test_preds, average=None, zero_division=1))

    print("\nConfuse Matrix: ", confusion_matrix(test_labels, test_preds))
    return

if __name__ == "__main__":
    test_with_different_metrics()

