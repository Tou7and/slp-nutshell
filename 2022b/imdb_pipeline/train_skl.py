""" Train sentiment classification models with scikit-learn.

2022.09.20, James H.
"""
import sys
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from prepare import load_imdb_sentiment_analysis_dataset

def train_skl_pipeline(embedding_type="bow", max_ngram=1, regex=r"\b\w\w+\b", model_type='Gaussian Naive Bayes'):
    """ Default: BOW(unigram) plus Gaussian Naive Bayes.
    """
    train_examples, test_examples = load_imdb_sentiment_analysis_dataset("./exp/")
    train_texts = train_examples[0]
    train_labels = train_examples[1]
    test_texts = test_examples[0]
    test_labels = test_examples[1]
    # print(train_texts[:2])
    # print(train_labels[:2])

    if embedding_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, max_ngram), token_pattern=regex, min_df=1)
    elif embedding_type == 'bow':
        vectorizer = CountVectorizer(ngram_range=(1, max_ngram), token_pattern=regex, min_df=1)
    else:
        print("This Method is not defined. Please specify `bow` or `tfidf`.")
        sys.exit(0)
    vectorizer.fit(train_texts)

    # Turn training text into vectors
    train_feat = vectorizer.transform(train_texts)
    test_feat = vectorizer.transform(test_texts) # select only 3 samples

    # Training
    if model_type == 'Logistic Regression':
        clf = LogisticRegression(random_state=0)
        clf.fit(train_feat, train_labels)
    elif model_type == 'Gaussian Naive Bayes':
        clf = GaussianNB()
        clf.fit(train_feat.toarray(), train_labels)
    else:
        print("This Method is not defined. Please specify `Logistic Regression` or `Gaussian Naive Bayes`.")
        sys.exit(0)
    
    # print(clf.predict(test_feat))
    # print(test_labels[:3])
    # print(test_texts[:3])

    # Testing
    test_preds = clf.predict(test_feat)

    print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
    print("Precision: ", precision_score(test_labels, test_preds, average="micro"))
    print("Recall: ", recall_score(test_labels, test_preds, average="micro"))
    # print("Macro F1: ", f1_score(test_labels, test_preds, average="macro"))
    print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))
    return 



if __name__ == "__main__":
    train_skl_pipeline()
    print("="*100)
    train_skl_pipeline(model_type="Logistic Regression")
    print("="*100)
    train_skl_pipeline(model_type="Transformer")

