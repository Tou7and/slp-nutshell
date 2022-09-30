""" Train sentiment classification models with scikit-learn neighbor modules.

2022.09.28, James H.
"""
import sys
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from prepare import load_imdb_sentiment_analysis_dataset

def train_knn(embedding_type="bow", max_ngram=1, regex=r"\b\w\w+\b", min_freq=1):
    """ Default: BOW(unigram) plus Gaussian Naive Bayes.
    """
    train_texts, train_labels, test_texts, test_labels = load_imdb_sentiment_analysis_dataset("./exp/")

    # Convert text to vectors
    if embedding_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, max_ngram), token_pattern=regex, min_df=min_freq)
    elif embedding_type == 'bow':
        vectorizer = CountVectorizer(ngram_range=(1, max_ngram), token_pattern=regex, min_df=min_freq)
    else:
        print("This Method is not defined. Please specify `bow` or `tfidf`.")
        sys.exit(0)
    vectorizer.fit(train_texts)
    train_feat = vectorizer.transform(train_texts)
    test_feat = vectorizer.transform(test_texts) # select only 3 samples

    # Training and testing
    clf = KNeighborsClassifier(n_neighbors=30, metric="cosine")
    # clf = KNeighborsClassifier(n_neighbors=10, metric="manhattan")
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    clf.fit(train_feat, train_labels)
    test_preds = clf.predict(test_feat)

    # Show metrics
    print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
    # print("Precision: ", precision_score(test_labels, test_preds, average="micro"))
    # print("Recall: ", recall_score(test_labels, test_preds, average="micro"))
    # print("Macro F1: ", f1_score(test_labels, test_preds, average="macro"))
    # print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))

    # print(clf.predict_proba(test_feat[1:2, :]))
    return 

if __name__ == "__main__":
    print("- BOW: ")
    train_knn(min_freq=100)

    print("- BOW bigram: ")
    train_knn(max_ngram=2, min_freq=100)

    print("- TFIDF bigram: ")
    train_knn(embedding_type="tfidf", max_ngram=2, min_freq=100)


