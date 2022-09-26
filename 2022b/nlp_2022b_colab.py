# -*- coding: utf-8 -*-
"""nlp_2022b.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y6LJVq6JXY1KEG74cyX59W5JqeOI43yc
"""

!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zvxf aclImdb_v1.tar.gz

!ls ./aclImdb

!ls aclImdb/train/neg | wc -l

import os
import sys
import random
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Set up the path
data_path = os.getcwd()
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

seed = 123
random.seed(seed)
random.shuffle(train_texts)
random.seed(seed)
random.shuffle(train_labels)

train_texts = train_texts[:5000]
train_labels = train_labels[:5000]

seed = 123
random.seed(seed)
random.shuffle(test_texts)
random.seed(seed)
random.shuffle(test_labels)

test_texts = test_texts[:5000]
test_labels = test_labels[:5000]

vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
vectorizer.fit(train_texts)

train_feat = vectorizer.transform(train_texts)
test_feat = vectorizer.transform(test_texts)

print(train_feat.shape)
print(test_feat.shape)
print(len(vectorizer.vocabulary_))

list(vectorizer.vocabulary_)[:10]

clf = GaussianNB()
clf.fit(train_feat.toarray(), train_labels)

test_preds = clf.predict(test_feat.toarray())
test_proba = clf.predict_proba(test_feat.toarray())

# Use some metrics to score the performance of the mode on testing data 
print("ROC AUC: ", roc_auc_score(test_labels, test_proba[:,1]))
print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))

vectorizer2 = CountVectorizer(ngram_range=(1, 2), min_df=100)
vectorizer2.fit(train_texts)

train_feat = vectorizer2.transform(train_texts)
test_feat = vectorizer2.transform(test_texts)

clf2 = GaussianNB()
clf2.fit(train_feat.toarray(), train_labels)

test_preds = clf2.predict(test_feat.toarray())
test_proba = clf2.predict_proba(test_feat.toarray())

# Use some metrics to score the performance of the mode on testing data 
print("ROC AUC: ", roc_auc_score(test_labels, test_proba[:,1]))
print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))

vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=100)
vectorizer3.fit(train_texts)

train_feat = vectorizer3.transform(train_texts)
test_feat = vectorizer3.transform(test_texts)

clf3 = GaussianNB()
clf3.fit(train_feat.toarray(), train_labels)

test_preds = clf3.predict(test_feat.toarray())
test_proba = clf3.predict_proba(test_feat.toarray())

# Use some metrics to score the performance of the mode on testing data 
print("ROC AUC: ", roc_auc_score(test_labels, test_proba[:,1]))
print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))

def predict_function(x):
    test_feat = vectorizer2.transform(x)
    test_preds = clf3.predict(test_feat.toarray())
    test_proba = clf3.predict_proba(test_feat.toarray())
    return test_proba

# import shap
masker = shap.maskers.Text(r"\.")
explainer = shap.Explainer(predict_function, masker, output_names=['neg', 'pos'])



shap_values = explainer(test_texts[:3])
shap_values = explainer(test_texts[:3])
shap.plots.text(shap_values)

shap_values = explainer(test_texts[-3:])
shap.plots.text(shap_values)

test_texts[:3]

shap.plots.text(shap_values)

list(vectorizer3.vocabulary_)[-10:]

error_idx = []
for idx, preds in enumerate(test_preds):
  if preds != test_labels[idx]:
    error_idx.append(idx)
print(len(error_idx))
print(error_idx[:3])

shap_values = explainer([test_texts[20]])
shap.plots.text(shap_values)

print(test_preds[20], test_labels[20])

vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=100)
vectorizer3.fit(train_texts)

train_feat = vectorizer3.transform(train_texts)
test_feat = vectorizer3.transform(test_texts)

clf_lr = LogisticRegression(random_state=0, max_iter=200)
clf_lr.fit(train_feat, train_labels)

test_preds = clf_lr.predict(test_feat.toarray())
test_proba = clf_lr.predict_proba(test_feat.toarray())

# Use some metrics to score the performance of the mode on testing data 
print("ROC AUC: ", roc_auc_score(test_labels, test_proba[:,1]))
print("F1 Score: ", f1_score(test_labels, test_preds, average="micro"))
print("Confuse Matrix: ", confusion_matrix(test_labels, test_preds))

def predict_lr(x):
    test_feat = vectorizer3.transform(x)
    test_preds = clf_lr.predict(test_feat.toarray())
    test_proba = clf_lr.predict_proba(test_feat.toarray())
    return test_proba

# import shap
masker2 = shap.maskers.Text(r"\.")
explainer2 = shap.Explainer(predict_lr, masker, output_names=['neg', 'pos'])

shap_values2 = explainer2([test_texts[20]])
shap.plots.text(shap_values2)

print(test_preds[20], test_labels[20])

print(test_proba[20])

error_idx = []
for idx, preds in enumerate(test_preds):
  if preds != test_labels[idx]:
    error_idx.append(idx)
print(len(error_idx))
print(error_idx[:3])