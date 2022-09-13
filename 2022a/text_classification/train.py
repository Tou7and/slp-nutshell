import os
import re
import numpy as np
from glob import glob
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from bag_of_ngrams import tokenize_unigram, tokenize_bigram


def keep_mandarin(sent):
    pattern_zh = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]')
    results = pattern_zh.finditer(sent)

    zh_chars = []
    for result in results:
        # print(result.group(), result.span())
        zh_chars.append(result.group())
    sent_new = "".join(zh_chars)
    return sent_new


def load_data_from_folder(folder):
    text_files = glob(
        os.path.join(folder, "*.txt")
    )
    
    corpus = []
    # only take the last 3 sents
    for text_file in text_files:
        with open(text_file, 'r') as reader:
            lines = reader.readlines()
        text = "".join(lines[-3:])
        text = text.replace("\n", "").replace(" ", "")
        corpus.append(text)
    return corpus


# /Users/mac/projects/public/ce2022/UdicOpenData/udicOpenData/Facebook/Negative/蔡英文/negative.txt
# /Users/mac/projects/public/ce2022/UdicOpenData/udicOpenData/Facebook/Positive/蔡英文粉專/positive.txt
def load_data_from_file(text_file):
    with open(text_file, 'r') as reader:
        lines = reader.readlines()
    
    corpus = []
    for line in lines:
        corpus.append(keep_mandarin(line))
    return corpus


def load_sentiment_data(pos_folder, neg_folder):
    """ Return corpus and corresponding labels """
    pos_data = load_data_fro_folder(pos_folder)
    neg_data = load_data_fro_folder(neg_folder)

    corpus = pos_data + neg_data
    labels = ["pos"]*len(pos_data) + ["neg"]*len(neg_data)
    return corpus, labels


def load_sentiment_data_from_file(pos_file, neg_file):
    """ Return corpus and corresponding labels """
    pos_data = load_data_from_file(pos_file)
    neg_data = load_data_from_file(neg_file)

    # pos_train = pos_data[:len(pos_data)-100]
    pos_train = pos_data[:150]
    pos_test = pos_data[-100:]
    # neg_train = neg_data[:len(neg_data)-100]
    neg_train = neg_data[:150]
    neg_test = neg_data[-100:]

    corpus_train = pos_train + neg_train
    labels_train  = ["pos"]*len(pos_train) + ["neg"]*len(neg_train)
    corpus_test = pos_test + neg_test
    labels_test  = ["pos"]*len(pos_test) + ["neg"]*len(neg_test)

    dataset = {
        "train": (corpus_train, labels_train),
        "test": (corpus_test, labels_test),
    }
    return dataset


def train_bow(cla="dctree", feat="bigram", verbose=0, min_df=2):
    # Load Data!
    # corpus, labels = load_sentiment_data("./ptt_movie_clean/train/pos/", "./ptt_movie_clean/train/neg/")
    dataset = load_sentiment_data_from_file(
        "./data/facebook_tsai/positive.txt",
        "./data/facebook_tsai/negative.txt"
    )

    train_corpus, train_labels = dataset["train"]
    test_corpus, test_labels = dataset["test"]

    if feat == "bigram":
        vectorizer = CountVectorizer(tokenizer=tokenize_bigram, stop_words=["，","。", "\n", " "], min_df=min_df)
    elif feat == "bigram+":
        vectorizer = CountVectorizer(tokenizer=tokenize_unigram, stop_words=["，","。", "\n", " "], ngram_range=(1, 2), min_df=min_df)
    elif feat == "trigram":
        vectorizer = CountVectorizer(tokenizer=tokenize_bigram, stop_words=["，","。", "\n", " "], min_df=min_df, ngram_range=(1, 3))
    else:
        vectorizer = CountVectorizer(tokenizer=tokenize_unigram, stop_words=["，","。", "\n", " "], min_df=min_df)

    counts = vectorizer.fit_transform(train_corpus).toarray()

    if verbose > 0:
        print(vectorizer.get_feature_names_out())
        print(counts)

    if cla == "dctree":
        clf = DecisionTreeClassifier(random_state=0)
    elif cla == "rf":
        clf = RandomForestClassifier(random_state=0)
    elif cla == "ada":
        clf = AdaBoostClassifier(random_state=0)
    elif cla == "svm":
        clf = svm.SVC(gamma=0.1)
    else:
        clf = GaussianNB()
    clf.fit(counts, train_labels)

    test_counts = vectorizer.transform(test_corpus).toarray()
    y_pred = []
    for kk in test_corpus:
        kk_counts = vectorizer.transform([kk]).toarray()
        y_pred.append(clf.predict(kk_counts)[0])

    print(accuracy_score(y_pred, test_labels))
    if verbose > 0:
        print(test_labels)
        print(y_pred)

    return vectorizer, clf


def train_tfidf(cla="gnb", feat="bigram"):
    dataset = load_sentiment_data_from_file(
        "./data/facebook_tsai/positive.txt",
        "./data/facebook_tsai/negative.txt"
    )

    train_corpus, train_labels = dataset["train"]
    test_corpus, test_labels = dataset["test"]

    vectorizer = CountVectorizer(tokenizer=tokenize_bigram, stop_words=["，","。", "\n", " "])

    counts = vectorizer.fit_transform(train_corpus).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit(counts)
    tfidf = transformer.transform(counts).toarray()

    if cla == "dctree":
        clf = DecisionTreeClassifier(random_state=0)
    elif cla == "svm":
        clf = svm.SVC() # gamma=0.5
    else:
        clf = GaussianNB()

    clf.fit(tfidf, train_labels)

    print("testing")
    
    test_counts = vectorizer.transform(test_corpus).toarray()
    test_tfidf = transformer.transform(test_counts).toarray()
    y_pred = []
    for kk in test_corpus:
        kk_counts = vectorizer.transform([kk]).toarray()
        kk_tfidf = transformer.transform(kk_counts).toarray()
        y_pred.append(clf.predict(kk_tfidf)[0])

    print(accuracy_score(y_pred, test_labels))
    return vectorizer, transformer, clf


def export_tree(clf, vectorizer, output_file):
    feat_names = vectorizer.get_feature_names_out().tolist()
    text_representation = tree.export_text(clf, feature_names=feat_names, max_depth=30) # 20

    with open(output_file, "w") as writer:
        writer.write(text_representation)
    return


if __name__ == "__main__":
    samples = [
        "政府實在過於無能",
        "政府很有效率",
        "阿不就好棒棒",
        "索尼罪大惡極 百姓怨聲載道",
    ]

    # train_tfidf_gnb()
    vectorizer, clf = train_bow(feat="unigram", cla="dctree")

    # dump tree as text
    export_tree(clf, vectorizer, "./tmp/unigram.txt")

    for sample in samples:
        print(sample, clf.predict(vectorizer.transform([sample]).toarray()))

    vectorizer, clf = train_bow(feat="bigram", cla="dctree")

    # dump tree as text
    export_tree(clf, vectorizer, "./tmp/bigram.txt")

    for sample in samples:
        print(sample, clf.predict(vectorizer.transform([sample]).toarray()))

    vectorizer, clf = train_bow(feat="unigram", cla="ada")
    vectorizer, clf = train_bow(feat="unigram", cla="rf")


