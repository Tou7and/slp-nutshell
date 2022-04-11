""" BOW for Text Classification.

Setup:
  pip install scikit-learn==1.0.2

"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB


def tokenize_unigram(text):
    words = list(text)
    return words


def tokenize_bigram(text):
    words = []
    for idx in range(len(text)-2+1):
        words.append(text[idx:idx+2])
    return words


def bag_of_unigram(corpus, apply_tfidf=False):
    # vectorizer = CountVectorizer(stop_words=["，","。"])
    vectorizer = CountVectorizer(tokenizer=tokenize_unigram, stop_words=["，","。"])
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    counts = X.toarray()
    print(counts)

    if apply_tfidf:
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit(counts)
        tfidf = transformer.transform(counts)
        # tfidf = transformer.fit_transform(counts)
        print(tfidf)
    return 


def bag_of_bigram(corpus):
    vectorizer = CountVectorizer(tokenizer=tokenize_bigram, stop_words=["，","。"])
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    print(X.toarray())
    return

def bag_of_bigram_sk(corpus):
    """ Bag of bigram with scikit-learn
    if ngram_range = (1, 2), it will count both unigram and bigram.
    """
    # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize_unigram) # token_pattern=r'\b\w+\b', min_df=1
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), tokenizer=tokenize_unigram) # token_pattern=r'\b\w+\b', min_df=1
    X = bigram_vectorizer.fit_transform(corpus)
    print(bigram_vectorizer.get_feature_names_out())
    print(X.toarray())
    return


def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    print(X.toarray())
    return


if __name__ == "__main__":
    corpus = [
        "唧唧復唧唧",
        "木蘭當戶織",
        "不聞機杼聲",
        "惟聞女嘆息",
        "問女何所思",
        "問女何所憶" 
    ]
    bag_of_bigram(corpus)

