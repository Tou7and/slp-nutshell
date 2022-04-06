""" Play with CountVectorizer
"""
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def build_count_vectorizer():
    vectorizer = CountVectorizer()

    corpus = [
        'this is an apple',
        'apple is sweet and orange is sour.',
        'banana is yellow and apple is red',
        'grape is both sweet and sour',
    ]

    vectorizer.fit(corpus)

    word_list = vectorizer.get_feature_names_out()
    print(word_list)
    print(vectorizer.vocabulary_)

    # Try to encode using this vectorizer
    sample = "I love banana, orange and banana!"
    sample_vec = vectorizer.transform([sample]).toarray()
    print(sample_vec)
    for idx, val in enumerate(sample_vec[0]):
        if val >= 1:
            print(word_list[idx], val)
    return vectorizer 

def fit_with_new_data():
    """ Fit with new data?
    """
    vectorizer = build_count_vectorizer()

    print("-"*10, "fit_with_new_data", "-"*10)
    # Fit some new data
    old_vocab = vectorizer.vocabulary_
    vectorizer = CountVectorizer(vocabulary=old_vocab)

    corpus_2 = [
        'this is an chicken',
        'chickens have two legs and pigs have four.',
    ]
    vectorizer.fit(corpus_2)
    word_list = vectorizer.get_feature_names_out()
    print(word_list)
    print(vectorizer.vocabulary_)

    sample = "I love banana, orange and banana, and I love chicken!"
    sample_vec = vectorizer.transform([sample]).toarray()

    print(sample_vec)
    for idx, val in enumerate(sample_vec[0]):
        if val >= 1:
            print(word_list[idx], val)

def fit_with_new_data_v2():
    """ Fit with new data?
    """
    vectorizer = build_count_vectorizer()
    old_word_list = vectorizer.get_feature_names_out().tolist()

    print("-"*10, "fit_with_new_data", "-"*10)
    corpus_2 = [
        'this is an chicken',
        'chickens have two legs and pigs have four.',
    ]
    vectorizer.fit(corpus_2)
    new_word_list = vectorizer.get_feature_names_out().tolist()

    final_word_list = old_word_list + new_word_list
    word_counter = Counter(final_word_list)
    final_vocab = dict()
    for idx, val in enumerate(list(word_counter)):
        final_vocab[val] = idx

    vectorizer = CountVectorizer(vocabulary=final_vocab)
    word_list = vectorizer.get_feature_names_out()
    sample = "I love banana, orange and banana, and I love chicken!"
    sample_vec = vectorizer.transform([sample]).toarray()
    print(sample_vec)
    for idx, val in enumerate(sample_vec[0]):
        if val >= 1:
            print(word_list[idx], val)



if __name__ == "__main__":
    fit_with_new_data_v2()


