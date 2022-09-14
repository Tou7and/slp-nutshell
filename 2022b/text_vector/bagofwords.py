import re
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import load_clean_corpus, load_complete_corpus

def main(bigram=False):
    # corpus = load_clean_corpus()
    corpus = load_complete_corpus()

    # Construct vocabulary from corpus
    if bigram:
        # vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
    else:
        vectorizer = CountVectorizer()

    vectorizer.fit(corpus)
    vocab = vectorizer.get_feature_names_out()
    print(f"Vocab size: {len(vocab)}\n")
    print("Top vocabulary: ")
    sorted_vocab = sorted(vectorizer.vocabulary_.items()) 
    for top_words in sorted_vocab[:5]:
        print(top_words[0], top_words[1])

    # Transform corpus into BOW vector
    X = vectorizer.transform(corpus[37:38]) # pick the Hamlet

    print(f"\nShape of bag-of-words array: {X.shape}\n")
    coordinate  = X.tocoo()
    for idx, value in enumerate(coordinate.data):
        sent_id = coordinate.row[idx]
        vocab_id = coordinate.col[idx]
        print(sent_id, vocab_id, vocab[vocab_id], value)

    return

if __name__ == "__main__":
    main(bigram=True)

