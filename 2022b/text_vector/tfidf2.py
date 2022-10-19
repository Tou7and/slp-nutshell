from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from preprocess import load_corpus2

def main(bigram=False):
    corpus = load_corpus2()

    # Construct vocabulary from corpus
    vectorizer = CountVectorizer()
    if bigram:
        # vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
    else:
        vectorizer = CountVectorizer()

    tfidf_extractor = TfidfTransformer(smooth_idf=False)

    vectorizer.fit(corpus)
    vocab = vectorizer.get_feature_names_out()
    print(f"Vocab size: {len(vocab)}")

    # Transform corpus into count vector
    count_x = vectorizer.transform(corpus)
    count_global = vectorizer.transform([" ".join(corpus)])
    print("global counts of target words:")
    coordinate  = count_global.tocoo()
    for idx, value in enumerate(coordinate.data):
        sent_id = coordinate.row[idx]
        vocab_id = coordinate.col[idx]
        the_word = vocab[vocab_id]
        if the_word in ['the', 'movie', 'is', 'good']:
            print(the_word, value)

    # Fit Tfidf extractor
    tfidf_extractor.fit(count_x)

    # Trasform to TF-IDF
    tfidf_x = tfidf_extractor.transform(count_x[1]) # id-37: to be or not to be that is the question 
    print(tfidf_x.toarray())

    coordinate  = tfidf_x.tocoo()
    for idx, value in enumerate(coordinate.data):
        sent_id = coordinate.row[idx]
        vocab_id = coordinate.col[idx]
        print(sent_id, vocab_id, vocab[vocab_id], value)
    return

if __name__ == "__main__":
    main()

