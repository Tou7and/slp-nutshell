from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from prepare import load_imdb_sentiment_analysis_dataset

def train_logistic_regression():
    train_examples, test_examples = load_imdb_sentiment_analysis_dataset("./exp/")
    train_texts = train_examples[0]
    train_labels = train_examples[1]
    test_texts = test_examples[0]
    test_labels = test_examples[1]
    # print(train_texts[:2])
    # print(train_labels[:2])

    # Build bag-of-words vectorizer
    vectorizer = TfidfVectorizer()
    # vectorizer = CountVectorizer()
    vectorizer.fit(train_texts)

    # Turn training text into vectors
    train_feat = vectorizer.transform(train_texts)
    test_feat = vectorizer.transform(test_texts) # select only 3 samples

    # Training
    clf = LogisticRegression(random_state=0)
    clf.fit(train_feat, train_labels)

    # print(clf.predict(test_feat))
    # print(test_labels[:3])
    # print(test_texts[:3])
    test_preds = clf.predict(test_feat)
    print(f1_score(test_labels, test_preds))
    return 

def train_transformer_models():
    train_examples, test_examples = load_imdb_sentiment_analysis_dataset("./exp/")

    return 

if __name__ == "__main__":
    train_logistic_regression()

