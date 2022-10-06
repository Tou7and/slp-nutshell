import pickle
from prepare import load_imdb_sentiment_analysis_dataset

# Load vectorizer
with open("exp/knn/vectorizer.sav", 'rb') as reader:
    vectorizer = pickle.load(reader)

# Load model
with open("exp/knn/model.sav", 'rb') as reader:
    model = pickle.load(reader)

sample = "I love this movie a lot."
feat = vectorizer.transform([sample])

def infer():
    print("Proba(neg, pos):")
    print(model.predict_proba(feat))
    return

def check_neighbors():
    k_neighbors = model.kneighbors(feat, 5, return_distance=False)
    print("ID of its neighbors: ", k_neighbors)
    
    train_texts, train_labels, test_texts, test_labels = load_imdb_sentiment_analysis_dataset("./exp/")
    for x in k_neighbors[0]:
        print("-"*100)
        print(train_texts[x])
    return

def reset_params_and_infer():
    model.set_params(n_neighbors=5)
    print("Proba(neg, pos):")
    print(model.predict_proba(feat))
    return

if __name__ == "__main__":
    infer()
    reset_params_and_infer()

