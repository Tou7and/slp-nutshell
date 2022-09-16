import re

def only_english(text):
    text = text.lower()
    pattern = re.compile("[a-z]+")
    words = pattern.findall(text)
    return " ".join(words)

def load_clean_corpus():
    with open("corpus.txt", 'r') as reader:
        corpus = reader.readlines()

    short_corpus = corpus[37:40]

    # Clean corpus
    clean_corpus = []
    for line in short_corpus:
        clean_corpus.append(only_english(line))
    return clean_corpus

def load_complete_corpus():
    with open("corpus.txt", 'r') as reader:
        corpus = reader.readlines()

    # Clean corpus
    clean_corpus = []
    for line in corpus:
        clean_corpus.append(only_english(line))
    return clean_corpus

if __name__ == "__main__":
    with open("corpus.txt", 'r') as reader:
        corpus = reader.readlines()
    short_corpus = corpus[37:40]
    for line in short_corpus:
        print(line)
        print("-"*100)
        print(only_english(line))
        print("*"*100)

