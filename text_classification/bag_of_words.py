from sklearn.feature_extraction.text import CountVectorizer

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


sample = "I love banana, orange and banana!"

sample_vec = vectorizer.transform([sample]).toarray()
print(sample_vec)

for idx, val in enumerate(sample_vec[0]):
    if val >= 1:
        print(word_list[idx], val)


