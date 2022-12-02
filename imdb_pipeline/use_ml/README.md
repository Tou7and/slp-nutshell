# Results of ML
Use Micro F1 as main metric to compute scores. <br>
All experiments use 5000 random selected samples.

We borrow the data prepare snippets from:
- [Google Guides for Text Classification](https://developers.google.com/machine-learning/guides/text-classification)
- [Google Guides for Text Classification - github links](https://github.com/google/eng-edu/tree/main/ml/guides/text_classification)

## GNB model
```
# min-df=1 
- BOW: 0.5852

# set min-df to 100
- BOW: 0.7862
- BOW bigram: 0.8064
- TFIDF bigram: 0.8238
```

## Logistic Regression
```
# min-df=1 
- BOW: 0.8434

# Fix min-df to 100
- BOW: 0.8204
- BOW bigram: 0.8184
- TFIDF bigram: 0.8459
```

## KNN model
```
# Different n-gram, with min-df=100, K=100, cosine distance
- BOW n1:  0.6584
- BOW n2:  0.6548
- BOW n3:  0.6514
- TFIDF n1:  0.7234
- TFIDF n2:  0.7142
- TFIDF n3:  0.7088

# Different distance
- BOW cosine:  0.6584
- BOW euclidean:  0.6316
- BOW manhattan:  0.6264
- TFIDF cosine:  0.7234
- TFIDF euclidean:  0.7234
- TFIDF manhattan:  0.6098

# Different neighbors, with min-df=100, cosine distance, TFIDF unigram
- K=10:  0.6888
- K=30:  0.7169
- K=100:  0.7234
- K=300:  0.7126
```



# References
- [Google Guides for Text Classification](https://developers.google.com/machine-learning/guides/text-classification)
- [Google Guides for Text Classification - github links](https://github.com/google/eng-edu/tree/main/ml/guides/text_classification)
- [IMDb Dataset information](http://ai.stanford.edu/~amaas/data/sentiment/) 
- [IMDb Dataset URL](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- [Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (pp. 142-150).](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
  - The corresponding paper of IMDB dataset

