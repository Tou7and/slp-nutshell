# Steps
- Prepare
- Train
- Test
- Deploy

## Download and Unzip
```
cd exp
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -zvxf aclImdb_v1.tar.gz
```

# Results
Use Micro F1 as main metric to compute scores. <br>
All experiments use 5000 random selected samples.

## GNB model
```
- BOW: 0.5852

# set min-df to 100
- BOW: 0.7862
- BOW bigram: 0.8064
- TFIDF bigram: 0.8238
```

## Logistic Regression
```
- BOW: 0.8434

# Fix min-df to 100
- BOW: 0.8204
- BOW bigram: 0.8184
- TFIDF bigram: 0.8459
```

## KNN model
```
Try different metrics, with (min-df = 100, K=10)
# manhattan
- BOW: 0.61
- BOW bigram: 0.6144
- TFIDF bigram: 0.5768

# euclidean
- BOW: 0.5964
- BOW bigram: 0.5988
- TFIDF bigram: 0.6774

# cosine
- BOW: 0.634
- BOW bigram: 0.6288
- TFIDF bigram: 0.6774

Try different number of neighbors, with (min-df = 100, cosine, tfidf-bigram)
- K=10: 0.6774
- K=20: 0.6954
- K=30: 0.7016
```


## Transformer
reference: https://huggingface.co/blog/sentiment-analysis-python

```
TBD
```

# References
- [Google Guides for Text Classification](https://developers.google.com/machine-learning/guides/text-classification)
- [Google Guides for Text Classification - github links](https://github.com/google/eng-edu/tree/main/ml/guides/text_classification)
- [IMDb Dataset information](http://ai.stanford.edu/~amaas/data/sentiment/) 
- [IMDb Dataset URL](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- [Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (pp. 142-150).](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
  - The corresponding paper of IMDB dataset

