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

# Results (Micro F1)

## Logistic Regression
```
# BOW
- unigram: 0.8639
- bigram: 0.8964
- bigram, df>10: 0.8933
- trigram, df>10: 0.8975

# TF-IDF
- unigram: 0.8831
- bigram: 0.8862
- bigram, df>10: 0.8963
- trigram, df>10: 0.8969
```

## GNB model
```
# BOW
- unigram: 0.5724
- bigram, df>10: 0.7996
- trigram, df>100: 0.8522

# TF-IDF
- unigram: 0.5737
- bigram, df>10: 0.7944
- bigram, df>100: 0.8588
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

