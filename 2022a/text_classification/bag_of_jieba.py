""" BOW for Text Classification.

Setup:
  pip install scikit-learn==1.0.2
  pip install jieba

Reference:
  https://investigate.ai/text-analysis/how-to-make-scikit-learn-natural-language-processing-work-with-japanese-chinese/

"""
import jieba

def tokenize_jieba(text):
    words = jieba.lcut(text)
    return words

