#!/usr/bin/env python
# coding: utf-8

# In[47]:


# A sentence sample: You can replace this to any sentence you want
sent = "Steve Jobs 手上拿著一隻 iphone，好潮ㄛ，我的褲子都濕ㄌ！"


# In[ ]:


# Try:
#   extracting English from the the text.
#   extracting Mandarin from the text.
#   spliting the sentences in a large corpus based on some marks.


# In[ ]:


# Example 1: Catch all English words
import re
pattern = re.compile('[A-Za-z]')
results = pattern.finditer(sent)

en_words = []
for result in results:
    print(result.group(), result.span())
    en_words.append(result.group())

sent_new = " ".join(en_words)
print("\n" + sent_new)


# In[ ]:


# Let's define a function that shows the results given regx rule and sentence.
def show_results(rule, sent):
    pattern = re.compile(rule)
    results = pattern.finditer(sent)
    words = []
    for result in results:
        print(result.group(), result.span())
        words.append(result.group())

    sent_new = " ".join(words)
    print("\nNew Sentence: " + sent_new)
    return


# In[ ]:


# Try to add a plus, and continuous matches will be treat as one match
show_results("[A-Za-z]+", sent)


# In[48]:


# Example 2: Catch all Mandarin characters using not-english logic
show_results("[^A-Za-z]+", sent)


# In[49]:


# Example 2: Catch all Mandarin characters using mystic rules found on Internet
show_results("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+", sent)


# In[ ]:


# Example3: Sentence segmentation using regular expression
corpora = """
正規表示式（英語：Regular Expression，常簡寫為regex、regexp或RE），又稱正規表達式、正規表示法、規則運算式、常規表示法，是電腦科學的一個概念。
正規表示式使用單個字串來描述、符合一系列符合某個句法規則的字串。在很多文字編輯器裡，正則表達式通常被用來檢索、替換那些符合某個模式的文字。
許多程式設計語言都支援利用正則表達式進行字串操作。例如，在Perl中就內建了一個功能強大的正則表達式引擎!正則表達式這個概念最初是由Unix中的工具軟體（例如sed和grep）普及開的。'
"""
show_results("[^。!]+", corpora)


# In[ ]:


# Practice: write a Regex that extract Bopomofo
your_regex_rule = ""
show_results(your_regex_rule, sent)


# In[ ]:




