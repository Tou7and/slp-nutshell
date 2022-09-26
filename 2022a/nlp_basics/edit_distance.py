#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Edit Distance
# https://pypi.org/project/edit-distance/

string_1 = "瑪莉有隻小綿羊"
string_2 = "我有一隻小毛驢"


# In[3]:


# Install a pip package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install edit-distance')


# In[26]:


import edit_distance
ref = list(string_1)
hyp = list(string_2)
sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
codes = sm.get_opcodes()

n_sub = 0
n_ins = 0
n_del = 0
n_eql = 0
for code in codes:
    if code[0] == 'insert':
        n_ins +=1
    elif code[0] == 'replace':
        n_sub +=1
    elif code[0] == 'delete':
        n_del +=1
    else:
        n_eql +=1
print("sub:",n_sub)
print("ins:",n_ins)
print("del:",n_del)
print("eql:",n_eql)

print()
dist = n_sub+n_ins+n_del
print("edit distance:", dist)


# In[29]:


# let's wrap it as a fucntion
def get_edit_distance(ref, hyp):
    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
    codes = sm.get_opcodes()
    n_sub = 0
    n_ins = 0
    n_del = 0
    n_eql = 0
    for code in codes:
        if code[0] == 'insert':
            n_ins +=1
        elif code[0] == 'replace':
            n_sub +=1
        elif code[0] == 'delete':
            n_del +=1
        else:
            n_eql +=1
    return n_sub+n_ins+n_del


# In[30]:


get_edit_distance("吃葡萄不吐葡萄皮", "不吃葡萄倒吐葡萄皮")


# In[ ]:




