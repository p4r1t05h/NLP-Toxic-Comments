
# coding: utf-8

# In[3]:


import os
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[4]:


os.chdir("D:\Data Science\Project 1")


# In[5]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')


# In[6]:


train.head()


# In[7]:


lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()


# In[8]:


#Creating the list of Labels to predict
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()


# In[9]:


#checking the length of Training and Testing Data set
len(train),len(test)


# In[10]:


#since the training and testing dataset are not equal, we'll try to make them equal by removing empty comment
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# In[11]:


len(train),len(test)


# In[12]:


#Creating Bag of Words & Term Document Matrix with TfIdf Method 
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[13]:


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


# In[14]:


trn_term_doc, test_term_doc


# In[16]:


#NAIVE BAYES Feature Selection
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[17]:


x = trn_term_doc
test_x = test_term_doc


# In[18]:


#Fitting a Model with 1 Dependency at a Time
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[19]:


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


# In[20]:


#Creating Submission File
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

