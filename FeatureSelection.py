#!/usr/bin/env python
# coding: utf-8

# In[3]:


import Preprocess
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


# In[4]:


#Bag of words
countV = CountVectorizer()
train_count = countV.fit_transform(Preprocess.train_news['Statement'].values)
print(countV)
print(train_count)


# In[5]:


def get_countVectorizer_stats():
    train_count.shape
    print(countV.vocabulary_)
    printf(countV.get_feature_names()[:25])

#get_countVectorizer_stats()


# In[6]:


#tf-idf
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)


# In[7]:


def get_tfidf_stats():
    train_tfidf.shape
    print(train_tfidf.A[:10])

#get_tfidf_stats()


# In[8]:


#bag of words - with n-grams
countV_ngram = CountVectorizer(ngram_range=(1,3),stop_words='english')
tfidf_ngram  = TfidfTransformer(use_idf=True,smooth_idf=True)


# In[9]:


tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)


# In[12]:


#POS Tagging
tagged_sentences = nltk.corpus.treebank.tagged_sents()

cutoff = int(0.75 * len(tagged_sentences))
training_sentences = Preprocess.train_news['Statement']
print(training_sentences)


# In[13]:


#training POS tagger based on words
def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


# In[14]:


#function to strip tags from tagged corpus
def untag(tagged_sentence):
    return [w for w,t in tagged_sentence]


# In[15]:


#using Word2Vec
with open("glove.6B.100d.txt","rb") as lines:
   w2v = {line.split()[0]: np.array(map(float,line.split()[1:])) for line in lines}


# In[16]:


#model = Word2Vec(x, size=100) # x be tokenized text
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))


# In[17]:


class MeanEmbeddingVectorizer(object):
    def __init__(self,word2vec):
        self.word2vec = word2vec
        self.length = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# In[18]:


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




