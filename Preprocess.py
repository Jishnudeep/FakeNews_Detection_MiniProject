
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
from matplotlib import pyplot as plt



train_news = pd.read_csv("train.csv")
test_news = pd.read_csv("test.csv")
#valid_news = pd.read_csv("valid.csv")
train_news.head()





def data_obs():
    print("training dataset size:")
    print(train_news.shape)
    print(train_news.head(10))

    #below dataset were used for testing and validation purposes
    print(test_news.shape)
    print(test_news.head(10))
    
    #print(valid_news.shape)
    #print(valid_news.head(10))





def create_distribution(dataFile):
    
    sb.countplot(x="Label",data=dataFile)
    




create_distribution(train_news)
create_distribution(test_news)
#create_distribution(valid_news)




def data_qualityCheck():
    
    print("Checking data qualitites...")
    train_news.isnull().sum()
    train_news.info()
        
    print("check finished.")

    #below datasets were used to 
    test_news.isnull().sum()
    test_news.info()

    #valid_news.isnull().sum()
    #valid_news.info()





data_qualityCheck()




#Stemming
eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))





def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed





#Data processing





def process_data(data, exclude_stopword=True, stem=True):
    tokens = [w for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed





process_data(stem_tokens(stopwords, eng_stemmer))





#creating ngrams
#unigram
def create_unigram(words):
    assert type(words) == list
    return words

#bigram
def create_bigram(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len-1):
            for j in range(1, skip+2):
                if i+ j < Len:
                    lst.append(join_str.join([words[i], words[i+j]]))
    else:
    #set it as unigram
        lst = create_unigram(words)
    return lst

#trigrams
def create_trigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 2:
        lst = []
        for i in range(1, skip+2):
            for j1 in range(1, skip+2):
                for j2 in range(1, skip+2):
                    if i+j1 < Len and i+j1+j2 < Len:
                        lst.append(join_str.join([words[i], words[i+j1+j2]]))
    else:
        #set is as bigram
        lst = create_bigram(words)





porter = PorterStemmer()





def tokenizer(text):
    return text.split()





def tokenizer_porter(text):
    return [porter.stem(word.lower()) for word in text.split()]





#doc = ['runners like running and thus they run','this is a test for tokens']
#tokenizer([[word for word in test_news.iloc[:,1]] for word in line.lower().split()])






