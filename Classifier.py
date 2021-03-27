# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:46:29 2021

@author: bjish
"""
#Importing all the libraries, Preprocess and Feature Selection
import Preprocess
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


#Building different models to find which model to use

#first we will use bag of words techniques

#building classifier using naive bayes 
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_nb = nb_pipeline.predict(Preprocess.test_news['Statement'])
np.mean(predicted_nb == Preprocess.test_news['Label'])

#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',FeatureSelection.countV),
        ('LogR_clf',LogisticRegression(random_state = 0, max_iter = 500))
        ])

logR_pipeline.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_LogR = logR_pipeline.predict(Preprocess.test_news['Statement'])
np.mean(predicted_LogR == Preprocess.test_news['Label'])


#random forest
random_forest = Pipeline([
        ('rfCV',FeatureSelection.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
random_forest.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_rf = random_forest.predict(Preprocess.test_news['Statement'])
np.mean(predicted_rf == Preprocess.test_news['Label'])

def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(Preprocess.train_news):
        train_text = Preprocess.train_news.iloc[train_ind]['Statement'] 
        train_y = Preprocess.train_news.iloc[train_ind]['Label']
    
        test_text = Preprocess.train_news.iloc[test_ind]['Statement']
        test_y = Preprocess.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(Preprocess.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
#K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline)
build_confusion_matrix(logR_pipeline)

