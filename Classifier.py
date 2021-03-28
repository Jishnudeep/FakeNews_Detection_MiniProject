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
build_confusion_matrix(random_forest)


"""So far we have used bag of words technique to extract the features and 
passed those featuers into classifiers. We have also seen the
f1 scores of these classifiers. now lets enhance these 
features using term frequency weights with various n-grams
"""

##Now using n-grams
#naive-bayes classifier
nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(Preprocess.test_news['Statement'])
np.mean(predicted_nb_ngram == Preprocess.test_news['Label'])


#logistic regression classifier
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',FeatureSelection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(Preprocess.test_news['Statement'])
np.mean(predicted_LogR_ngram == Preprocess.test_news['Label'])


#random forest classifier
random_forest_ngram = Pipeline([
        ('rf_tfidf',FeatureSelection.tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_rf_ngram = random_forest_ngram.predict(Preprocess.test_news['Statement'])
np.mean(predicted_rf_ngram == Preprocess.test_news['Label'])


#K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline_ngram)
build_confusion_matrix(logR_pipeline_ngram)
build_confusion_matrix(random_forest_ngram)


print(classification_report(Preprocess.test_news['Label'], predicted_nb_ngram))
print(classification_report(Preprocess.test_news['Label'], predicted_LogR_ngram))
print(classification_report(Preprocess.test_news['Label'], predicted_rf_ngram))

Preprocess.test_news['Label'].shape

"""
Out of all the models fitted, we would take 2 best performing model.  we would call them 
candidate models from the confusion matrix, we can see that random forest 
and logistic regression are best performing in terms of precision and recall 
(take a look into false positive and true negative counts which appeares
to be low compared to rest of the models)
"""

#grid-search parameter optimization
#random forest classifier parameters
parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'rf_tfidf__use_idf': (True, False),
               'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
}

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(Preprocess.train_news['Statement'][:10000],Preprocess.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'LogR_tfidf__use_idf': (True, False),
               'LogR_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(Preprocess.train_news['Statement'][:10000],Preprocess.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_


#by running above commands we can find the model with best performing parameters


#running both random forest and logistic regression models again with best parameter found with GridSearch method
random_forest_final = Pipeline([
        ('rf_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=10))
        ])
    
random_forest_final.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_rf_final = random_forest_final.predict(Preprocess.test_news['Statement'])
np.mean(predicted_rf_final == Preprocess.test_news['Label'])
print(classification_report(Preprocess.test_news['Label'], predicted_rf_final))

logR_pipeline_final = Pipeline([
        #('LogRCV',countV_ngram),
        ('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_final.fit(Preprocess.train_news['Statement'],Preprocess.train_news['Label'])
predicted_LogR_final = logR_pipeline_final.predict(Preprocess.test_news['Statement'])
np.mean(predicted_LogR_final == Preprocess.test_news['Label'])
#accuracy = 0.62
print(classification_report(Preprocess.test_news['Label'], predicted_LogR_final))


"""
by running both random forest and logistic regression with GridSearch's best parameter estimation, we found that for random 
forest model with n-gram has better accuracty than with the parameter estimated. The logistic regression model with best parameter 
has almost similar performance as n-gram model so logistic regression will be out choice of model for prediction.
"""
