#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:39:31 2020

@author: adrianomartinelli
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import re
import itertools
from functools import partial
from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid 
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

#%%

# Global variables
nuc = ['A', 'T', 'G', 'C']
dinuc = itertools.product(nuc, repeat = 2)
dinuc =[i[0]+i[1] for i in dinuc]

random_state = 42

# Metrics
metrics = [accuracy_score, precision_score, recall_score, f1_score]
metrics_name =[i.__name__ for i in metrics]
metrics = [accuracy_score, partial(precision_score,zero_division=0), recall_score, f1_score]

# #Helper Functions
def central_ag(x):
    ''' Finde the most central AG dinucleotide'''
    acc = [i.start() for i in re.finditer('AG', x)]
    dis = [abs(i-len(x)/2) for i in acc]
    return acc[np.argmin(dis)]
    
def prep_data(X,y):
    '''Engineer features'''
    
    # # Features engineering
    nuc_counts = X.apply(lambda x: [x.count(i) for i in nuc])
    dinuc_counts = X.apply(lambda x: [x.count(i) for i in dinuc])
    df = nuc_counts + dinuc_counts
    df = pd.DataFrame([*df], columns = nuc+dinuc)
    
    return df.values, y.values


class Data:
    train = pd.concat([pd.read_csv('exercise_data/human_dna_train_split.csv', header=0, names =['seq', 'label']),
                       pd.read_csv('exercise_data/human_dna_validation_split.csv', header=0, names =['seq', 'label'])])
    test = pd.read_csv('exercise_data/human_dna_test_split.csv', header=0, names =['seq', 'label'])
    test_hidden = pd.read_csv('exercise_data/human_dna_test_split.csv', header=0, names =['seq', 'label'])
#%%

# #Read in data
df = Data()
y = df.train.label
X = df.train.seq
centralAG = X.apply(lambda x: central_ag(x)).value_counts()

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = random_state)

#%%
sns.distplot([len(i) for i in X])

#%%

# Parameter Grid
parameters ={'rf': {'n_estimators': [50,100,300],
                    'n_jobs': [-1],
                    'class_weight': ['balanced', None],
                    'random_state': [random_state]},
             
             'xgboost': {'learning_rate': np.logspace(-2,0,3),
                         'n_jobs': [8],
                         'scale_pos_weight': [1, y_train.value_counts()[-1]/y_train.value_counts()[1]]
                 },
             
             'svc': {'C': np.logspace(-2,2,4),
                     'kernel': ['linear', 'poly', 'rbf'],
                     'class_weight': ['balanced', None],
                     'random_state': [random_state]
                 },
             'nb': {'alpha': np.logspace(-2,1,4),
                    'fit_prior': [True, False],
                    'class_prior': [[.5, .5], None]
                    },
             'logreg': {
                 'penalty': ['l2', 'l1', 'elasticnet'],
                 'class_weight': [None, 'balanced'],
                 'solver': ['saga'],
                 'random_state': [random_state],
                 'n_jobs':[-1]
                 }
    }

#%%

def model_evaluation(model, param_grid, model_name, cv=5):
    
    # # Results, initialise
    res = pd.DataFrame(index= ['setting', 'params', 'split']+metrics_name)

    # # Cross Validation
    # kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    setting = 1
    for params in param_grid:
        split = 1
        for idx_train, idx_val in kf.split(x_prep, y_prep):
            
            #Sample dist
            n_pos_train = np.sum(y_prep[idx_train] == 1)
            n_neg_train = np.sum(y_prep[idx_train] == -1)
            n_pos_val = np.sum(y_prep[idx_val] == 1)
            n_neg_val = np.sum(y_prep[idx_val] == -1)
            
            #Set parameters
            model.set_params(**params)
            
            #Fit model
            model.fit(x_prep[idx_train], y_prep[idx_train])
            
            #Predict on validation set
            pred = model.predict(x_prep[idx_val])
            
            #Compute scores
            score = [m(y_prep[idx_val], pred) for m in metrics]
            
            #Append to result df
            s = [setting, params, split] + score + [n_pos_train, n_neg_train, n_pos_val, n_neg_val]
            s = pd.Series(s, index = ['setting', 'params', 'split']+metrics_name + ['n_pos_train', 'n_neg_train', 'n_pos_val', 'n_neg_val'], name=model_name)
            res = pd.concat([res, s], axis=1)
            split += 1
        
        setting += 1
        
    return res
            
#%%

# # Models
rf = RandomForestClassifier()
xgboost = XGBClassifier()
svc = SVC()
nb = CategoricalNB()
logreg = LogisticRegression()

# # Preprocess data
x_prep, y_prep = prep_data(x_train, y_train)

#
results = dict()

# ## Random Forest
results['rf'] = model_evaluation(rf, list(ParameterGrid(parameters['rf'])), 'rf', cv=5)

# ## xgboost
results['xgboost'] = model_evaluation(xgboost, list(ParameterGrid(parameters['xgboost'])), 'xgboost', cv=5)

# ## SVC
results['svc'] = model_evaluation(svc, list(ParameterGrid(parameters['svc'])), 'svc', cv=5)

# ## Naive Bayes
results['nb'] = model_evaluation(nb, list(ParameterGrid(parameters['nb'])), 'nb', cv=5)

# ## Logistic Regression
results['logreg'] = model_evaluation(logreg, list(ParameterGrid(parameters['logreg'])), 'logreg', cv=5)

