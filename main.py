#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:18:08 2020

@author: adrianomartinelli
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import re
import itertools
from functools import partial

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid 
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

import tensorflow as tf
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
def central_as(x):
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


#%%

# #Read in data
df = pd.read_csv('exercise_data/C_elegans_acc_seq.csv', header=None, names =['label', 'seq'])
y = df.label
X = df.seq

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

# #Further models to test
# * Naive Bayes
# * Logistic Regression


#%%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def prep_data_nn(X,y):
    le = LabelEncoder()
    le.fit(nuc)
    X = X.apply(lambda x: le.transform([i for i in x]))
    X = pd.DataFrame([*X])
    
    y[y==-1] = 0
    
    return X.values, y.values

def prep_data_cnn(X,y):
    ohe = OneHotEncoder()
    ohe.fit(np.array(nuc).reshape(-1,1))
    
    #Split sequence in list of letters
    tmp = X.apply(lambda x: [i for i in x])
    
    #Convert to array
    tmp = np.array([*tmp])
    
    #Encode each sequence (2D array) and store as list of arrays
    tmp = [ohe.transform(i.reshape(-1,1)).A for i in tmp]
    
    #Reshape to prepare for concat, i.e. create shape (1,82,4)
    tmp = [i.reshape(1,-1,4) for i in tmp]
    
    #Concat along axis=0
    tmp = np.concatenate(tmp, axis=0)
    
    y[y==-1] = 0
    
    return tmp, y.values

#%%
# # Basline NN
def nn_sequential(hidden, output=1, hidden_activation='relu',output_activation='sigmoid'):
    
    model = Sequential()
    model.add(Dense(hidden[0], input_shape = (82,), activation=hidden_activation))
    
    for i in hidden[1:]:
        model.add(Dense(i, activation=hidden_activation))
        
    model.add(Dense(output, activation=output_activation))
    return model

# # Convolutional NN

#%%
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy

# # NN
x_prep, y_prep = prep_data_nn(x_train, y_train)

BATCH_SIZE = 10
EPOCHS=3

model = nn_sequential([128,64,32])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_prep, y_prep, batch_size=BATCH_SIZE, epochs=EPOCHS)

x_prep, y_prep = prep_data_nn(x_test, y_test)
pred = model.predict(x_prep)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
#Compute scores
score = [m(y_prep, pred) for m in metrics]




