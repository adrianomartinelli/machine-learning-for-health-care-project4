#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:06:41 2020

@author: adrianomartinelli
"""


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
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

# # Basline NN
def nn_sequential(hidden, output=1, hidden_activation='relu',output_activation='sigmoid'):
    
    model = Sequential()
    model.add(Dense(hidden[0], input_shape = (82,), activation=hidden_activation))
    
    for i in hidden[1:]:
        model.add(Dense(i, activation=hidden_activation))
        
    model.add(Dense(output, activation=output_activation))
    return model

# # Convolutional NN
def nn_conv():
    model = Sequential()
    
    model.add(Conv1D(filters = 32,
                     kernel_size = 3,
                     strides = 1,
                     padding = 'valid',
                     activation = 'relu'))
    
    model.add(MaxPooling1D(4, 2))
    
    model.add(Conv1D(filters = 16,
                     kernel_size = 3,
                     strides = 1,
                     padding = 'valid',
                     activation = 'relu'))
    
    model.add(MaxPooling1D(4, 2))
    
    #Dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu')),
    model.add(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))
    
    return model

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

#%%

# ## CNN
x_prep, y_prep = prep_data_cnn(x_train, y_train)

BATCH_SIZE = 10
EPOCHS=3

model = nn_conv()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])

model.fit(x_prep, y_prep, batch_size=BATCH_SIZE, epochs=EPOCHS)

x_prep, y_prep = prep_data_cnn(x_test, y_test)
pred = model.predict(x_prep)
pred = np.argmax(pred, axis=1)
#Compute scores
score = [m(y_prep, pred) for m in metrics]



#%%

# ## Experimental Grid-Search visualisation
import pandas
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns

res = results['svc']
tmp = res.loc['params']
df = pd.Series()
for i in tmp:
    df = pd.concat([df,pd.Series(i)], axis = 1)

df = df.iloc[:,1:]
df = df.T

df['C'] = df['C'].astype(np.float32)
df = df.drop(['random_state'], axis = 1)

d = dict()
for k in df.select_dtypes('object').columns.tolist():
    d[k] = {j:i for i,j in enumerate(df[k].unique())}

df = df.replace(d)

df.C = df.C.apply(lambda x: np.log10(x))
df['f1']= res.loc['f1_score'].values
df['setting']= res.loc['setting'].values

parallel_coordinates(df, 'setting')
plt.show()


#%%

tbl = pd.concat(results, axis=1)
idx = list(tbl.columns)
idx = [i[0] for i in idx]
tbl.columns = range(tbl.shape[1])

s = pd.DataFrame(idx, columns =['model'])
s
tmp = pd.concat([tbl, s.T], axis = 0)

tmp.groupby(['model', 'setting'])
tmp = tmp.T

cols = ['setting', 'accuracy_score', 'precision_score', 'recall_score','f1_score']
for i in cols:
    tmp[i]=tmp[i].astype('float32')

cols = ['model'] + cols
tmp[cols].groupby(['model', 'setting']).agg([min,max, "mean", np.std])


tt = tmp[cols].melt(id_vars=['model', 'setting'])

#%%
sns.boxplot(x = 'variable', y='value', data =tt, hue='model')