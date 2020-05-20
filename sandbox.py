import numpy as np
import pandas as pd

from functools import partial
import itertools
from sklearn.model_selection import train_test_split

import tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy


# Global variables
random_state = 42
nuc = ['A', 'T', 'G', 'C']
dinuc = itertools.product(nuc, repeat = 2)
dinuc =[i[0]+i[1] for i in dinuc]

# Metrics
metrics = [accuracy_score, precision_score, recall_score, f1_score]
metrics_name =[i.__name__ for i in metrics]
metrics = [accuracy_score, partial(precision_score,zero_division=0), recall_score, f1_score]

# #Read in data
df = pd.read_csv('exercise_data/C_elegans_acc_seq.csv', header=None, names =['label', 'seq'])
y = df.label
X = df.seq

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = random_state)

print("xtrain:", x_train)
print("X seq:", X)

random_state = 42


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


def nn_conv_sigmoid():
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
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))
    return model


# Todo: implement this model

class conv_block(layers.Layer):
    def __init__(self):
        super(conv_block, self).__init__()
        l2_reg = tf.keras.regularizers.l2
        self.block = tf.keras.models.Sequential([
            layers.Conv1D(filters=32,
                          kernel_size=5,
                          activation=tf.nn.relu,
                          kernel_regularizer=l2_reg(0.001)),
            layers.MaxPool1D(4, 2),
            ])

    def call(self, x):
        return self.block(x)

class rnn_block(layers.Layer):
    def __init__(self, n_nodes=128):
        super(rnn_block, self).__init__()
        self.block = layers.LSTM(n_nodes)

    def call(self, x):
        print(x.shape)
        return self.block(x)

class ffl_block(layers.Layer):
    def __init__(self, n_nodes=256):
        super(ffl_block, self).__init__()
        l2_reg = tf.keras.regularizers.l2
        self.block = tf.keras.models.Sequential([
            layers.Dense(units=n_nodes,
                         activation=tf.nn.relu,
                         kernel_regularizer=l2_reg(0.001)),
            layers.Dropout(rate=0.5),
        ])

    def call(self, x):
        return self.block(x)


def init_conv_net(n_conv=2, n_ffl=3, n_nodes=1024):
    model = tf.keras.models.Sequential()
    for c in range(n_conv):
        model.add(conv_block())
    model.add(layers.Flatten())
    for f in range(n_ffl):
        model.add(ffl_block(int(n_nodes / (f+1))))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    return model


def init_rnn_net(n_ffl=3, n_nodes=512):
    model = tf.keras.models.Sequential()
    model.add(rnn_block(64)) # if you want to add more use return_sequences=True and add another lstm layer on top of that
    # model.add(layers.Flatten())
    for f in range(n_ffl):
        model.add(ffl_block(int(n_nodes / (f+1))))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    return model


# # NN
x_prep, y_prep = prep_data_nn(x_train, y_train)

print("prepared x:",x_prep)
print("prepared y:",y_prep)
print(x_prep.shape)
print(y_prep.shape)

BATCH_SIZE = 10
EPOCHS=3

"""
model = nn_sequential([256, 256, 64])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_prep, y_prep, batch_size=BATCH_SIZE, epochs=EPOCHS)

x_prep, y_prep = prep_data_nn(x_test, y_test)
pred = model.predict(x_prep)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
#Compute scores
score = [m(y_prep, pred) for m in metrics]

#%%
"""
# ## CNN
x_prep, y_prep = prep_data_cnn(x_train, y_train)
print("cnn input shape:", x_prep.shape)
print("cnn output shape:", y_prep.shape)


BATCH_SIZE = 10
EPOCHS=30

cnn_model = init_conv_net()
rnn_model = init_rnn_net()


print("TRAINING CNN MODEL:")

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])

# cnn_model.fit(x_prep, y_prep, batch_size=BATCH_SIZE, epochs=EPOCHS)
print("TRAINING RNN MODEL:")

rnn_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])
rnn_model.fit(x_prep, y_prep, batch_size=BATCH_SIZE, epochs=EPOCHS)
pred = rnn_model.predict(x_prep)
pred = np.argmax(pred, axis=1)
#Compute scores
score = [m(y_prep, pred) for m in metrics]
print(score)

#%%
