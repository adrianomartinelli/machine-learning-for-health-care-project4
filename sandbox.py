import numpy as np
import pandas as pd

from functools import partial
import itertools

import tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight

# Global variables
random_state = 42
nuc = ['A', 'T', 'G', 'C']
dinuc = itertools.product(nuc, repeat = 2)
dinuc =[i[0]+i[1] for i in dinuc]


# Metrics
metrics = [accuracy_score, precision_score, recall_score, f1_score]
metrics_name =[i.__name__ for i in metrics]
metrics = [accuracy_score, partial(precision_score,zero_division=0), recall_score, f1_score]


# Helper Functions
def central_as(x):
    ''' Finde the most central AG dinucleotide'''
    acc = [i.start() for i in re.finditer('AG', x)]
    dis = [abs(i-len(x)/2) for i in acc]
    return acc[np.argmin(dis)]
    

# Features engineering
def prep_data(X,y):
    '''Engineer features'''
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


# One-hot encoding 
def prep_data_cnn(X,y=None, use_y=True):
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
    if use_y:
        y[y==-1] = 0
        return tmp, y.values
    else:
        return tmp


# Basline NN (only applies a linear layer -> not used in our analysis)
def nn_sequential(hidden, output=1, hidden_activation='relu',output_activation='sigmoid'):
    model = Sequential()
    model.add(Dense(hidden[0], input_shape = (82,), activation=hidden_activation))
    for i in hidden[1:]:
        model.add(Dense(i, activation=hidden_activation))
    model.add(Dense(output, activation=output_activation))
    return model


# Convolutional NN referred to as 'cnn2' in our notebooks
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


# cnn2 with a sigmoid output layer for 0-1 loss (compared to categorical crossentropy -> did not use this)
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


# Convolutional Layer used in 'cnn' model
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

    
# Recurrent (LSTM) Layer used in 'rnn' model
class rnn_block(layers.Layer):
    def __init__(self, n_nodes=128):
        super(rnn_block, self).__init__()
        self.block = layers.LSTM(n_nodes)

    def call(self, x):
        return self.block(x)

    
# Feedforward Layer used in both 'cnn' and 'rnn' model
class ffl_block(layers.Layer):
    def __init__(self, n_nodes):
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


# Initialize 'cnn' model
def init_conv_net(n_conv=2, n_ffl=3, n_nodes=256):
    model = tf.keras.models.Sequential()
    for c in range(n_conv):
        model.add(conv_block())
    model.add(layers.Flatten())
    for f in range(n_ffl):
        model.add(ffl_block(int(n_nodes / (f+1))))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    return model


# Initialize 'rnn' model
def init_rnn_net(n_ffl=3, n_nodes=256):
    model = tf.keras.models.Sequential()
    model.add(rnn_block(32)) # if you want to add more use return_sequences=True and add another lstm layer on top of that
    # model.add(layers.Flatten())
    for f in range(n_ffl):
        model.add(ffl_block(int(n_nodes / (f+1))))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    return model


# Used to flexibly call model architecture based on 'model_type'
def init_model(model_type='cnn', lr=1e-3):
    if model_type == 'cnn':
        model = init_conv_net()
    elif model_type == 'cnn2':
        model = nn_conv()
    elif model_type == 'rnn':
        model = init_conv_net()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])
    return model


# Crossvalidation function (takes as input training data, a list of settings testing data (with (x_test&y_test) and /wo (x_testh) labels, the crossvalidationsplit ~ 5, number of epochs (default 30) and batch_size.
def cross_validation(x_train, y_train, settings, x_test=None, y_test=None, x_testh=None, k=5, epochs=30, batch_size=8,
                    test_hidden=True):
    results = []
    models = []
    kf = KFold(n_splits=k)
    run = -1
    x = np.asarray(x_train)
    y = np.asarray(y_train)
    f = sum(y)
    tot = len(y)
    t = tot - f
    wt = (1 / t) * (tot) / 2.0
    wf = (1 / f) * (tot) / 2.0
    class_weights = {0: wt, 1: wf}
    # ca. 341
    print("Weights for classes:", class_weights)
    for s in settings:
        for train_idx, val_idx in kf.split(x, y):
            model = init_model(model_type=s[0], lr=s[1])
            train_x = x[train_idx]
            train_y = y[train_idx]
            val_x = x[val_idx]
            val_y = y[val_idx]
            hist = model.fit(train_x, train_y, epochs=epochs, batch_size=s[2], validation_data=(val_x, val_y),
                             class_weight = class_weights)
            acc = hist.history['val_loss'][-1]  # final validation loss
            run += 1
            if test_hidden:
                # Test set with labels
                pred_ = model.predict(x_test)
                pred = np.argmax(pred_, axis=1)
                score = [m(y_test, pred) for m in metrics]
                # Test set hidden labels
                predh_ = model.predict(x_testh)
                predh = np.argmax(predh_, axis=1)
                results.append([run, s, acc, pred_, score, predh_])
            else:
                # Test set with labels
                pred_ = model.predict(x_test)
                pred = np.argmax(pred_, axis=1)
                score = [m(y_test, pred) for m in metrics]
                results.append([run, s, acc, pred_, score])
            del model
    return results


# Run test set with models | not in use anymore
def test_run(models, x_test, y_test=None):
    output = []
    for model in models:
        score = None
        if y_test != None:
            pred = model.predict(x_test)
            pred = np.argmax(pred, axis=1)
            predh = model.predict(x_testh)
            predh = np.argmax(predh, axis=1)
            score = [m(y_test, pred) for m in metrics]  # loss on the testing set
            output.append(predh, pred, score)
        else:
            pred = model.predict(x_test)
            pred = np.argmax(pred, axis=1)
            output.append(pred)
    return output


# Plotting function for AUC and ROC
def plotting_results(cv):
    y_prop = nb.predict_proba(x)
    y_prop =y_prop[:,1]
    roc =roc_curve(y_test, y_prop)

    label = 'AUC: {:.4}'.format(auc(roc[0], roc[1]))
    plt.plot(roc[0], roc[1], label = label)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',label='Random', alpha=.8)
    plt.legend()
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    #PR curve
    prc = plot_precision_recall_curve(nb, x, y)
    prc.ax_.set_title('Precision-Recall Curve')
    


