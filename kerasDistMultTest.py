# coding: utf-8

# # Daten einlesen
#
# Hier werden die Daten eingelesen, dabei werden die Veränderungstypen ignoriert. Parameter:
#
# - `train_file`, die Trainingsdaten
# - `test_file`, die Testdaten
# - `embeddings`, die Embeddings
#
# Anmerkungen Bodo: mal randfälle konstruieren und betrachten, wie gut das funktioniert.
#

# In[33]:
from __future__ import print_function


train_file = 'changes0910DETAILSDIST2REVERSED.dat'
test_file = 'changes1011DETAILSDIST2REVERSED.dat'
embedding_file = './res/embedding_2009_distmult.vec.json'
embeddings_in_json = 'ent_embeddings'
ids = 'uri2id.txt'

#Loading the required libraries and modules
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import json

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from tensorflow.keras import layers
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Flatten, Input, concatenate, Convolution2D, MaxPooling2D
# Keras specific
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from tensorflow.keras import layers
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Flatten, Input, Concatenate, Convolution2D, MaxPooling2D

id_file = open(ids)
id_dict = {}

for line in id_file:
    spl = str.split(line)
    id_dict[spl[0]] = int(spl[1])

data = json.load(open(embedding_file))[embeddings_in_json]

def get_vec_from_uri(uri):
    return data[id_dict[uri]]


def check_vec_in_dict(uri):
    return uri in id_dict


x_train = open(train_file, "r")
x_test = open(test_file, "r")




train_values_embeddings = []
train_labels = []
test_values_embeddings = []
test_labels = []
train_values_embeddings_changes = []
test_values_embeddings_changes = []
train_values_changes = []
test_values_changes = []


def toOneHot(actualChangeDetails):
    #arr = np.array([1, 1, 1, 1, 1])
    arr = np.array([0, 0, 0, 0, 0])
    changeDetails = [item.strip() for item in actualChangeDetails]
    if "Deleted" in changeDetails:
        arr[0] = 1
    if "Added" in changeDetails:
        arr[1] = 1
    if "Superclass" in changeDetails:
        arr[2] = 1
    if "Annotation" in changeDetails:
        arr[3] = 1
    if "Renamed" in changeDetails:
        arr[4] = 1
    return arr

for line in x_train:
    tokens = line.split(" ")
    item = tokens[0]
    label = tokens[1]
    changeDetails = tokens[4:]
    if (check_vec_in_dict(item)):
        vec = get_vec_from_uri(item)
        train_labels.append(int(label))
        train_values_embeddings.append(vec)
        oneHotDetails = toOneHot(changeDetails)
        train_values_changes.append(oneHotDetails)
        resVec = np.append(vec, oneHotDetails)
        train_values_embeddings_changes.append(resVec)

for line in x_test:
    tokens = line.split(" ")
    item = tokens[0]
    label = tokens[1]
    changeDetails = tokens[4:]
    if (check_vec_in_dict(item)):
        vec = get_vec_from_uri(item)
        test_labels.append(int(label))
        test_values_embeddings.append(vec)
        oneHotDetails = toOneHot(changeDetails)
        test_values_changes.append(oneHotDetails)
        resVec = np.append(vec, oneHotDetails)
        test_values_embeddings_changes.append(resVec)

def evaluate_data(train_values_embeddings, train_values_changes,test_values_embeddings, test_values_changes, train_labels, test_labels):

    left_branch_input = Input(shape=(100,), name='Embeddings')
    left_branch_input_1 = Dense(100, activation='tanh')(left_branch_input)
    left_branch_input_2 = Dense(200, activation='tanh')(left_branch_input_1)
    left_branch_input_3 = Dense(100, activation='tanh')(left_branch_input_2)
    left_branch_output = Dense(1, activation='softmax')(left_branch_input_3)
    left_flatten = Flatten()(left_branch_output)



    right_branch_input = Input(shape=(5,), name='Changes')
    right_branch_input_1 = Dense(5, activation='tanh')(right_branch_input)
    right_branch_input_2 = Dense(10, activation='tanh')(right_branch_input_1)
    right_branch_input_3 = Dense(5, activation='tanh')(right_branch_input_2)
    right_branch_output = Dense(1, activation='softmax')(right_branch_input_3)
    right_flatten = Flatten()(right_branch_output)
    # merge samples, two input must be same shape --> axis=0
    # merge row must same column size
    # # merge column must same row size --> axis=1

    concat = concatenate([left_flatten, right_flatten],axis=1, name='Concatenate')
    final_model_output = Dense(1, activation='softmax')(concat)
    final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                        name='Final_output')

    final_model.compile(optimizer='adam', loss='binary_crossentropy')
    # To train
    final_model.fit([np.array(train_values_embeddings), np.array(train_values_changes)], np.array(train_labels), epochs=50, verbose=0)

    final_model.summary()
    # plot graph
    plot_model(final_model, to_file='distmult_multilayer_perceptron_graph.png')

    result = final_model.predict([np.array(test_values_embeddings), np.array(test_values_changes)])
    name = 'DistMult'
    f1 = sklearn.metrics.f1_score(test_labels, result)
    acc = sklearn.metrics.accuracy_score(test_labels, result)
    prec = sklearn.metrics.precision_score(test_labels, result)
    rec = sklearn.metrics.recall_score(test_labels, result)
    roc_auc = sklearn.metrics.roc_auc_score(test_labels, result)
    msg = "%s: f1:%f acc:%f prec:%f rec:%f roc_auc:%f" % (name, f1, acc, prec, rec, roc_auc)
    print(msg)

evaluate_data(train_values_embeddings, train_values_changes,test_values_embeddings, test_values_changes,train_labels, test_labels)


