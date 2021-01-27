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

# Loading the required libraries and modules
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
from tensorflow.keras import layers
import keras.backend as K
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Flatten, Input, Concatenate, Add
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

train_labels = []
test_labels = []

train_values_embeddings = []
test_values_embeddings = []

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


def evaluate_data(x_train_1, x_train_2, train_labels, test_labels):

    model_1 = Sequential()
    model_1.add(Dense(50, activation='relu', input_shape=(100,)))
    model_1.add(Dropout(0.1))
    model_1.add(Dense(25, activation='relu'))
    model_1.add(Dropout(0.1))
    model_1.add(Dense(10, activation='relu'))
    model_1.add(Dropout(0.1))
    model_1.add(Dense(1, activation='softmax'))
    model_1.add(Dropout(0.1))

    model_2 = Sequential()
    model_2.add(Dense(50, activation='relu', input_shape=(5,)))
    model_2.add(Dropout(0.1))
    model_2.add(Dense(25, activation='relu'))
    model_2.add(Dropout(0.1))
    model_2.add(Dense(10, activation='relu'))
    model_2.add(Dropout(0.1))
    model_2.add(Dense(1, activation='softmax'))
    model_2.add(Dropout(0.1))

    result = Sequential()
    fused = Concatenate([model_1, model_2])
    result.add(fused)
    #ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)

    #fused.compile(optimizer='Adagrad', loss='binary_crossentropy',metrics=['accuracy'])

    result.add(Dense(3, activation='sigmoid'))
    #model = Model(inputs=[input_tensor_1, input_tensor_2], outputs=[prediction])
    result.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    result.fit([np.array(x_train_1), np.array(x_train_2)], validation_data=(np.array(train_labels),np.array(test_labels)))
    result.summary()


evaluate_data(train_values_embeddings, train_values_changes, train_labels, test_labels)
# evaluate_data(train_values_embeddings, test_values_embeddings, train_labels, test_labels)
# evaluate_data(train_values_changes, test_values_changes, train_labels, test_labels)
