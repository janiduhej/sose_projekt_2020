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
embedding_file = './res/embedding_2009_transH.vec.json'
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
from tensorflow.keras import layers
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout

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

def evaluate_data(x_train,x_test, y_train, y_test):



    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(105,)))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    # Compiling the neural network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    # build the model
    model.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_test),np.array(y_test)), epochs=20, verbose=0)
    score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

evaluate_data(train_values_embeddings_changes,test_values_embeddings_changes,train_labels, test_labels)
#evaluate_data(train_values_embeddings, test_values_embeddings, train_labels, test_label)
#evaluate_data(train_values_changes, test_values_changes, train_labels, test_label)
