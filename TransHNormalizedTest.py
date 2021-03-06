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


train_file = 'changes0910DETAILSDIST2REVERSED.dat'
test_file = 'changes1011DETAILSDIST2REVERSED.dat'
embedding_file = './res/embedding_2009_transH.vec.json'
embeddings_in_json = 'ent_embeddings'
ids = 'uri2id.txt'

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
import sklearn
import numpy as np
import json

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


f_train = open(train_file, "r")
f_test = open(test_file, "r")

train_values_embeddings = []
train_labels = []
test_values_embeddings = []
test_labels = []
train_values_embeddings_changes = []
test_values_embeddings_changes = []
train_values_changes = []
test_values_changes = []



def toOneHot(actualChangeDetails):
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


for line in f_train:
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

for line in f_test:
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

# In[34]:


input_data = train_values_changes + test_values_changes
plot_labels = ["Deleted", "Added", "Superclass", "Annotation", "Renamed"]
diag_data = []
for i in range(0, len(plot_labels)):
    complete = [vec[i] for vec in input_data]
    diag_data.append(np.sum(complete) / len(input_data))

fig, axes = plt.subplots()

axes.plot(plot_labels, diag_data, 'o')
axes.set_title('Rel.Häufigkeit der Veränderungstypen')
plt.tight_layout()
plt.show()

# In[35]:


deleted_ind = []
added_ind = []
superclass_ind = []
annotation_ind = []
renamed_ind = []

all_ind = [deleted_ind, added_ind, superclass_ind, annotation_ind, renamed_ind]

print(type(train_labels), type(test_labels))
input_data = test_values_changes

all_labels = test_labels

for ind_list_index in range(len(all_ind)):
    for case_index in range(len(all_labels)):
        if input_data[case_index][ind_list_index] > 0:
            all_ind[ind_list_index].append(all_labels[case_index])
diag_data = []
for label, case_labels in zip(plot_labels, all_ind):
    if (len(case_labels) > 0):
        avg = np.sum(case_labels) / len(case_labels)
    else:
        avg = 0
    diag_data.append(avg)
    print(label, avg)
overall_avg = np.sum(all_labels) / len(all_labels)
print('Overall: ', overall_avg)

fig, axes = plt.subplots()
axes.plot(plot_labels, diag_data, 'o')
axes.set_title("Relative Häufigkeit der Klasse NOCHANGE nach Veränderungstyp")
axes.set_ylabel('Häufigkeit')
axes.axhline(y=overall_avg)
plt.show()

# In[36]:

from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


pca = PCA(n_components=2)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

data = preprocessing.normalize(train_values_embeddings)
norm = normalize(data)

normalized_data = pca.fit_transform(norm)

xs_normalized = [v[0] for v in normalized_data]
ys_normalized = [v[1] for v in normalized_data]

nochange_indices = [i for i in range(len(train_labels)) if train_labels[i] >0]
change_indices = [i for i in range(len(train_labels)) if train_labels[i] == 0]

xs_normalized_nochange = [xs_normalized[i] for i in nochange_indices]
ys_normalized_nochange = [ys_normalized[i] for i in nochange_indices]

xs_normalized_change = [xs_normalized[i] for i in change_indices]
ys_normalized_change = [ys_normalized[i] for i in change_indices]

# # Durchführen des Trainings und der Evaluation
#
# Params:
#
# - `models`, die Modelle, die evaluiert werden

# In[24]:

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn import metrics

import pandas as pd



def evaluate_data(train_values, train_labels, test_values, test_labels, graph_label):
    names = [" "]
    recs = []
    precs = []
    roc_aucs = []
    accs = []
    models = []
    models.append(('MLP 250', MLPClassifier(hidden_layer_sizes=(250,))))
    models.append(('SVM - linear', SVC(kernel="linear")))
    models.append(('SVM - rbf', SVC()))
    models.append(('RandomForest', RandomForestClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('LR', LogisticRegression()))

    models = reversed(models)
    for name, model in models:
        model.fit(train_values, train_labels)
        result = model.predict(test_values)
        f1 = sklearn.metrics.f1_score(test_labels, result)
        acc = sklearn.metrics.accuracy_score(test_labels, result)
        prec = sklearn.metrics.precision_score(test_labels, result)
        rec = sklearn.metrics.recall_score(test_labels, result)
        roc_auc = sklearn.metrics.roc_auc_score(test_labels, result)

        msg = "%s: \t f1:%f \t acc:%f \t prec:%f \t rec:%f \t roc_auc:%f" % (name, f1, acc, prec, rec, roc_auc)
        print(msg)
        names.append(name)
        recs.append(rec)
        precs.append(prec)
        roc_aucs.append(roc_auc)
        accs.append(acc)
    fig, axes = plt.subplots()
    for l in range(len(precs)):
        axes.axvline(x=l, ls='dashed', color='grey', alpha=0.5)
    axes.axhline(y=0.64)
    axes.plot(precs, 'o', label='Precision', ms=7)
    axes.plot(recs, '^', label='Recall', ms=7)
    axes.plot(accs, 's', label='accuracy', ms=7)
    axes.legend()
    axes.set_title(graph_label)
    axes.set_xticklabels(names, rotation=90)
    # plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


# In[25]:


train_values = train_values_changes
test_values = test_values_changes

evaluate_data(train_values, train_labels, test_values, test_labels, 'only changes')

# In[26]:


train_values = train_values_embeddings_changes
test_values = test_values_embeddings_changes

evaluate_data(train_values, train_labels, test_values, test_labels, 'embeddings and changes')

# In[61]:


train_values = train_values_embeddings
test_values = test_values_embeddings

evaluate_data(train_values, train_labels, test_values, test_labels, 'embeddings only')


