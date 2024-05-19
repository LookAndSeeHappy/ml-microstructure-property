import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
import joblib

vector1 = np.zeros(1)
vector2 = np.zeros(1)
vector3 = np.zeros(1)
vector4 = np.zeros(1)
labels1 = np.zeros(1)
labels2 = np.zeros(1)
labels3 = np.zeros(1)
labels4 = np.zeros(1)

for root, dirs, my_files in os.walk('histogram'):
    for file in my_files:
        a = np.load(root + '/' + file)
        for x in a.files:
            if file == 'normal_histogram_labels_1.npz':
                labels1 = a[x]
            elif file == 'normal_histogram_labels_2.npz':
                labels2 = a[x]
            elif file == 'normal_histogram_labels_3.npz':
                labels3 = a[x]
            elif file == 'normal_histogram_labels_4.npz':
                labels4 = a[x]
            elif file == 'normal_histogram_vector_1.npz':
                vector1 = a[x]
            elif file == 'normal_histogram_vector_2.npz':
                vector2 = a[x]
            elif file == 'normal_histogram_vector_3.npz':
                vector3 = a[x]
            elif file == 'normal_histogram_vector_4.npz':
                vector4 = a[x]

my_label = [labels4, labels2, labels3, labels1]
my_vector = [vector4, vector2, vector3, vector1]

temp1 = np.append(my_label[0], my_label[1], axis=0)
temp2 = np.append(temp1, my_label[2], axis=0)
labels_train = np.append(temp2, my_label[3], axis=0)

t1 = np.append(my_vector[0], my_vector[1], axis=0)
t2 = np.append(t1, my_vector[2], axis=0)
vector_train = np.append(t2, my_vector[3], axis=0)

# you can do cross validation here by changing the hyperparameter
clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=6)
clf.fit(vector_train,labels_train)

fig = plt.figure(figsize=(32, 18))
tree.plot_tree(clf, node_ids=True,rounded=True,fontsize=20,impurity=False,filled=True,class_names=['low','median','high'])
fig.savefig("tree")
joblib.dump(clf,'tree_model')