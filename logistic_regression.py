import numpy as np
import joblib
import csv
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def analyze_tree(col_num_left, col_num_right):
    my_datas = dict()
    with open('LLZTO_grain_pore_standardization.csv', encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            my_datas[row[0]] = row

    vector1 = np.zeros(1)
    vector2 = np.zeros(1)
    vector3 = np.zeros(1)
    vector4 = np.zeros(1)

    for root, dirs, my_files in os.walk('histogram'):
        for file in my_files:
            a = np.load(root + '/' + file)
            for x in a.files:
                if file == 'normal_histogram_vector_1.npz':
                    vector1 = a[x]
                elif file == 'normal_histogram_vector_2.npz':
                    vector2 = a[x]
                elif file == 'normal_histogram_vector_3.npz':
                    vector3 = a[x]
                elif file == 'normal_histogram_vector_4.npz':
                    vector4 = a[x]

    temp1 = np.append(vector1, vector2, axis=0)
    temp2 = np.append(temp1, vector3, axis=0)
    temp3 = np.append(temp2, vector4, axis=0)

    file_name = []
    f = open("normal_histogram_filename_all.txt", "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        file_name.append(line)

    clf = joblib.load('tree_model')
    my_path = clf.decision_path(temp3).toarray()

    data_train_test_left,data_train_test_right = [],[]
    for data_col_num in range(1,23):
        data_list_left = []
        data_list_right = []
        a = np.where(my_path[:, col_num_left] == 1)
        b = np.where(my_path[:, col_num_right] == 1)

        for i in a[0]:
            data_list_left.append(float(my_datas[file_name[i]][data_col_num]))
        for i in b[0]:
            data_list_right.append(float(my_datas[file_name[i]][data_col_num]))
        data_train_test_left.append(data_list_left)
        data_train_test_right.append(data_list_right)
    left = np.array(data_train_test_left).transpose()
    right = np.array(data_train_test_right).transpose()
    return left,right

num = [(1,28),(2,17),(3,16),(4,9),(6,7),(10,13),(11,12),(14,15),(18,25),(20,21),(22,23),(26,27),(29,32),(34,35),(33,38),(30,31),(19,24),(5,8),(36,37)]

for number in range(19):
    which = num[number]
    left, right = analyze_tree(which[0], which[1])
    left = np.insert(left, 0, values=np.zeros(left.shape[0]), axis=1)
    right = np.insert(right, 0, values=np.ones(right.shape[0]), axis=1)
    data = np.concatenate((left, right))
    n, l = data.shape

    np.random.shuffle(data)

    train_num = n
    train_data = data[:train_num, 1:]
    train_lab = data[:train_num, 0]
    # test_data = data[train_num: , 1: ]
    # test_lab = data[train_num: , 0]

    # you can do cross validation here by changing the hyperparameter
    clf = LogisticRegression(penalty="l1", C=0.9, solver="liblinear",max_iter= 1000,tol=1e-6)
    clf.fit(train_data, train_lab)

    # print(accuracy_score(clf.predict(test_data), test_lab))
    # print(clf.coef_, clf.intercept_)

    file = open('result_params.txt', mode='a')
    file.write(str(which))
    file.write('\n')
    file.write(str(clf.coef_))
    file.write(str(clf.intercept_))
    file.write('\n')
    file.close()
