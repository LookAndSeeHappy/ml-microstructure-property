import os
import cv2
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

NAME = 'result/'
NUM_OF_CLUSTER = 40

flag = 0
input = np.zeros(0)
dict = {}
for root, dirs, my_files in os.walk('YOUR_VECTOR_AFTER_CAM'):
    for file in my_files:
        a = np.load(root + '/' + file)
        for x in a.files:
            if flag != 0:
                input = np.append(input, a[x], axis=0)
                dict[str(a[x][0])] = file
            else:
                input = a[x]
                dict[str(a[x][0])] = file
                flag = 1
data = np.float32(input)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.1)

flags = cv2.KMEANS_PP_CENTERS

compactness, labels, centers = cv2.kmeans(data, NUM_OF_CLUSTER, None, criteria, 20, flags)

for i in range(NUM_OF_CLUSTER):
    np.savez('center_class' + str(NUM_OF_CLUSTER) + '/' + str(i), centers[i])

my_labels = labels.ravel()
model = TSNE()
np.set_printoptions(suppress=True)
Y = model.fit_transform(data)

for i in range(NUM_OF_CLUSTER):
    Y_0 = []
    Y_1 = []
    for j in range(len(Y[:,0])):
        if i == my_labels[j]:
            Y_0.append(Y[:,0][j])
            Y_1.append(Y[:,1][j])
    if i < 11:
        plt.scatter(Y_0, Y_1, 1)
    elif 11 <= i < 21:
        plt.scatter(Y_0, Y_1, 10, marker='s')
    elif 21 <= i < 31:
        plt.scatter(Y_0, Y_1, 5, marker='o')
    elif 31 <= i < 41:
        plt.scatter(Y_0, Y_1, 5, marker='v')

led = []
for i in range(NUM_OF_CLUSTER):
    led.append(str(i))
plt.legend(led,loc=2, bbox_to_anchor=(1,1.2),borderaxespad = 0.)
plt.savefig(NAME + "Kmeans.png")
plt.show()
