import os
import numpy as np

NUM_OF_CLUSTER = 40

# Transform all patches of the origin images into a row vector.
list = []
for i in range(NUM_OF_CLUSTER):
    vector = np.load("center_class40" + '/' + str(i) + '.npz')
    for x in vector.files:
        list.append(vector[x])

flag = 0
output = np.zeros(0)
labels = []
file_name = []
for root, dirs, my_files in os.walk('YOUR_PATH'):
    for file in my_files:
        if root[31] == 'l':
            labels.append(0)
        elif root[31] == 'm':
            labels.append(1)
        elif root[31] == 'h':
            labels.append(2)
       
        par = file.partition("_")
        # change this part according to your file name
        file_name.append((par[2].partition("_"))[0][:])

        # a is all vectors of a patch
        a = np.load(root + '/' + file)
        histogram = np.zeros(NUM_OF_CLUSTER)
        for x in a.files:
            for i in range(a[x].shape[0]):
                distance = np.linalg.norm(a[x][i] - list[0])
                dex = 0
                for j in range(1,NUM_OF_CLUSTER):
                    if distance > np.linalg.norm(a[x][i] - list[j]):
                        distance = np.linalg.norm(a[x][i] - list[j])
                        dex = j
                histogram[dex] += 1
            histogram = histogram / a[x].shape[0]

            if flag == 0:
                output = histogram
                flag = 1
            else:
                output = np.append(output, histogram, axis=0)

result_labels = np.array(labels)
result_vector = output.reshape((-1, 40))
print(result_labels.shape)
print(result_vector.shape)
np.savez("normal_histogram_labels_1", result_labels)
np.savez("normal_histogram_vector_1", result_vector)
with open("normal_histogram_filename_1.txt", 'w') as f:
    for i in file_name:
        f.write(i+'\n')