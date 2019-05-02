import numpy as np
import operator
# 2019.4.8

"""
ceate fake data set
"""
def createDataSet():
    group = np.array(
        [[1, 100], [5, 89], [108, 5], [115, 8]]
    )
    labels = ['love', 'love', 'exercise', 'exercise']

    return group, labels

"""
 KNN algorithm implementation by using numpy
"""
def KNN(testdata, traindataSet, labels, k):

    dataSetSize = traindataSet.shape[0]
    diffMat = np.tile(testdata, (dataSetSize, 1)) - traindataSet
    # calculation
    diffMat = diffMat**2
    diffMat = diffMat.sum(axis=1)
    distance = diffMat**0.5
    # sort
    sortedindex = distance.argsort()
    classCount = {}
    # count
    for i in range(k):
        label_index = sortedindex[i]
        classCount[labels[label_index]] = classCount.get(labels[label_index], 0) + 1

    stortedclasses = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return stortedclasses[0][0]

if __name__ == '__main__':
    train_data, train_label = createDataSet()
    test_data = np.array([100, 2])
    test_label = KNN(test_data, train_data, train_label, 3)
    print(test_label)