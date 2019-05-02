import numpy as np
import operator
# 2019.4.8
# using datingTestSet.txt

"""
transform data in file to dataset 
"""
def file2dataset(filename):
    f = open(filename)
    lines = f.readlines()
    dataMatrix = [] # data
    labelVector = [] # label
    for line in lines:
        line = line.strip()
        data = line.split('\t')

        data[0:3] = map(eval, data[0:3])
        dataMatrix.append(data[0:3])

        if data[3] == 'largeDoses':
            labelVector.append(2)
        elif data[3] == 'smallDoses':
            labelVector.append(1)
        elif data[3] == 'didntLike':
            labelVector.append(0)

    dataMatrix = np.array(dataMatrix)
    labelVector = np.array(labelVector)
    return dataMatrix, labelVector
"""
data nomalization  (data - min)/(max - min)
"""
def dataScaling(dataset):
    data_min = dataset.min(0)
    data_max = dataset.max(0)
    data_range = data_max - data_min
    rows = len(dataset)
    data_min = np.tile(data_min, (rows, 1))
    data_range = np.tile(data_range, (rows, 1))
    dataset = (dataset - data_min)/data_range
    return dataset, data_range, data_min

"""
 KNN algorithm implementation by using numpy
"""
def KNN(testdata, traindataSet, labels, k):
    dataSize = len(traindataSet)
    testdata = np.tile(testdata, (dataSize, 1))
    diffMat = (testdata-traindataSet)**2
    distance = (diffMat.sum(axis=1))**0.5
    classindex = distance.argsort()
    classCount = {}
    for i in range(k):
        label = labels[classindex[i]]
        classCount[label] = classCount.get(label, 0) + 1

    classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]

def classification():
    filename = 'datingTestSet.txt'
    dataMatrix, labelVector = file2dataset(filename)
    dataMatrix, data_range, data_min = dataScaling(dataMatrix)
    ratio = 0.1
    dataSize = len(dataMatrix)
    testSize = int(dataSize*ratio)
    # create train set and test set
    testData = dataMatrix[:testSize]
    testLabel = labelVector[:testSize]
    trainData = dataMatrix[testSize:]
    trainLabel = labelVector[testSize:]
    true = 0
    # predict
    for i in range(testSize):
        prediction = KNN(testData[i, :], trainData, trainLabel, 4)
        print("prediction :%d, test_label: %d"%(prediction, testLabel[i]))
        if prediction == testLabel[i]:
            true += 1
    print("acc is %d%%" % (true / testSize * 100))


if __name__ == '__main__':
    classification()