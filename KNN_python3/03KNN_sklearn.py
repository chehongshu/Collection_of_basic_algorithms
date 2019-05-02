from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from os import listdir
# 2019.4.9
# using trainingDigits and testDigits data set

"""
create data numpy
"""
def DigitsToData(filename):
    fr = open(filename)
    new_data = np.zeros((1, 1024))
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            new_data[0, i*32+j] = line[j]
    return new_data

"""
 KNN algorithm implementation by using sklearn 
"""
def classification():
    train_path = 'trainingDigits'
    trainDataName = listdir(train_path)

    trainSize = len(trainDataName)
    trainData = np.zeros((trainSize, 1024))
    trainLabels = []
    # get labels
    for i in range(trainSize):
        trainData[i, :] = DigitsToData(train_path+"/"+trainDataName[i])
        label = trainDataName[i].split('_')[0]
        trainLabels.append(int(label))
    # create KNN model
    KNN = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    # train
    KNN.fit(trainData, trainLabels)

    test_path = 'testDigits'
    testDataName = listdir(test_path)

    testSize = len(testDataName)
    true_count = 0
    # predict
    for i in range(testSize):
        testData = DigitsToData(test_path + "/" + testDataName[i])
        testlabel = int(testDataName[i].split('_')[0])
        prediction = KNN.predict(testData)
        print("预测为%d 真实值为%d"%(prediction, testlabel))
        if prediction == testlabel:
            true_count += 1

    print("error number is %d" % (testSize-true_count))
    print("acc is %f%%" % (true_count/testSize*100))


if __name__ == '__main__':
    classification()

