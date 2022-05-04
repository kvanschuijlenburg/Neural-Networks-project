import numpy as np
from keras.datasets import cifar10
# load and ordering of the dataset

class Dataset:
    def __init__():
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print("Shape of training data:")
        print(X_train.shape)
        print(y_train.shape)
        print("Shape of test data:")
        print(X_test.shape)
        print(y_test.shape)


    def trainingSet():
        trainingData = []
        trainingLabels = []
        trainingSet = {"data": np.asarray(trainingData), "labels": np.asarray(trainingLabels)}
        return trainingSet

    def testingSet():
        # Note: never augment the testing set
        testingData = []
        testingLabels = []
        testingSet = {"data": np.asarray(testingData), "labels": np.asarray(testingLabels)}
        return testingSet

if __name__ == '__main__':
    test = Dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print()