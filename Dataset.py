import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self):
        self.classNames = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        (self.trainingData, self.trainingLabels), (self.testingData, self.testingLabels) = cifar10.load_data()

    def trainingSet(self):
        """
        Returns the full training dataset
        """
        trainingSet = {"data": self.normalizeImages(self.trainingData), "labels": self.trainingLabels}
        return trainingSet

    def trainValidateSet(self):
        """
        Splits the full training set in a training and validation set.
        The split is randomly selected. However, the seed is fixed. Therefore, the sets will be equal every time.
        """
        trainingSplitData, validationData, trainingSplitLabels, validationLabels = train_test_split(self.trainingData, self.trainingLabels, test_size=0.10, random_state=10)
        trainingSplitSet = {"data": self.normalizeImages(trainingSplitData), "labels": trainingSplitLabels}
        validationSet = {"data": self.normalizeImages(validationData), "labels": validationLabels}
        return trainingSplitSet, validationSet

    def testingSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.normalizeImages(self.testingData), "labels": self.testingLabels}
        testingSet = None # Note: the testing can only be used in the end
        return testingSet

    def normalizeImages(self, data):
        normalizedData = data.astype('float32')/255.0
        return normalizedData

    def classToLabel(self, integer):
        return self.classNames(integer)