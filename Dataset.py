from matplotlib import pyplot
import numpy as np
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras_preprocessing.text import one_hot
# load and ordering of the dataset

class Dataset:
    def __init__(self):
        self.classNames = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        (self.trainingData, self.trainingLabels), (self.testingData, self.testingLabels) = cifar10.load_data()

    def trainingSet(self):
        trainingSet = {"data": self.normalizeImages(self.trainingData), "labels": to_categorical(self.trainingLabels)}
        return trainingSet

    def testingSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.normalizeImages(self.testingData), "labels": to_categorical(self.testingLabels)}
        return testingSet

    def normalizeImages(self, data):
        normalizedData = data.astype('float32')/255.0
        return normalizedData

    def oneHotToLabel(self, oneHot):
        index = np.argmax(oneHot, axis=-1)
        return self.classNames(index)