from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import sys

class Dataset:
    def __init__(self):
        if not (exists('./fer2013/trainingX.npy') and exists('./fer2013/trainingY.npy') and exists('./fer2013/testingX.npy') and 
            exists('./fer2013/testingY.npy')  and exists('./fer2013/benchmarkX.npy') and exists('./fer2013/benchmarkY.npy')):
            self.csvToNumpy()

        self.trainingData = np.load('./fer2013/trainingX.npy')
        self.trainingLabels = np.load('./fer2013/trainingY.npy')
        self.testingData = np.load('./fer2013/testingX.npy')
        self.testingLabels = np.load('./fer2013/testingY.npy')
        self.benchmarkData = np.load('./fer2013/benchmarkX.npy')
        self.benchmarkLabels = np.load('./fer2013/benchmarkY.npy')

        self.classNames = {
            0:'anger',
            1:'disgust',
            2:'fear',
            3:'happiness',
            4:'sadness',
            5:'surprise',
            6:'neutral'
        }

        for i in range(9):
            labelName = self.oneHotToLabel(self.trainingLabels[i])
            
            plt.subplot(3, 3, i+1) 
            plt.imshow(self.trainingData[i], cmap = 'gray')    
            plt.title(labelName)   
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.tight_layout(pad=3.0)
        plt.show()   
        


    def trainingSet(self):
        """
        Returns the full training dataset
        """
        trainingSet = {"data": self.trainingData, "labels": self.trainingLabels}
        return trainingSet

    def trainValidateSet(self):
        """
        Splits the full training set in a training and validation set.
        The split is randomly selected. However, the seed is fixed. Therefore, the sets will be equal every time.
        """
        trainingSplitData, validationData, trainingSplitLabels, validationLabels = train_test_split(self.trainingData, self.trainingLabels, test_size=0.10, random_state=10)
        trainingSplitSet = {"data": trainingSplitData, "labels": trainingSplitLabels}
        validationSet = {"data": validationData, "labels": validationLabels}
        return trainingSplitSet, validationSet

    def testingSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.testingData, "labels": self.testingLabels}
        testingSet = None # Note: the testing can only be used in the end
        return testingSet

    def normalizeImages(self, data):
        normalizedData = data.astype('float32')/255.0
        return normalizedData

    def classToLabel(self, integer):
        return self.classNames[integer]

    def oneHotToLabel(self, oneHot):
        return self.classToLabel(np.argmax(oneHot))

    def csvToNumpy(self):
        csvFile = pd.read_csv('./fer2013/fer2013.csv')
        pixels = csvFile['pixels'].tolist()

        dataX = []
        for row in pixels:
            pixelArray = [int(pixel) for pixel in row.split(' ')]
            dataX.append(np.asarray(pixelArray).reshape(48, 48).astype('float32'))
        dataX = np.asarray(dataX)
        dataX = np.expand_dims(dataX, -1)
        dataX = self.normalizeImages(dataX)
        dataY = pd.get_dummies(csvFile['emotion']).values
        dataUsage = pd.get_dummies(csvFile['Usage']).values

        trainingX = []
        trainingY = []
        testingX = []
        testingY = []
        benchmarkX = []
        benchmarkY = []

        for index, image in enumerate(dataX):
            usage = np.argmax(dataUsage[index])
            if usage == 2:
                trainingX.append(image)
                trainingY.append(dataY[index])
            elif usage == 1:
                benchmarkX.append(image)
                benchmarkY.append(dataY[index])
            elif usage == 0:
                testingX.append(image)
                testingY.append(dataY[index])
            else:
                sys.exit('Error: usage undefined')
        
        trainingX = np.asarray(trainingX)
        trainingY = np.asarray(trainingY)
        testingX = np.asarray(testingX)
        testingY = np.asarray(testingY)
        benchmarkX = np.asarray(benchmarkX)
        benchmarkY = np.asarray(benchmarkY)

        np.save('./fer2013/trainingX', trainingX)
        np.save('./fer2013/trainingY', trainingY)
        np.save('./fer2013/testingX', testingX)
        np.save('./fer2013/testingY', testingY)
        np.save('./fer2013/benchmarkX', benchmarkX)
        np.save('./fer2013/benchmarkY', benchmarkY)


if __name__ == '__main__':
    test = Dataset()    