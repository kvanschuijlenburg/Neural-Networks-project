import pandas as pd
import numpy as np
import sys
from os.path import exists
import tensorflow as tf
import Utilities
from skimage.util import random_noise
import random


class Dataset:
    def __init__(self):
        self.classNames = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}

        if not exists('./fer2013/processedData.npy'): self.csvToNumpy()

        processedData = np.load('./fer2013/processedData.npy', allow_pickle=True)
        self.trainingData = processedData[0]
        self.trainingLabels = processedData[1]
        self.balancedData = processedData[2]
        self.balancedLabels = processedData[3]
        self.testingData = processedData[4]
        self.testingLabels = processedData[5]
        self.benchmarkData = processedData[6]
        self.benchmarkLabels = processedData[7]

        countsPerClass = self.samplesPerClass(self.trainingLabels)
        self.classWeights = dict(enumerate(np.min(countsPerClass)/countsPerClass))
      
    def trainingSet(self):
        """
        Returns the normalized training dataset of the FER2013 contest
        """
        trainingSet = {"data": self.trainingData, "labels": self.trainingLabels}
        return trainingSet

    def balancedTrainingSet(self):
        """
        Returns the balanced by augmnentation, and normalized training dataset of the FER2013 contest
        """
        trainingSet = {"data": self.balancedData, "labels": self.balancedLabels}
        return trainingSet

    def validationSet(self):
        """
        Returns the FER2013 test set which is used for validation
        """
        testingSet = {"data": self.testingData, "labels": self.testingLabels}
        return testingSet

    def benchmarkSet(self):
        """
        Returns the FER2013 benchmark set
        """
        benchmarkSet = {"data": self.benchmarkData, "labels": self.benchmarkLabels}
        return benchmarkSet

    def samplesPerClass(self, classLabels):
        counts = [0]*len(self.classNames)
        for label in classLabels: counts[np.argmax(label)]+=1
        return counts

    def classToLabel(self, integer):
        return self.classNames[integer]

    def oneHotToLabel(self, oneHot):
        labels = []
        for encoded in oneHot: labels.append(self.classToLabel(np.argmax(encoded)))
        return labels

    def augmentImages(self, data, augmentations):
        augmentedData = []
        for index, image in enumerate(data):
            augmentedImages = [tf.image.flip_left_right(image).numpy()]
            for _ in range(augmentations-1):
                if bool(random.getrandbits(1)):
                    augmentedImages.append(random_noise(image, mode='gaussian', var=0.0005))
                else:
                    augmentedImages.append(random_noise(augmentedImages[0], mode='gaussian', var=0.0005))    
            #if augmentations >= 9 and index < 8:
            #    Utilities.plotDatasetImages("./figures/Dataset/datasetBalanced" + str(index), augmentedImages)
            augmentedData.extend(augmentedImages)
        return augmentedData

    def balanceByAugmentation(self, dataX, dataY, augmentPerClass):
        # create and fill the labels and data arrays per class
        dataPerClass = [[]]
        labelsPerClass= [[]]
        for index in range(len(augmentPerClass)-1):
            dataPerClass.append([])
            labelsPerClass.append([])
        for index, sample in enumerate(dataX):
            dataPerClass[np.argmax(dataY[index])].append(sample.copy())
            labelsPerClass[np.argmax(dataY[index])].append(dataY[index].copy())
        
        # Augment the data per class
        balancedX = []
        for index, augmentations in enumerate(augmentPerClass):
            augmentationsEachSample, toSample = divmod(augmentations, len(dataPerClass[index]))
            
            dataInClass = np.asarray(dataPerClass[index])
            sampledFromClass = dataInClass[np.random.choice(dataInClass.shape[0],toSample, replace=False), :]

            for image in sampledFromClass:
                balancedX.append(random_noise(image, mode='gaussian', var=0.0005))
            
            if augmentationsEachSample > 0:
                balancedX.extend(self.augmentImages(dataPerClass[index], augmentationsEachSample))         
            balancedX.extend(dataPerClass[index])

        balancedY = []
        for index, labels in enumerate(labelsPerClass):
            numberOfLabels = len(dataPerClass[index]) + augmentPerClass[index]
            for _  in range(numberOfLabels): balancedY.append(labels[0])

        balancedX = np.asarray(balancedX)
        balancedY = np.asarray(balancedY)
        p = np.random.permutation(len(balancedX))
        return balancedX[p], balancedY[p]

    def normalizeImages(self, data):
        normalizedData = data.copy()
        normalizedData -= np.mean(normalizedData, axis=0)
        normalizedData /= np.std(normalizedData, axis=0)
        return normalizedData

    def csvToNumpy(self):
        # Read 
        csvFile = pd.read_csv('./fer2013/fer2013.csv')
        pixels = csvFile['pixels'].tolist()
        datasetY = pd.get_dummies(csvFile['emotion']).values
        datasetUsage = csvFile['Usage'].tolist()

        trainingX, trainingY = [], []
        testingX, testingY = [], []
        benchmarkX, benchmarkY = [], []

        for index, row in enumerate(pixels):
            pixelArray = [int(pixel) for pixel in row.split(' ')]
            datasetX = np.asarray(pixelArray).reshape(48, 48).astype('float32')
            datasetX /=255.0
            datasetX = np.expand_dims(datasetX, -1)
            if datasetUsage[index] == 'Training':
                trainingX.append(datasetX)
                trainingY.append(datasetY[index])
            elif datasetUsage[index] == 'PublicTest':
                benchmarkX.append(datasetX)
                benchmarkY.append(datasetY[index])
            elif datasetUsage[index] == 'PrivateTest':
                testingX.append(datasetX)
                testingY.append(datasetY[index])
            else:
                sys.exit('Error: usage undefined')
        
        countsPerClass = self.samplesPerClass(trainingY)
        augmentPerClass = np.max(countsPerClass)-countsPerClass
        balancedX, balancedY = self.balanceByAugmentation(trainingX, trainingY, augmentPerClass)

        saveArray = []
        saveArray.append(self.normalizeImages(np.asarray(trainingX)))
        saveArray.append(np.asarray(trainingY))
        saveArray.append(self.normalizeImages(balancedX))
        saveArray.append(balancedY)
        saveArray.append(self.normalizeImages(np.asarray(testingX)))
        saveArray.append(np.asarray(testingY))
        saveArray.append(self.normalizeImages(np.asarray(benchmarkX)))
        saveArray.append(np.asarray(benchmarkY))
        saveArray = np.asarray(saveArray)

        np.save('./fer2013/processedData', saveArray)

if __name__ == '__main__':
    dataset = Dataset()    
    Utilities.plotDatasetImages("./figures/Dataset/datasetBalanced", dataset.balancedData, dataset.oneHotToLabel(dataset.balancedLabels))
    Utilities.plotDatasetImages("./figures/Dataset/datasetSamples", dataset.trainingData, dataset.oneHotToLabel(dataset.trainingLabels))
    wrongExamples = dataset.trainingData[[2810, 1775, 5882, 25647, 3928, 18337, 21275, 4961, 20312]]
    Utilities.plotDatasetImages("./figures/Dataset/datasetWrong", wrongExamples)
    #Utilities.plotSummary("./figures/Dataset/datasetBalance", list(dataset.classNames.values()), dataset.samplesPerClass(dataset.trainingLabels))