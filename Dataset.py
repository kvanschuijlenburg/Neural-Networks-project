import pandas as pd
import numpy as np
import sys
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf

oneColumnFigureWidth = 10 # For latex

class Dataset:
    def __init__(self):

        self.classNames = {
            0:'anger',
            1:'disgust',
            2:'fear',
            3:'happiness',
            4:'sadness',
            5:'surprise',
            6:'neutral'
        }

        if not exists('./fer2013/processedData.npy'):
            self.csvToNumpy()

        processedData = np.load('./fer2013/processedData.npy', allow_pickle=True)
        
        self.trainingData = processedData[0]
        self.trainingLabels = processedData[1]
        self.augmentedData = processedData[2]
        self.augmentedLabels = processedData[3]
        self.testingData = processedData[4]
        self.testingLabels = processedData[5]
        self.benchmarkData = processedData[6]
        self.benchmarkLabels = processedData[7]


        countsPerClass = self.samplesPerClass()
        self.classWeights = dict(enumerate(np.min(countsPerClass)/countsPerClass))

    def plotSampleImages(self, images, labels):
        plt.figure(figsize = (oneColumnFigureWidth, 8))
        fig, ax = plt.subplots(3, 3, figsize=(4, 4))
        fig.subplots_adjust(hspace=0.3, wspace=1.0)

        for i in range(9):
            labelName = self.oneHotToLabel(labels[i])
            plt.subplot(3, 3, i+1) 
            plt.imshow(images[i], cmap = 'gray')    
            plt.title(labelName)   
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.savefig("./figures/datasetSample.png", dpi = 300, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf() 
    
    def samplesPerClass(self):
        counts = [0]*len(self.classNames)
        for label in self.trainingLabels:
            counts[np.argmax(label)]+=1
        return counts
        
    def plotSummary(self):
        classNames = list(self.classNames.values())
        counts = self.samplesPerClass()

        fig = plt.figure(figsize = (oneColumnFigureWidth, 5))
        plt.bar(classNames, counts, width = 0.4)  
        plt.xlabel("Emotions")
        plt.ylabel("Number of samples")
        plt.savefig("./figures/datasetBalance.png", dpi = 300, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()
    
    def augmentImages(self, data, labels):
        augmentedData = []
        augmentedLabels = []
        
        
        for index, image in enumerate(data):
            augmentations = [image]
            flipped = tf.image.flip_left_right(image).numpy()
            augmentations.append(flipped)
            #augmentations.append(tf.image.random_brightness(image,max_delta=0.05).numpy())
            #augmentations.append(tf.image.random_brightness(flipped,max_delta=0.05).numpy())
            #extraLabels = labels[index]*len(augmentedData)
            augmentedData.extend(augmentations)
            augmentedLabels.append(labels[index])
            augmentedLabels.append(labels[index])
            #augmentedLabels.extend(extraLabels)         

        self.plotSampleImages(augmentedData,augmentedLabels)
        return augmentedData,augmentedLabels


    def trainingSet(self, augment = False):
        """
        Returns the full training dataset
        """
        if augment:
            trainingData = self.augmentedData
            trainingLabels = self.augmentedLabels
        else:
            trainingData = self.trainingData
            trainingLabels = self.trainingLabels

        trainingSet = {"data": trainingData, "labels": trainingLabels}
        return trainingSet

    def testingSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.testingData, "labels": self.testingLabels}
        return testingSet

    def benchmarkSet(self):
        # Note: never augment the testing set
        benchmarkSet = {"data": self.benchmarkData, "labels": self.benchmarkLabels}
        #benchmarkSet = None # Note: the testing can only be used in the end
        return benchmarkSet

    def normalizeImages(self, data):
        # normalizedData = data.astype('float32')/255.0
        # normalizedData = tf.keras.utils.normalize(data)
        normalizedData = data.copy()
        normalizedData -= np.mean(normalizedData, axis=0)
        normalizedData /= np.std(normalizedData, axis=0)

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
        augmentedTrainingX, augmentedTrainingY = self.augmentImages(trainingX, trainingY)
        testingX = np.asarray(testingX)
        testingY = np.asarray(testingY)
        benchmarkX = np.asarray(benchmarkX)
        benchmarkY = np.asarray(benchmarkY)

        processedSet = []
        processedSet.append(self.normalizeImages(trainingX))
        processedSet.append(trainingY)
        processedSet.append(self.normalizeImages(augmentedTrainingX))
        processedSet.append(augmentedTrainingY)
        processedSet.append(self.normalizeImages(testingX))
        processedSet.append(testingY)
        processedSet.append(self.normalizeImages(benchmarkX))
        processedSet.append(benchmarkY)

        processedSet = np.asarray(processedSet)

        np.save('./fer2013/processedData', processedSet)


if __name__ == '__main__':
    test = Dataset()    
    test.plotSampleImages(test.augmentedData, test.augmentedLabels)
    test.plotSummary()