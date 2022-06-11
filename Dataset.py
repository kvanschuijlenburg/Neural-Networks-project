from secrets import choice
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
        plt.savefig("./figures/Dataset/datasetSamples.png", dpi = 300, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf() 

    def plotWrongImages(self, images):
        indexes = [2810, 1775, 5882, 25647, 3928, 18337, 21275, 4961, 20312]
        plt.figure(figsize = (oneColumnFigureWidth, 8))
        fig, ax = plt.subplots(3, 3, figsize=(4, 4))
        fig.subplots_adjust(hspace=0.3, wspace=1.0)
        for i in range(9):
            imageIndex = indexes[i]
            plt.subplot(3, 3, i+1) 
            plt.imshow(images[imageIndex], cmap = 'gray')    
            plt.title(chr(65+i))   
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.savefig("./figures/Dataset/datasetWrong.png", dpi = 300, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf() 
    
    def samplesPerClass(self, classLabels):
        counts = [0]*len(self.classNames)
        for label in classLabels:
            counts[np.argmax(label)]+=1
        return counts
        
    def plotSummary(self):
        classNames = list(self.classNames.values())
        counts = self.samplesPerClass(self.trainingLabels)
        fig = plt.figure(figsize = (oneColumnFigureWidth, 5))
        plt.bar(classNames, counts, width = 0.4)  
        plt.xlabel("Emotions")
        plt.ylabel("Number of samples")
        plt.savefig("./figures/Dataset/datasetBalance.png", dpi = 300, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()
    
    def augmentImages(self, data, labels, augmentations=2):
        augmentedData = []
        augmentedLabels = []
        augmentations=10
        for _, image in enumerate(data):
            augmentedImages = [tf.image.flip_left_right(image).numpy()]
            contrastLower = 1
            contrastUpper = 50
            brightness = 50
            if augmentations >= 2:
                augmentedImages.append(tf.image.random_brightness(image,max_delta=brightness).numpy())
            if augmentations >= 3:
                augmentedImages.append(tf.image.random_brightness(augmentedImages[0],max_delta=brightness).numpy())
            if augmentations >= 4:
                augmentedImages.append(tf.image.random_contrast(image,contrastLower, contrastUpper).numpy())
            if augmentations >= 5:
                augmentedImages.append(tf.image.random_contrast(augmentedImages[0],contrastLower, contrastUpper).numpy())
            if augmentations >= 6:
                augmentedImages.append(tf.image.random_contrast(augmentedImages[1],contrastLower, contrastUpper).numpy())
            if augmentations >= 7:
                augmentedImages.append(tf.image.random_contrast(augmentedImages[2],contrastLower, contrastUpper).numpy())#0.02, 0.1
            if augmentations >= 8:
                extraAgumentations = augmentations-7
                for _ in range(extraAgumentations):
                    augmentedImages.append(tf.image.random_brightness(image,max_delta=brightness).numpy())

            augmentedData.extend(augmentedImages)
            #augmentedLabels.append(labels[index])
            #augmentedLabels.append(labels[index])
        return augmentedData,augmentedLabels

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

    def rawTrainingData(self):
        trainingData = self.trainingData
        trainingLabels = self.trainingLabels
        trainingSet = {"data": trainingData, "labels": trainingLabels}
        return trainingSet

    def validationSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.testingData, "labels": self.testingLabels}
        return testingSet

    def benchmarkSet(self):
        # Note: never augment the testing set
        benchmarkSet = {"data": self.benchmarkData, "labels": self.benchmarkLabels}
        return benchmarkSet

    def normalizeImages(self, data):
        normalizedData = data.copy()
        normalizedData -= np.mean(normalizedData, axis=0)
        normalizedData /= np.std(normalizedData, axis=0)
        return normalizedData

    def balanceByAugmentation(self, dataX, dataY, augmentPerClass):
        dataPerClass = [[]]
        labelsPerClass= [[]]
        balancedX = []
        balancedY = []
        for index in range(len(augmentPerClass)-1):
            dataPerClass.append([])
            labelsPerClass.append([])

        for index, sample in enumerate(dataX):
            dataPerClass[np.argmax(dataY[index])].append(sample.copy())
            labelsPerClass[np.argmax(dataY[index])].append(dataY[index].copy())
        
        for index, augmentations in enumerate(augmentPerClass):
            samplesInClass = len(dataPerClass[index])
            augmentationsPerSample, toSample = divmod(augmentations, samplesInClass)
            
            extraFromSamples = []
            dataArray = np.asarray(dataPerClass[index])
            samplesToBeAugmented = dataArray[np.random.choice(dataArray.shape[0],toSample, replace=False), :]

            for image in samplesToBeAugmented:
                extraFromSamples.append(tf.image.random_brightness(image, max_delta=50))
                labelsPerClass[index].append(labelsPerClass[index][0])

            if augmentationsPerSample > 0:
                extraSamples,_ = self.augmentImages(dataPerClass[index], labelsPerClass[index], augmentations=augmentationsPerSample)
                dataPerClass[index].extend(extraSamples)
                toExtend = []
                for _ in range(len(extraSamples)):
                    toExtend.append(labelsPerClass[index][0])
                labelsPerClass[index].extend(toExtend)

            dataPerClass[index].extend(extraFromSamples)
      
            balancedX.extend(dataPerClass[index])
            balancedY.extend(labelsPerClass[index])
        
        balancedX = np.asarray(balancedX)
        balancedY = np.asarray(balancedY)
        p = np.random.permutation(len(balancedX))

        return balancedX[p], balancedY[p]

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
        countsPerClass = self.samplesPerClass(trainingY)
        augmentPerClass = np.max(countsPerClass)-countsPerClass
        #trainingX, trainingY = self.balanceByAugmentation(trainingX, trainingY, augmentPerClass)
        balancedX, balancedY = self.balanceByAugmentation(trainingX, trainingY, augmentPerClass)


        #augmentedTrainingX, augmentedTrainingY = self.augmentImages(trainingX, trainingY)
        #augmentedTrainingX=[]
        #augmentedTrainingY=[]
        testingX = np.asarray(testingX)
        testingY = np.asarray(testingY)
        benchmarkX = np.asarray(benchmarkX)
        benchmarkY = np.asarray(benchmarkY)

        processedSet = []
        processedSet.append(self.normalizeImages(trainingX))
        processedSet.append(trainingY)
        processedSet.append(self.normalizeImages(balancedX))
        processedSet.append(balancedY)
        processedSet.append(self.normalizeImages(testingX))
        processedSet.append(testingY)
        processedSet.append(self.normalizeImages(benchmarkX))
        processedSet.append(benchmarkY)

        processedSet = np.asarray(processedSet)

        np.save('./fer2013/processedData', processedSet)


if __name__ == '__main__':
    test = Dataset()    
    test.plotSampleImages(test.trainingData, test.trainingLabels)
    #test.plotWrongImages(test.trainingData)
    #test.plotSummary()