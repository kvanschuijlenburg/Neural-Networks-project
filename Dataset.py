from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import sys

oneColumnFigureWidth = 10 # For latex

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
        countsPerClass = self.samplesPerClass()
        self.classWeights = np.min(countsPerClass)/countsPerClass

    def plotSampleImages(self):
        plt.figure(figsize = (oneColumnFigureWidth, 8))
        fig, ax = plt.subplots(3, 3, figsize=(4, 4))
        fig.subplots_adjust(hspace=0.3, wspace=1.0)

        for i in range(9):
            labelName = self.oneHotToLabel(self.trainingLabels[i])
            plt.subplot(3, 3, i+1) 
            plt.imshow(self.trainingData[i], cmap = 'gray')    
            plt.title(labelName)   
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        #plt.tight_layout(pad=3.0)
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

    def trainingSet(self):
        """
        Returns the full training dataset
        """
        trainingSet = {"data": self.trainingData, "labels": self.trainingLabels}
        return trainingSet

    def testingSet(self):
        # Note: never augment the testing set
        testingSet = {"data": self.testingData, "labels": self.testingLabels}
        return testingSet

    def benchmarkSet(self):
        # Note: never augment the testing set
        benchmarkSet = {"data": self.benchmarkData, "labels": self.benchmarkLabels}
        benchmarkSet = None # Note: the testing can only be used in the end
        return benchmarkSet

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
    #test.plotSampleImages()
    test.plotSummary()