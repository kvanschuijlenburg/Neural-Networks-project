from copyreg import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import os

from sqlalchemy import true
#from Dataset import Dataset

oneColumnFigureWidth = 10 # For latex

# def plotTrainValResults(history, folder, name):
def plotTrainValResults(data, folder, name):
    # Data is [epoch, loss, accuracy, validationLoss, validationAccuracy]
    saveLocation = folder + name

    plt.plot(data[:,2])
    plt.plot(data[:,4])
    plt.title('Baseline accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Accuracy.png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(data[:,1])
    plt.plot(data[:,3])
    plt.title('Baseline loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Loss.png", dpi = 300, bbox_inches='tight')

def plotTopOneConfusionMatrix(labelTrue, labelPredicted, labelsDict, fileName):
    confusionMatrix = confusion_matrix(labelTrue, labelPredicted)

    # Normalize matrix, and convert it to a pandas dataframe
    confusionMatrixNormalized = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    labels = []
    for index in range(7): labels.append(labelsDict[index])
    matrixPandas = pd.DataFrame(confusionMatrixNormalized,labels, labels)

    # Plot confusion matrix
    plt.figure(figsize = (oneColumnFigureWidth, 5))
    sn.set(font_scale=1.4)
    sn.heatmap(matrixPandas, cmap = 'OrRd', annot=True, annot_kws={"size": 16}, fmt='.2f') # font size
    plt.savefig(fileName + ".png", dpi = 300, bbox_inches='tight')

def plotDatasetImages(fileName, images, labels = None):
    plt.figure(figsize = (oneColumnFigureWidth, 10))
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for i in range(9):
        if labels == None:
            labelName = chr(65+i)
        else:
            labelName = labels[i]
        plt.subplot(3, 3, i+1) 
        plt.imshow(images[i], cmap = 'gray')    
        plt.title(labelName)   
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.savefig(fileName + ".png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf() 

def plotSummary(fileName, classNames, countsPerClass):
    fig = plt.figure(figsize = (oneColumnFigureWidth, 5))
    plt.bar(classNames, countsPerClass, width = 0.4)  
    plt.xlabel("Emotions")
    plt.ylabel("Number of samples")
    plt.savefig(fileName + ".png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

def saveTrainingHistory(history, saveLocation):
    epochs = len(history.epoch)
    #'Epoch', 'loss', 'accuracy', 'validation loss', 'validation accuracy']
    dataRows = []
    for epoch in range(epochs):
        loss = history.history['loss'][epoch]
        accuracy = history.history['accuracy'][epoch]
        validationLoss = history.history['val_loss'][epoch]
        validationAccuracy = history.history['val_accuracy'][epoch]
        dataRows.append([epoch+1, loss, accuracy, validationLoss, validationAccuracy]) 
    dataRows = np.asarray(dataRows)
    np.save(saveLocation +'/trainingHistory', dataRows)

def plotTrainingResults(directory = "./TrainedModels"):
    for folder in os.listdir(directory):
        data = np.load(directory + '/'+ folder + '/trainingHistory.npy')
        plotTrainValResults(data, './figures/validationResults/',folder)

def plotTestResults(directory = "./TrainedModels"):
    classLabels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}
    for folder in os.listdir(directory):
        data = np.load(directory + '/'+ folder + '/testResults.npy', allow_pickle=True)
        # plotTrainValResults(data, './figures/validationResults/',folder)
        plotTopOneConfusionMatrix(data[0], data[1], classLabels, "./figures/Results/confusionMatrix" + folder)

#plotTrainingResults()
plotTestResults()