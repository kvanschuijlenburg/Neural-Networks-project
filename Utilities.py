import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

oneColumnFigureWidth = 10 # For latex

def plotTrainValResults(history, folder, name):
    # summarize history for accuracy
    saveLocation = folder + name
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Baseline accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Accuracy.png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Baseline loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Loss.png", dpi = 300, bbox_inches='tight')

def plotTopOneConfusionMatrix(labelTrue, labelPredicted, labels, fileName):
    confusionMatrix = confusion_matrix(labelTrue, labelPredicted)

    # Normalize matrix, and convert it to a pandas dataframe
    confusionMatrixNormalized = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    labels = []
    for index in range(7): labels.append(labels[index])
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