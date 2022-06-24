import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import os
from IPython.display import display

#from sqlalchemy import true
#from Dataset import Dataset

oneColumnFigureWidth = 10 # For latex

# def plotTrainValResults(history, folder, name):
def plotTrainValResults(data, folder, name):
    # Data is [epoch, loss, accuracy, validationLoss, validationAccuracy]
    saveLocation = folder + name

    plt.plot(data[:,2])
    plt.plot(data[:,4])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Accuracy.png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(data[:,1])
    plt.plot(data[:,3])
    plt.title(name + 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveLocation + "Loss.png", dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

def plotTopOneConfusionMatrix(labelPrediction, labelTrue, labelsDict, fileName):
    trueClass = np.argmax(labelTrue, axis=1)
    predictedClass = np.argmax(labelPrediction, axis=1)
    confusionMatrix = confusion_matrix(trueClass, predictedClass)

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

def tableToLatex(table : pd.DataFrame):
    with open("./figures/ParameterSearch/LatexTable.txt", 'w') as latex:
        latex.write('    \\begin{tabular}{c')
        for column in range(table.columns.size):
            latex.write('|c')
        latex.write('}\n')
        latex.write('        \hline \n')
        latex.write('        Parameter search')
        
        for column in table.columns:
            latex.write(' & '+ column)
        latex.write(' \\\\ \n')
        latex.write('        \\hline \n')
        
        for index, row in table.iterrows():
            rowText = '        '
            rowText += str(index+1)
            for subIndex, variable in enumerate(table.columns):
                rowText += ' & '
                if subIndex < table.columns.size-4:
                    rowText += row[variable]
                else:
                    maxValue = row[variable] == table.max()[subIndex]
                    if maxValue:
                        rowText += '\\textbf{' + str(row[variable]) + '}'
                    else:
                        if row[variable] == 0.0:
                            rowText += '-'
                        else:
                            rowText += str(row[variable])
            rowText += ' \\\\ \n'
            latex.write(rowText)
        latex.write('    \\end{tabular}\n')

def topNAccuracy(predicted, true):
    trueClass = np.argmax(true, axis=1)
    nCorrect = [0]*len(predicted[0])
    for row, prediction in enumerate(predicted):
        correctClass = trueClass[row]
        sortedPrediction = np.argsort(prediction, axis=0)

        predictedCorrect = False
        for topN in range(len(predicted[0])):
            predictedClass = sortedPrediction[len(sortedPrediction)-topN-1]
            
            if predictedClass == correctClass or predictedCorrect:
                predictedCorrect = True
                nCorrect[topN] +=1
    nCorrect = np.asarray(nCorrect)
    nAccuracy = nCorrect/len(true)
    return nAccuracy

def classPrecision(predicted, true):
    trueClass = np.argmax(true, axis=1)
    tempClasses = [0]*len(predicted[0])
    nCorrectPerClass = []
    for i in range(len(predicted[0])):
        nCorrectPerClass.append(tempClasses.copy())
    # nCorrectPerClass = [tempClasses.copy()]*len(predicted[0])
    samplesPerClass = [0]*len(predicted[0])
    for row, prediction in enumerate(predicted):
        correctClass = trueClass[row]
        samplesPerClass[correctClass] +=1
        sortedPrediction = np.argsort(prediction, axis=0)

        predictedCorrect = False
        for topN in range(len(predicted[0])):
            predictedClass = sortedPrediction[len(sortedPrediction)-topN-1]
            if predictedClass == correctClass or predictedCorrect:
                predictedCorrect = True
                # nCorrect[topN] +=1
                nCorrectPerClass[topN][correctClass] +=1
    for i in range(len(nCorrectPerClass[0])):
        for j in range(len(nCorrectPerClass[0])):
            nCorrectPerClass[i][j] /= samplesPerClass[j]
    return nCorrectPerClass

def plotTrainingResults(directory = "./TrainedModels"):
    for folder in os.listdir(directory):
        data = np.load(directory + '/'+ folder + '/trainingHistory.npy')
        print(folder + ': max validaton accuracy ' + str(round(max(data[:, 4]),3)))
        plotTrainValResults(data, './figures/validationResults/',folder)

def plotAllValidatonAccuracy(directory = "./TrainedModels"):
    for folder in os.listdir(directory):
        data = np.load(directory + '/'+ folder + '/trainingHistory.npy')
        plt.plot(data[:,4])
        plt.title('Validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Baseline Augmentation', 'Baseline Loss', 'Deep Augmentation', 'Deep Loss', 'Shallow Augmentation', 'Shallow Loss'], loc='lower right')
    plt.savefig('./figures/validationResults/ValidationsAccuracy.png', dpi = 300, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()

def plotTestResults(directory = "./TrainedModels"):
    classLabels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}
    for folder in os.listdir(directory):
        data = np.load(directory + '/'+ folder + '/testResults.npy', allow_pickle=True)
        # print(folder + ': ' + str(data[2]))
        # plotTrainValResults(data, './figures/validationResults/',folder)
        print()
        print(folder)
        precisions = classPrecision(data[0], data[1])[:2]
        table = pd.DataFrame(precisions)
        display(table.transpose())
        #accuracy = topNAccuracy(data[0], data[1])
        #print(folder + ' top 1 and 2 class precision ' + str(precisions[:2]))
        #plotTopOneConfusionMatrix(data[0], data[1], classLabels, "./figures/Results/confusionMatrix" + folder)

# plotAllValidatonAccuracy()
# plotTrainingResults()
plotTestResults()