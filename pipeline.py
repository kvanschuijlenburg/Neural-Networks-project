import platform
if platform.system() == 'Windows':
    import os
    if os.getlogin() == 'kvans':
        os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
        os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from models import deep, shallow
from sklearn.model_selection import KFold
from keras.models import Sequential
import csv
import sys
import numpy as np

parameterSearch = [
    #{'architecture' : 'shallow', 'filters': 64, 'dropoutCNN': 0.25, 'dropoutFC': 0.5, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 2, 'augmentedData' : False},
    #{'architecture' : 'deep', 'filters': 64, 'dropoutCNN': 0.3, 'dropoutFC': 0.6, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 100, 'augmentedData' : False},
    {'architecture' : 'deep', 'filters': 64, 'dropoutCNN': 0.25, 'dropoutFC': 0.6, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 100, 'augmentedData' : False},
    {'architecture' : 'deep', 'filters': 64, 'dropoutCNN': 0.35, 'dropoutFC': 0.6, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 100, 'augmentedData' : False},
    ]


def crossValidate(model : Sequential, dataset, hyperParameters):
    folds = 5

    # Save the initial weights such that they can be loaded before each new validation
    initialWeights = model.get_weights() 

    summary = []
    foldNumber = 1
    kfold = KFold(n_splits=folds, random_state=1, shuffle=True)
    for trainingIndexes, validationIndexes in kfold.split(dataset["data"]):
        print("  Fold "+ str(foldNumber) + "/" + str(folds), end='\r')
        
        trainingData = dataset["data"][trainingIndexes]
        trainingLabels = dataset["labels"][trainingIndexes]
        validationData = dataset["data"][validationIndexes]
        validationLabels = dataset["labels"][validationIndexes]
        model.set_weights(initialWeights)
        history = model.fit(trainingData, trainingLabels, epochs=hyperParameters['epochs'], batch_size=hyperParameters['batchSize'], validation_data=(validationData, validationLabels),class_weight=classWeights, verbose=1)
        summary.append(history)
        print("  Fold "+ str(foldNumber) + "/" + str(folds) + " max validation accuracy " + str(round(np.max(history.history['val_accuracy']), 4)))
        foldNumber +=1
    return summary

def saveModelResult(parameterSet, modelSummary):
    saveLocation = "./gridsearchResults/"
    filename = ""
    for key, value in parameterSet.items():
        filename += key + "=" + str(value) + "_"
    filename = filename[:len(filename)-1] + ".csv"

    folds = len(modelSummary)
    epochs = len(modelSummary[0].epoch)

    header = ['Epoch', 'loss', 'accuracy', 'validation loss', 'validation accuracy']
    dataRows = []
    for epoch in range(epochs):
        loss = 0
        accuracy=0
        validationLoss=0
        validationAccuracy=0

        for fold in range(folds):
            loss += modelSummary[fold].history['loss'][epoch]
            accuracy += modelSummary[fold].history['accuracy'][epoch]
            validationLoss += modelSummary[fold].history['val_loss'][epoch]
            validationAccuracy += modelSummary[fold].history['val_accuracy'][epoch]
        loss /= folds
        accuracy /= folds
        validationLoss /= folds
        validationAccuracy /= folds
        dataRows.append([epoch+1, loss, accuracy, validationLoss, validationAccuracy])
    
    with open(saveLocation + filename, 'w', encoding='UTF8', newline='') as f:
        csvFile = csv.writer(f) 
        csvFile.writerow(header)
        for _, row in enumerate(dataRows): csvFile.writerow(row)

dataset = Dataset()
classWeights = dataset.classWeights

# For each search, change the model and execute cross validation
for index, hyperParameters in enumerate(parameterSearch):
    print("Search " + str(index+1) + "/" + str(len(parameterSearch)))
    trainingSet = dataset.trainingSet(augment=hyperParameters['augmentedData'])
    testingSet = dataset.testingSet()
    trainingSet['data'] = np.append(trainingSet['data'] ,testingSet['data'],0)
    trainingSet['labels'] = np.append(trainingSet['labels'] ,testingSet['labels'],0)
    if hyperParameters['architecture'] == 'shallow':
        model = shallow(hyperParameters).model
    elif hyperParameters['architecture'] == 'deep':
        model = deep(hyperParameters).model
    else:
        sys.exit("No architecture chosen")
    modelSummary = crossValidate(model,trainingSet, hyperParameters)
    saveModelResult(hyperParameters, modelSummary)
    print("Search " + str(index+1) + " done")
    print()