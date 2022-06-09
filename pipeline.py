import platform
if platform.system() == 'Windows':
    import os
    if os.getlogin() == 'kvans':
        os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
        os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from models import deep, shallow
from keras.models import Sequential
import csv
import sys
import numpy as np

parameterSearch = [
    #{'architecture' : 'shallow', 'filters': 64, 'dropoutCNN': 0.25, 'dropoutFC': 0.5, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 2, 'augmentedData' : False},
    #{'architecture' : 'deep', 'filters': 64, 'dropoutCNN': 0.3, 'dropoutFC': 0.6, 'optimizer': "Adam", 'learningRate': 0.001, 'decaySteps' : 0, 'decayRate': 0.0, 'batchSize' : 64, 'epochs' : 100, 'augmentedData' : False},
    #{'name' : 'BalanceAugment', 'arch' : 'deep', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6, 'opti': "Adam", 'LR': 0.001, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    {'name' : 'BalanceAugment', 'arch' : 'shallow', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6, 'opti': "Adam", 'LR': 0.001, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    #{'name' : 'SGD.05', 'arch' : 'deep', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.4, 'opti': "SGD", 'LR': 0.05, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    #{'name' : 'SGD.01','arch' : 'deep', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.4, 'opti': "SGD", 'LR': 0.01, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    #{'name' : 'SGD.005','arch' : 'deep', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.4, 'opti': "SGD", 'LR': 0.005, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    ]

def saveModelResult(parameterSet, history):
    saveLocation = "./gridsearchResults/"
    filename = ""
    for key, value in parameterSet.items():
        filename += key + "=" + str(value) + "_"
    filename = filename[:len(filename)-1] + ".csv"

    epochs = len(history.epoch)

    header = ['Epoch', 'loss', 'accuracy', 'validation loss', 'validation accuracy']
    dataRows = []
    for epoch in range(epochs):
        loss = history.history['loss'][epoch]
        accuracy = history.history['accuracy'][epoch]
        validationLoss = history.history['val_loss'][epoch]
        validationAccuracy = history.history['val_accuracy'][epoch]
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
    trainingSet = dataset.trainingSet(augment=hyperParameters['augmented'])
    testingSet = dataset.testingSet()
    if hyperParameters['arch'] == 'shallow':
        model = shallow(hyperParameters).model
    elif hyperParameters['arch'] == 'deep':
        model = deep(hyperParameters).model
    else:
        sys.exit("No architecture chosen")
    print(model.summary())
    #history = model.fit(trainingSet["data"], trainingSet["labels"], epochs=hyperParameters['epochs'], batch_size=hyperParameters['batch'], validation_data=(testingSet["data"], testingSet["labels"]),class_weight=classWeights, verbose=1)
    history = model.fit(trainingSet["data"], trainingSet["labels"], epochs=hyperParameters['epochs'], batch_size=hyperParameters['batch'], validation_data=(testingSet["data"], testingSet["labels"]), verbose=1)
    print("max validation accuracy " + str(round(np.max(history.history['val_accuracy']), 4)))
    saveModelResult(hyperParameters, history)
    print("Search " + str(index+1) + " done")
    print()