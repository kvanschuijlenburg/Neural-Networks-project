import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from models import deep, shallow
import csv
import sys
import numpy as np

batchSize = 64
epochs = 100
saveLocation = "./gridsearchResults/"

#parameterSearch = [
    # Deep, class weights
    #{'architecture' : 'deep', 'balancing' : 'Loss', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6},
    #{'arch' : 'deep', 'balancing' : 'Augmentation', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6},
    #{'arch' : 'deep', 'balancing' : 'Loss', 'filters': 64, 'dropCNN': 0.5, 'dropFC': 0.6},

    # Deep, augmentation

    # Shallow, class weights
    #{'architecture' : 'shallow', 'balancing' : 'Loss', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.5},

    # Shallow, augmentation
    
    
    
    #{'name' : 'BalanceAugment', 'arch' : 'deep', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6, 'opti': "Adam", 'LR': 0.001, 'momentum' : 0, 'batch' : 64, 'epochs' : 100, 'augmented' : False},
    #{'name' : 'BalanceAugment', 'arch' : 'deep', 'filters': 64, 'dropCNN': 0.3, 'dropFC': 0.6, 'opti': "Adam", 'LR': 0.001, 'momentum' : 0, 'balancing' : 'Loss'},
    #{'name' : 'deep', 'arch' : 'deep', 'filters': 64, 'dropCNN': 0.5, 'dropFC': 0.6, 'opti': "Adam", 'LR': 0.0005, 'momentum' : 0, 'balancing' : 'Augmention'},
    #]

shallowAugmentationTime = 1350
shallowLossTime = 750
deepAugmentationTime = 1670
deepLossTime = 1100
estimatedTime = 0
parameterSearch = []

# Shared settings for all models
architectures = ['shallow','deep']
balancing = ['Augmentation','Loss']
filters = [64]
dropCNN = [0.1, 0.2, 0.3, 0.4, 0.5]
dropFC =  [0.4, 0.5, 0.6, 0.7]
for arch in architectures:
    for balance in balancing:
        for filter in filters:
            for dropoutCNN in dropCNN:
                for dropoutFC in dropFC:
                    if arch == 'deep' and balance == 'Loss': estimatedTime +=deepLossTime
                    elif arch == 'deep' and balance == 'Augmentation': estimatedTime +=deepAugmentationTime
                    elif arch == 'shallow' and balance == 'Loss': estimatedTime +=shallowLossTime
                    elif arch == 'shallow' and balance == 'Augmentation': estimatedTime += shallowAugmentationTime 
                    experimentDict = {'arch' : arch, 'balancing' : balance, 'filters': filter, 'dropCNN': dropoutCNN, 'dropFC': dropoutFC}
                    parameterSearch.append(experimentDict)

# Deep augmentation model
filters = [64]
#dropCNN = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
dropCNN = [0.05, 0.1, 0.15]
#dropFC =  [0.5, 0.6, 0.7]
dropFC =  [0.55, 0.6, 0.65]
for filter in filters:
    for dropoutCNN in dropCNN:
        for dropoutFC in dropFC:
            estimatedTime +=deepAugmentationTime
            experimentDict = {'arch' : 'deep', 'balancing' : 'Augmentation', 'filters': filter, 'dropCNN': dropoutCNN, 'dropFC': dropoutFC}
            parameterSearch.append(experimentDict)

# Deep loss model
filters = [64]
dropCNN = [0.15, 0.20, 0.25]
dropFC =  [0.6,0.7,0.8]
for filter in filters:
    for dropoutCNN in dropCNN:
        for dropoutFC in dropFC:
            estimatedTime +=deepLossTime
            experimentDict = {'arch' : 'deep', 'balancing' : 'Loss', 'filters': filter, 'dropCNN': dropoutCNN, 'dropFC': dropoutFC}
            parameterSearch.append(experimentDict)

# Shallow augmentation
filters = [64]
dropCNN = [0.15, 0.25]
dropFC =  [0.55, 0.65]
for filter in filters:
    for dropoutCNN in dropCNN:
        for dropoutFC in dropFC:
            estimatedTime +=shallowAugmentationTime
            experimentDict = {'arch' : 'shallow', 'balancing' : 'Augmentation', 'filters': filter, 'dropCNN': dropoutCNN, 'dropFC': dropoutFC}
            parameterSearch.append(experimentDict)


# Shallow loss
filters = [64]
dropCNN = [0.15, 0.25]
dropFC =  [0.2, 0.3, 0.4]
for filter in filters:
    for dropoutCNN in dropCNN:
        for dropoutFC in dropFC:
            estimatedTime +=shallowLossTime
            experimentDict = {'arch' : 'shallow', 'balancing' : 'Loss', 'filters': filter, 'dropCNN': dropoutCNN, 'dropFC': dropoutFC}
            parameterSearch.append(experimentDict)


hours = round(estimatedTime/3600)
minutes = round((estimatedTime - hours*3600)/60)
print('Number of experiments is '+ str(len(parameterSearch)) + ". Estimated time is " + str(hours) + " hours, and " + str(minutes) + " Minutes.")

def parametersToName(parameterSet):
    name = ""
    for key, value in parameterSet.items():
        name += key + "=" + str(value) + "_"
    name = name[:len(name)-1]
    return name

def saveModelResult(parameterSet, history):
    epochs = len(history.epoch)
    header = ['Epoch', 'loss', 'accuracy', 'validation loss', 'validation accuracy']
    dataRows = []
    for epoch in range(epochs):
        loss = history.history['loss'][epoch]
        accuracy = history.history['accuracy'][epoch]
        validationLoss = history.history['val_loss'][epoch]
        validationAccuracy = history.history['val_accuracy'][epoch]
        dataRows.append([epoch+1, loss, accuracy, validationLoss, validationAccuracy]) 
    with open(saveLocation + parametersToName(parameterSet) + ".csv", 'w', encoding='UTF8', newline='') as f:
        csvFile = csv.writer(f) 
        csvFile.writerow(header)
        for _, row in enumerate(dataRows): csvFile.writerow(row)

dataset = Dataset()
classWeights = dataset.classWeights

# For each search, change the model and execute cross validation
for index, hyperParameters in enumerate(parameterSearch):
    print("Search " + str(index+1) + "/" + str(len(parameterSearch)))
    
    experimentName = parametersToName(hyperParameters)
    if os.path.exists(saveLocation + experimentName + ".csv"):
        print("Experiment " + experimentName + " already performed, skip experiment")
        continue

    # Load the training and validation data
    if hyperParameters['balancing'] == 'Loss': trainingSet = dataset.trainingSet()
    elif hyperParameters['balancing'] == 'Augmentation': trainingSet = dataset.balancedTrainingSet()
    else: sys.exit("Wrong balancing parameter given")
    validationSet = dataset.validationSet()

    # Create instance of the model
    if hyperParameters['arch'] == 'shallow': model = shallow(hyperParameters).model
    elif hyperParameters['arch'] == 'deep': model = deep(hyperParameters).model
    else: sys.exit("Wrong architecture parameter given")

    # train the model
    if hyperParameters['balancing'] == 'Loss':
        history = model.fit(trainingSet["data"], trainingSet["labels"], epochs=epochs, batch_size=batchSize, validation_data=(validationSet["data"], validationSet["labels"]),class_weight=classWeights, verbose=1)
    else:
        history = model.fit(trainingSet["data"], trainingSet["labels"], epochs=epochs, batch_size=batchSize, validation_data=(validationSet["data"], validationSet["labels"]), verbose=1)
    
    print("max validation accuracy " + str(round(np.max(history.history['val_accuracy']), 4)))
    saveModelResult(hyperParameters, history)
    print("Search " + str(index+1) + " done")
    print()