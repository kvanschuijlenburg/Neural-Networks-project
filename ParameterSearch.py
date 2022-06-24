import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from Models import deep, shallow

import csv
import sys
import numpy as np
import datetime

saveLocation = "./gridsearchResults/"
batchSize = 64
epochs = 100

shallowAugmentationTime = 1350
shallowLossTime = 750
deepAugmentationTime = 1670
deepLossTime = 1100
estimatedTime = 0
search = []

def combinator(architectures, balancing, filters, dropCNN, dropFC):
    estimatedTime = 0
    parameters = []
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
                        parameters.append(experimentDict)
    return parameters

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

search = []

# Shared settings for all models
search.extend(combinator(['shallow','deep'], ['Augmentation','Loss'], [64], [0.1, 0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7]))

# Deep augmentation model
search.extend(combinator(['deep'], ['Augmentation'], [64], [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7]))
search.extend(combinator(['deep'], ['Augmentation'], [64], [0.05, 0.1, 0.15], [0.55, 0.6, 0.65]))
# best: search.extend(combinator(['deep'], ['Augmentation'], [64], [0.1], [0.6]))

# Deep loss model
search.extend(combinator(['deep'], ['Loss'], [64], [0.15, 0.20, 0.25], [0.6,0.7,0.8]))
search.extend(combinator(['deep'], ['Loss'], [64], [0.1, 0.15, 0.20], [0.55,0.6,0.65]))

# Shallow augmentation
search.extend(combinator(['shallow'], ['Augmentation'], [64], [0.15, 0.25], [0.55, 0.65]))
search.extend(combinator(['shallow'], ['Augmentation'], [64], [0.20, 0.25, 0.3], [0.5, 0.55, 0.6]))

# Shallow loss
search.extend(combinator(['shallow'], ['Loss'], [64], [0.15, 0.25], [0.2, 0.3, 0.4]))
search.extend(combinator(['shallow'], ['Loss'], [64], [0.2, 0.25, 0.3], [0.25, 0.3, 0.35]))

print('Number of experiments is '+ str(len(search)) + ". Estimated time is " + str(datetime.timedelta(seconds=estimatedTime)))

dataset = Dataset()
classWeights = dataset.classWeights

for index, hyperParameters in enumerate(search):
    print("Search " + str(index+1) + "/" + str(len(search)))
    
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