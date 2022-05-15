import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer
from keras import backend
from keras.losses import categorical_crossentropy
import csv
import numpy as np
import torch

dataset = Dataset()
classWeights = backend.variable(dataset.classWeights)

def crossValidate(model, dataset):
    folds = 5
    summary = []

    kfold = KFold(n_splits=folds, random_state=1, shuffle=True)
    for trainingIndexes, validationIndexes in kfold.split(dataset["data"]):
        trainingData = dataset["data"][trainingIndexes]
        trainingLabels = dataset["labels"][trainingIndexes].astype(np.float32)
        validationData = dataset["data"][validationIndexes]
        validationLabels = dataset["labels"][validationIndexes].astype(np.float32)
        history = model.fit(trainingData, trainingLabels, epochs=20, batch_size=16, validation_data=(validationData, validationLabels))
        summary.append(history)
    return summary

def weighted_loss(y_true, y_pred):
    # https://gist.github.com/skeeet/cad06d584548fb45eece1d4e28cfa98b
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= backend.sum(y_pred,axis=-1, keepdims=True)
    # clip
    y_pred = backend.clip(y_pred, backend.epsilon(), 1)
    # calc
    loss = y_true*backend.log(y_pred)*classWeights
    loss =-backend.sum(loss,-1)
    return loss

def createModel(parameters):
    dropoutLayers = 4 # do not change
    convolutionalLayers = parameters['layers']
    useWeightedLoss = parameters['balancing'] == 'LossFunction'
    
    model = Sequential()
    model.add(InputLayer(input_shape=(48, 48, 1)))

    for dropout in range(dropoutLayers):
        filters = 2**dropout*64
        for layer in range(convolutionalLayers):
            model.add(Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=2))

    # from here the model should not be changed 
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(7, activation='softmax'))

    if useWeightedLoss:
        loss = weighted_loss
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])
    print(model.summary())
    return model



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

def main():
    trainingSet = dataset.trainingSet()

    parameterGrid = [
        {'layers': 2, 'balancing': "LossFunction", 'augmentation': True},
        {'layers': 1, 'balancing': "LossFunction", 'augmentation': True},
        {'layers': 4, 'balancing': "Preprocessing", 'augmentation': False},
    ]

    # For each search, change the model and execute cross validation
    for parameterSet in parameterGrid:
        model = createModel(parameterSet)
        modelSummary = crossValidate(model,trainingSet)
        saveModelResult(parameterSet, modelSummary)

if __name__ == '__main__':
    main()