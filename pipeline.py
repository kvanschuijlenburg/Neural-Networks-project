import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from sklearn.model_selection import KFold
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
import csv


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


def crossValidate(model, dataset):
    folds = 2
    summary = []

    kfold = KFold(n_splits=folds, random_state=1, shuffle=True)
    for trainingIndexes, validationIndexes in kfold.split(dataset["data"]):
        trainingData = dataset["data"][trainingIndexes]
        trainingLabels = to_categorical(dataset["labels"][trainingIndexes])
        validationData = dataset["data"][validationIndexes]
        validationLabels = to_categorical(dataset["labels"][validationIndexes])

        history = model.fit(trainingData, trainingLabels, epochs=2, batch_size=128, validation_data=(validationData, validationLabels))
        #_, acc = model.evaluate(validationData, validationLabels, verbose=0)
        summary.append(history)
    return summary


def createModel(layers=1):
    model = Sequential()

    # define architecture
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    dataset = Dataset()
    trainingSet = dataset.trainingSet()

    parameterGrid = [
        {'layers': 3, 'kernel': "linear"},
        {'layers': 5, 'kernel': "linear"},
    ]

    # For each search, change the model and execute cross validation
    for parameterSet in parameterGrid:
        model = createModel(parameterSet)
        modelSummary = crossValidate(model,trainingSet)
        saveModelResult(parameterSet, modelSummary)


if __name__ == '__main__':
    main()