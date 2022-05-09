import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from sklearn.model_selection import KFold
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

folds = 5

def crossValidate(model, dataset):
    kf = KFold(n_splits=folds, random_state=1, shuffle=True)
    accuracies = []

    for trainingIndexes, validationIndexes in kf.split(dataset["data"]):
        trainingData = dataset["data"][trainingIndexes]
        trainingLabels = dataset["labels"][trainingIndexes]
        validationData = dataset["data"][validationIndexes]
        validationLabels = dataset["labels"][validationIndexes]    

        history = model.fit(trainingData, trainingLabels, epochs=2, batch_size=64, validation_data=(validationData, validationLabels))
        _, acc = model.evaluate(validationData, validationLabels, verbose=0)
        accuracies.append(acc)
    return np.average(accuracies)


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

    # For a grid search, change the architecture and run new tests
    model = createModel()
    modelAccuracy = crossValidate(model,trainingSet)
    print(modelAccuracy)


if __name__ == '__main__':
    main()