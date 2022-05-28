from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer, Dropout, BatchNormalization
import sys
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def optimizers(hyperParameters):
    if hyperParameters['optimizer'] == "Adam":
        optimizer = Adam(learning_rate=hyperParameters['learningRate']) # Default LR = 0.001
    elif hyperParameters['optimizer'] == "SGD":
        optimizer = SGD(learning_rate=hyperParameters['learningRate']) # Default LR = 0.01
    elif hyperParameters['optimizer'] == "Decay":
        optimizer = ExponentialDecay(hyperParameters['learningRate'], hyperParameters['decaySteps'], hyperParameters['decayRate'])
    else:
        sys.exit("No optimizer specified")
    return optimizer

class shallow():
    def __init__(self, hyperParameters):
        # Shallow network
        dropoutCNN = hyperParameters['dropoutCNN'] # fraction of the input units to be dropped
        dropoutFC = hyperParameters['dropoutFC']
        filters = hyperParameters['filters']

        kernelSize = (7,7)
        poolingStrides = (2,2)
        poolingSize = (2,2)

        model = Sequential()
        model.add(InputLayer(input_shape=(48, 48, 1)))

        model.add(Conv2D(filters*2, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters*2, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters*2, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(dropoutFC))
        model.add(BatchNormalization())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropoutFC))
        model.add(BatchNormalization())

        model.add(Dense(7, activation='softmax'))

        optimizer = optimizers(hyperParameters)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

class deep():
    def __init__(self, hyperParameters):
        #Deep network
        dropoutCNN = hyperParameters['dropoutCNN'] # fraction of the input units to be dropped
        dropoutFC = hyperParameters['dropoutFC']
        filters = hyperParameters['filters']

        kernelSize = (3,3) # keep 3,3 for this deep network
        poolingStrides = (2,2)
        poolingSize = (2,2)

        model = Sequential()
        model.add(InputLayer(input_shape=(48, 48, 1)))

        model.add(Conv2D(filters, kernelSize, activation='relu'))
        model.add(Dropout(dropoutCNN))
        model.add(Conv2D(filters, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters*2, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(Conv2D(filters*2, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters*4, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(Conv2D(filters*4, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Conv2D(filters*8, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(Conv2D(filters*8, kernelSize, activation='relu', padding='same'))
        model.add(Dropout(dropoutCNN))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(poolingSize, poolingStrides))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(dropoutFC))
        model.add(BatchNormalization())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropoutFC))
        model.add(BatchNormalization())

        model.add(Dense(7, activation='softmax'))

        optimizer = optimizers(hyperParameters)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model