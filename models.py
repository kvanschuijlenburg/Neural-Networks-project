from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer, Dropout, BatchNormalization

class baseline():
    def __init__(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(48, 48, 1))) 
        model.add(Flatten()) # 2304 dimensional vector
        model.add(Dense(2304, activation='relu'))
        model.add(Dense(7, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

class shallow():
    def __init__(self, hyperParameters):
        # Shallow network
        dropoutCNN = hyperParameters['dropCNN'] # fraction of the input units to be dropped
        dropoutFC = hyperParameters['dropFC']
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

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

class deep():
    def __init__(self, hyperParameters):
        #Deep network
        dropoutCNN = hyperParameters['dropCNN'] # fraction of the input units to be dropped
        dropoutFC = hyperParameters['dropFC']
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

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model