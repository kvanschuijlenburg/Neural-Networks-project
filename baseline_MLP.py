import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
import matplotlib.pyplot as plt
import numpy as np

# create instance of the dataset
dataset = Dataset()

# get the training and validation set
trainingSet = dataset.trainingSet()
trainingData = trainingSet["data"]
trainingLabels = trainingSet["labels"]

validationSet = dataset.testingSet()
validationData = validationSet["data"]
validationLabels = validationSet["labels"]

# Create the baseline architecture, and compile it
model = Sequential()
model.add(InputLayer(input_shape=(48, 48, 1)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Fit the model on the training set
history = model.fit(trainingData, trainingLabels, epochs=20, batch_size=16, validation_data=(validationData, validationLabels))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Baseline accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./figures/baselineMLPAccuracy.png", dpi = 300, bbox_inches='tight')
plt.close()
plt.cla()
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Baseline loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./figures/baselineMLPLoss.png", dpi = 300, bbox_inches='tight')