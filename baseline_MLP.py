import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
import Utilities
from models import baseline

import numpy as np

# Instantiate dataset
dataset = Dataset()

# get the training and validation set
trainingSet = dataset.trainingSet()
trainingData = trainingSet["data"]
trainingLabels = trainingSet["labels"]

validationSet = dataset.validationSet()
validationData = validationSet["data"]
validationLabels = validationSet["labels"]

# Create the baseline architecture, and compile it
model = baseline().model
print(model.summary())

# Fit the model on the training set
history = model.fit(trainingData, trainingLabels, epochs=100, batch_size=64, validation_data=(validationData, validationLabels))
print("max validation accuracy " + str(round(np.max(history.history['val_accuracy']), 4)))

Utilities.plotTrainValResults(history, "./figures/baseline/", "baselineMLP")