import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
from Models import baseline
import Utilities

from tensorflow.keras import callbacks
import numpy as np

dataset = Dataset()
trainingSet = dataset.trainingSet()
trainingData = trainingSet["data"]
trainingLabels = trainingSet["labels"]
balancedTrainingSet = dataset.balancedTrainingSet()
balancedTrainingData = balancedTrainingSet['data']
balancedTrainingLabels = balancedTrainingSet['labels']
validationSet = dataset.validationSet()
validationData = validationSet["data"]
validationLabels = validationSet["labels"]

# Create the baseline architecture, and compile it
def trainBaseline(name, balanced):
    model = baseline().model
    checkPointFilePath = './TrainedModels/' + name
    checkPoint = callbacks.ModelCheckpoint(checkPointFilePath, save_best_only=True)
    if balanced:
        history = model.fit(balancedTrainingData, balancedTrainingLabels, epochs=100, batch_size=64, validation_data=(validationData, validationLabels), callbacks=checkPoint)
    else:
        history = model.fit(trainingData, trainingLabels, epochs=100, batch_size=64, validation_data=(validationData, validationLabels), callbacks=checkPoint)
    print("max validation accuracy " + str(round(np.max(history.history['val_accuracy']), 4)))
    Utilities.saveTrainingHistory(history, checkPointFilePath)

trainBaseline("Baseline Loss", False)
trainBaseline("Baseline Augmentation", True)