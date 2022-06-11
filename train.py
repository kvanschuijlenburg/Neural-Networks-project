import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset
from models import deep, shallow
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import Utilities

modelType = 'deep'

shallowParameters = {
    'filters': 64, 
    'dropoutCNN': 0.25, 
    'dropoutFC': 0.5, 
    'optimizer': "Adam", 
    'learningRate': 0.001, 
    'decaySteps' : 0, 
    'decayRate': 0.0,
    'batchSize' : 64,
    'epochs' : 40,
    'augmentedData' : False,
    }

deepParameters = {
    'filters': 64, 
    'dropoutCNN': 0.3, 
    'dropoutFC': 0.6, 
    'optimizer': "Adam", 
    'learningRate': 0.001, 
    'decaySteps' : 0, 
    'decayRate': 0.0,
    'batchSize' : 64,
    'epochs' : 40,
    'augmentedData' : False,
    }

if modelType == 'shallow':
    hyperParameters = shallowParameters
    model = shallow(hyperParameters=hyperParameters).model
    checkPointFilePath = './checkpoints/cnnShallow'
else:
    hyperParameters = deepParameters
    model = deep(hyperParameters=hyperParameters).model
    checkPointFilePath = './checkpoints/cnnDeep'
print(model.summary())

# create instance of the dataset, and get the training and validation set
dataset = Dataset()
trainingSet = dataset.trainingSet(hyperParameters['augmentedData'])
trainingData = trainingSet["data"]
trainingLabels = trainingSet["labels"]
validationSet = dataset.validationSet()
validationData = validationSet["data"]
validationLabels = validationSet["labels"]

# Fit the model on the training set
checkPoint = callbacks.ModelCheckpoint(checkPointFilePath, save_best_only=True)
history = model.fit(trainingData, trainingLabels, epochs=hyperParameters['epochs'], batch_size=hyperParameters['batchSize'], validation_data=(validationData, validationLabels), class_weight=dataset.classWeights, callbacks=checkPoint)

Utilities.plotTrainValResults(history, "./figures/Validation/", modelType)