import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset
from Models import deep, shallow
from tensorflow.keras import callbacks
import Utilities

# The best hyper parameters found during the parameter search
experiments = [
    {'arch' : 'deep', 'balancing' : 'Augmentation', 'filters': 64, 'dropCNN': 0.1, 'dropFC': 0.6},
    {'arch' : 'deep', 'balancing' : 'Loss', 'filters': 64, 'dropCNN': 0.15, 'dropFC': 0.6},
    {'arch' : 'shallow', 'balancing' : 'Augmentation', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.55},
    {'arch' : 'shallow', 'balancing' : 'Loss', 'filters': 64, 'dropCNN': 0.25, 'dropFC': 0.3},
]

# create instance of the dataset, and get the training and validation set
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

for experiment in experiments:
    if experiment['arch'] == 'deep':
        model = deep(hyperParameters=experiment).model
        checkPointFilePath = './TrainedModels/Deep'
    elif experiment['arch'] == 'shallow':
        model = shallow(hyperParameters=experiment).model
        checkPointFilePath = './TrainedModels/Shallow'
    print(model.summary())
    
    if experiment['balancing'] == 'Loss':
        checkPointFilePath += ' Loss'
        checkPoint = callbacks.ModelCheckpoint(checkPointFilePath, save_best_only=True)
        history = model.fit(trainingData, trainingLabels, epochs=2, batch_size=64, validation_data=(validationData, validationLabels), class_weight=dataset.classWeights, callbacks=checkPoint)
    else:
        checkPointFilePath += 'Augmentation'
        checkPoint = callbacks.ModelCheckpoint(checkPointFilePath, save_best_only=True)
        history = model.fit(balancedTrainingData, balancedTrainingLabels, epochs=2, batch_size=256, validation_data=(validationData, validationLabels), callbacks=checkPoint)
    Utilities.saveTrainingHistory(history, checkPointFilePath)