import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset
import Utilities

import tensorflow as tf
import numpy as np
import sys

testModel = 'deepAugmented'

# Load model
if testModel == 'shallowLoss': modelFilePath = './checkpoints/cnnShallowLoss'
elif testModel == "shallowAugmented": modelFilePath = './checkpoints/cnnShallowAugmented'
elif testModel == "deepLoss": modelFilePath = './checkpoints/cnnDeepLoss'
elif testModel == "deepAugmented": modelFilePath = './checkpoints/cnnDeepAugmented'
else: sys.exit("Wrong model given")
model = tf.keras.models.load_model(modelFilePath)

# Load dataset
dataset = Dataset()
testSet = dataset.benchmarkSet()

# predict all samples from the test set
print(model.evaluate(testSet["data"], testSet["labels"], batch_size = 1))
predicted = model.predict(testSet["data"])
labelPredicted = []
labelTrue = []
for index, prediction in enumerate(predicted):
    labelPredicted.append(int(np.argmax(prediction)))
    labelTrue.append(int(np.argmax(testSet["labels"][index])))

classLabels = dataset.classNames
Utilities.plotTopOneConfusionMatrix(labelTrue, labelPredicted, classLabels, "./figures/Results/confusionMatrix" + testModel)