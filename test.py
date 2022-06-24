import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset

import tensorflow as tf
import numpy as np

directory = './TrainedModels'

# Load dataset
dataset = Dataset()
testSet = dataset.benchmarkSet()

for folder in os.listdir(directory):
    model = tf.keras.models.load_model(directory+ '/'+folder)
    # predict all samples from the test set
    metrics = model.evaluate(testSet["data"], testSet["labels"], batch_size = 64)
    predicted = model.predict(testSet["data"])
    labelPredicted = []
    labelTrue = []
    for index, prediction in enumerate(predicted):
        labelPredicted.append(int(np.argmax(prediction)))
        labelTrue.append(int(np.argmax(testSet["labels"][index])))
    saveTestData = np.asarray([labelPredicted, labelTrue, metrics])
    np.save(directory+ '/'+folder+'/testResults', saveTestData)