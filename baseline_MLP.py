import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")

from Dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from models import baseline

# create instance of the dataset
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