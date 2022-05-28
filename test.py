import platform
if platform.system() == 'Windows':
    import os
    if os.getlogin() == 'kvans':
        os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
        os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

testModel = 'shallow'
useTestSet = 'benchmark'

oneColumnFigureWidth = 10 # For latex

# Load model
if testModel == 'shallow':
    modelFilePath = './checkpoints/cnnShallow'
else:
    modelFilePath = './checkpoints/cnnDeep'
model = tf.keras.models.load_model(modelFilePath)

# Load dataset
dataset = Dataset()
if useTestSet == 'benchmark':
    testSet = dataset.benchmarkSet()
else:
    testSet = dataset.testingSet()

# predict all samples from the test set
print(model.evaluate(testSet["data"], testSet["labels"], batch_size = 1))
predicted = model.predict(testSet["data"])
labelPredicted = []
labelTrue = []
for index, prediction in enumerate( predicted):
    labelPredicted.append(int(np.argmax(prediction)))
    labelTrue.append(int(np.argmax(testSet["labels"][index])))
confusionMatrix = confusion_matrix(labelTrue, labelPredicted)

# Normalize matrix, and convert it to a pandas dataframe
confusionMatrixNormalized = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
labels = []
for index in range(7): labels.append(dataset.classToLabel(index))
matrixPandas = pd.DataFrame(confusionMatrixNormalized,labels, labels)

# Plot confusion matrix
plt.figure(figsize = (oneColumnFigureWidth, 5))
sn.set(font_scale=1.4)
sn.heatmap(matrixPandas, cmap = 'OrRd', annot=True, annot_kws={"size": 16}, fmt='.2f') # font size
plt.savefig("./figures/confusionMatrix"+testModel+".png", dpi = 300, bbox_inches='tight')