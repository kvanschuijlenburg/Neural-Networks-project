import os
import platform
if platform.system() == 'Windows' and os.getlogin() == 'kvans':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/zlib123dllx64/dll_x64")
from Dataset import Dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from numpy import expand_dims

TrainedModel = tf.keras.models.load_model('./checkpoints/cnnDeep')

# Load dataset
dataset = Dataset()
testSet = dataset.benchmarkSet()
image = testSet['data'][0]
image = expand_dims(image, axis=0)

convLayersIndexes = []
for i in range(len(TrainedModel.layers)):
    layer = TrainedModel.layers[i]
    if 'conv' in layer.name: 
        convLayersIndexes.append(i)

numberOfLayers = 8
plotFeaturesPerLayer = 4

#fig = plt.figure(figsize=(30,10))
rows = ['Filter {}'.format(col) for col in range(1, plotFeaturesPerLayer+1)]
cols = ['CNN layer {}'.format(col) for col in range(1, numberOfLayers+1)]
fig, axes = plt.subplots(nrows=plotFeaturesPerLayer, ncols=numberOfLayers, figsize=(20, 10))

pad = 5
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.tight_layout()
#fig.subplots_adjust(left=0.15, top=0.95)

for index, layerIndex in enumerate(convLayersIndexes):
    model = Model(inputs=TrainedModel.inputs , outputs=TrainedModel.layers[layerIndex].output)
    features = model.predict(image)

    for i in range(1,plotFeaturesPerLayer+1):
        #plotNumber = index * plotFeaturesPerLayer + i
        plotNumber = (i-1)*numberOfLayers + index +1
        plt.subplot(plotFeaturesPerLayer, numberOfLayers,plotNumber)
        plt.imshow(features[0,:,:,plotNumber-1] , cmap='gray')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
plt.savefig("figures/Features/EightLayers.png", dpi = 300, bbox_inches='tight')