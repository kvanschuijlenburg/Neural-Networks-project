import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from IPython.display import display
import numpy as np
from matplotlib import cm
import Utilities


twoColumnFigureWidth = 20 # For latex
cmap = {0:'red',1:'blue',2:'yellow',3:'green'}

pltValAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
#pltValLoss = plt.figure(figsize = (twoCik olumnFigureWidth, 10))
axValAcc = pltValAcc.add_subplot()
#axValLoss = pltValLoss.add_subplot()

parameters = []
maxValidationAccuracy = []
for index, filename in enumerate(os.listdir("./gridsearchResults")):
    experiment = dict([parameter.split('=') for parameter in filename.removesuffix('.csv').split('_')])
    for element in experiment.keys():
        if element == 'arch':
            modelType = experiment[element]      
        elif element == 'balancing':
            balancingType = experiment[element]          
    del experiment['arch']
    del experiment['balancing']
    dataframe = pd.read_csv(os.path.join("gridsearchResults", filename))

    index = 0
    if modelType == 'shallow':
        index += 2
    if balancingType == 'Loss':
        index +=1  

    axValAcc.plot(dataframe["Epoch"], dataframe["validation accuracy"], color=cmap[index])
    #axValLoss.plot(dataframe["Epoch"], dataframe["validation loss"], color=cmap[index])
    axValAcc.tick_params(axis ='both', which ='both', labelsize = 20)
    patchOne = mpatches.Patch(color=cmap[0], label='Deep, augmentation')
    patchTwo = mpatches.Patch(color=cmap[1], label='Deep, weighted loss')
    patchThree = mpatches.Patch(color=cmap[2], label='Shallow, augmentation')
    patchFour = mpatches.Patch(color=cmap[3], label='Shallow, weighted loss')
    
    found = False
    maxValAccuracy = round(dataframe["validation accuracy"].max(),3)

    for ind, parameter in enumerate(parameters):
        if experiment == parameter:
            maxValidationAccuracy[ind][index] = maxValAccuracy
            found = True
    if not found:
        parameters.append(experiment)
        resultArray = [0.0]*4
        resultArray[index] = maxValAccuracy
        maxValidationAccuracy.append(resultArray.copy())

plt.legend(handles=[patchOne, patchTwo,patchThree,patchFour],fontsize=20)
plt.xlabel("Epoch", fontsize=20)
plt.tick_params('both')
plt.ylabel("Validation accuracy", fontsize=20)
plt.savefig('./figures/ParameterSearch/parameterSearchValidationAccuracies.png', dpi = 300, bbox_inches='tight')
plt.close()
plt.cla()
plt.clf()

maxValidationAccuracy = np.asarray(maxValidationAccuracy)

table = pd.DataFrame(parameters)
table = table.drop('filters',1)

table['deep, augmentation'] = maxValidationAccuracy[:,0]
table['deep, loss'] = maxValidationAccuracy[:,1]
table['shallow, augmentation'] = maxValidationAccuracy[:,2]
table['shallow, loss'] = maxValidationAccuracy[:,3]

Utilities.tableToLatex(table)

fontSize = 20
fig = plt.figure(figsize=(20,20))
for modelNumber in range(4):
    print('max validation accuracy is '+str(np.max(maxValidationAccuracy[:,modelNumber])))
    nonZeroIndices = np.where(maxValidationAccuracy[:,modelNumber] != 0)[0]
    ax = fig.add_subplot(2, 2, modelNumber+1, projection='3d')
    indexMax = np.argmax(maxValidationAccuracy[:,modelNumber])
    ax.plot_trisurf(table['dropCNN'][nonZeroIndices], table['dropFC'][nonZeroIndices], maxValidationAccuracy[:,modelNumber][nonZeroIndices],cmap=cm.coolwarm)
    ax.tick_params(axis ='both', which ='both', labelsize = fontSize)
    x = float(table['dropCNN'][indexMax])
    y=  float(table['dropFC'][indexMax])
    z=maxValidationAccuracy[:,modelNumber][indexMax]+0.002
    #ax.scatter(x,y,z,s=10**2)
    if modelNumber == 0: modelName = 'Deep, augmentation'
    elif modelNumber == 1: modelName = 'Deep, weighted loss'
    elif modelNumber == 2: modelName = 'Shallow, augmentation'
    else: modelName = 'Shallow, weighted loss'
    ax.set_title(modelName, fontsize=fontSize)
    ax.set_xlabel('Dropout CNN', fontsize=fontSize, labelpad=20)
    ax.set_ylabel('Dropout FC', fontsize=fontSize, labelpad=20)
    ax.set_zlabel('Validation accuracy', fontsize=fontSize, labelpad=20)
    
    
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#plt.show()
plt.savefig('./figures/ParameterSearch/dropoutLandscapes.png', dpi = 300, bbox_inches='tight')
plt.close()
plt.cla()
plt.clf()


