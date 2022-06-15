import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

twoColumnFigureWidth = 20 # For latex

parameters = []
results = []
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
    maxValAccuracy = round(dataframe["validation accuracy"].max(),3)
    index = 0
    if modelType == 'shallow':
        index += 2
    if balancingType == 'Loss':
        index +=1  
    found = False
    for ind, parameter in enumerate(parameters):
        if experiment == parameter:
            results[ind][index] = maxValAccuracy
            found = True
    if not found:
        parameters.append(experiment)
        resultArray = [0.0]*4
        resultArray[index] = maxValAccuracy
        results.append(resultArray.copy())

results = np.asarray(results)
table = pd.DataFrame(parameters)

table['deep, augmentation'] = results[:,0]
table['deep, loss'] = results[:,1]
table['shallow, augmentation'] = results[:,2]
table['shallow, loss'] = results[:,3]
display(table.transpose())


fig = plt.figure() #figsize=(20,20)

for modelNumber in range(4):
    nonZeroIndices = np.where(results[:,modelNumber] != 0)[0]
    ax = fig.add_subplot(2, 2, modelNumber+1, projection='3d')
    ax.plot_trisurf(table['dropCNN'][nonZeroIndices], table['dropFC'][nonZeroIndices], results[:,modelNumber][nonZeroIndices],cmap=cm.coolwarm)
    #surf = ax.plot_wireframe(table['dropCNN'][nonZeroIndices], table['dropFC'][nonZeroIndices], results[:,modelNumber][nonZeroIndices])#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if modelNumber == 0: modelName = 'Deep, augmentation'
    elif modelNumber == 1: modelName = 'Deep, loss'
    elif modelNumber == 2: modelName = 'Shallow, augmentation'
    else: modelName = 'Shallow, loss'
    ax.set_title(modelName)
    ax.set_xlabel('Dropout CNN')
    ax.set_ylabel('Dropout FC')
    ax.set_zlabel('Validation accuracy')
    

#ax = fig.add_subplot(1, 2, 2, projection='3d')
#surf = ax.plot_trisurf(table['dropCNN'][:24], table['dropFC'][:24], results[:,0][:24])#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()





# pltTrainAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
# pltValAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
# axTrainAcc = pltTrainAcc.add_subplot()
# axValAcc = pltValAcc.add_subplot()


#axTrainAcc.plot(dataframe["Epoch"], dataframe["accuracy"], label=experimentName)   
#axValAcc.plot(dataframe["Epoch"], dataframe["validation accuracy"], label = experimentName)

# axTrainAcc.legend()
# axValAcc.legend()
# pltTrainAcc.savefig("./figures/crossValidateTrainAccuracy.png", dpi = 300, bbox_inches='tight')
# pltValAcc.savefig("./figures/crossValidateValidationAccuracy.png", dpi = 300, bbox_inches='tight')
# plt.close()
# plt.cla()
# plt.clf() 