import os
import matplotlib.pyplot as plt
import pandas as pd

twoColumnFigureWidth = 20 # For latex



pltTrainAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
pltValAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
axTrainAcc = pltTrainAcc.add_subplot()
axValAcc = pltValAcc.add_subplot() 

for index, filename in enumerate(os.listdir("./gridsearchResults")):
    parameters = dict([parameter.split('=') for parameter in filename.removesuffix('.csv').split('_')])
    experimentName = parameters["name"]
    dataframe = pd.read_csv(os.path.join("gridsearchResults", filename))
    #dataframe.plot(x="Epoch", y=["accuracy","validation accuracy"])
    #plot = dataframe.plot(x="Epoch", y=["validation accuracy"])
    
    axTrainAcc.plot(dataframe["Epoch"], dataframe["accuracy"], label=experimentName)   
    axValAcc.plot(dataframe["Epoch"], dataframe["validation accuracy"], label = experimentName)

axTrainAcc.legend()
axValAcc.legend()
pltTrainAcc.savefig("./figures/crossValidateTrainAccuracy.png", dpi = 300, bbox_inches='tight')
pltValAcc.savefig("./figures/crossValidateValidationAccuracy.png", dpi = 300, bbox_inches='tight')
plt.close()
plt.cla()
plt.clf() 