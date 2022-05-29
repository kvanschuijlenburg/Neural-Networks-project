import os
import matplotlib.pyplot as plt
import pandas as pd

twoColumnFigureWidth = 20 # For latex


pltTrainAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))
pltValAcc = plt.figure(figsize = (twoColumnFigureWidth, 10))

for filename in os.listdir("./gridsearchResults"):
    parameters = dict([parameter.split('=') for parameter in filename.removesuffix('.csv').split('_')])

    dataframe = pd.read_csv(os.path.join("gridsearchResults", filename))
    #dataframe.plot(x="Epoch", y=["accuracy","validation accuracy"])
    #plot = dataframe.plot(x="Epoch", y=["validation accuracy"])

    axTrainAcc = pltTrainAcc.add_subplot() 
    axTrainAcc.plot(dataframe["Epoch"], dataframe["accuracy"])
    
    axValAcc = pltValAcc.add_subplot() 
    axValAcc.plot(dataframe["Epoch"], dataframe["validation accuracy"])


pltTrainAcc.savefig("./figures/crossValidateTestAccuracy.png", dpi = 300, bbox_inches='tight')
pltValAcc.savefig("./figures/crossValidateValidationAccuracy.png", dpi = 300, bbox_inches='tight')
plt.close()
plt.cla()
plt.clf() 