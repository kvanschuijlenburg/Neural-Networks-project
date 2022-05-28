import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


for filename in os.listdir("./gridsearchResults"):
    dataframe = pd.read_csv(os.path.join("gridsearchResults", filename))
    dataframe.plot(x="Epoch", y=["accuracy","validation accuracy"])
    plt.show()

    with open(os.path.join("gridsearchResults", filename), newline='') as file:
        rows = csv.reader(file, delimiter=',')#, quotechar='|')
        trainingLoss = []
        trainingAccuracy = []
        validationLoss = []
        validationAccuracy = []
        
        for index, row in enumerate(rows):
            if index >0:
                trainingLoss.append(row[1])
                trainingAccuracy.append(row[2])
                validationLoss.append(row[3])
                validationAccuracy.append(row[4])

