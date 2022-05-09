from Dataset import Dataset
import numpy as np
from sklearn import linear_model

# create instance of the dataset
dataset = Dataset()

# get and flatten training set
trainingSet = dataset.trainingSet(oneHot = False)
trainingData = trainingSet["data"]
trainingData = trainingData.reshape(trainingData.shape[0],trainingData.shape[1] * trainingData.shape[2]* trainingData.shape[3])
trainingLabels = trainingSet["labels"]

# get and flatten testing set
testingSet = dataset.testingSet(oneHot = False)
testingData = np.asarray(testingSet["data"])

#testingData = testingData.reshape(testingData.shape[0],testingData.shape[1] * testingData.shape[2]* testingData.shape[3])
testingLabels = testingSet["labels"]

# Fit a linear model on the training set
linearModel = linear_model.LinearRegression() 
linearModel.fit(trainingData, trainingLabels)

# testing the linear model on the testing set
correct = 0
for index, image in enumerate(testingSet["data"]):
    prediction = linearModel.predict(image.reshape(1,-1))
    print(prediction)
    if np.round(prediction) == testingSet["labels"][index]:
        correct +=1

accuracy = np.sum(correct)/len(testingSet["data"])
print(accuracy)