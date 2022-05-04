from Dataset import Dataset
from sklearn.model_selection import KFold
import numpy as np

folds = 5

def crossValidate(model, dataset):
    kf = KFold(n_splits=folds, random_state=1, shuffle=True)
    accuracies = []

    for trainingIndexes, validationIndexes in kf.split(dataset["data"]):
        trainingData = dataset["data"][trainingIndexes]
        trainingLabels = dataset["labels"][trainingIndexes]
        validationData = dataset["data"][validationIndexes]
        validationLabels = dataset["labels"][validationIndexes]
        model = 0 # model.fit(trainingData, trainingLabels)
        accuracy = 0 # model.evaluate(validationData, validationLabels)validate model
        accuracies.append(accuracy)

    return np.average(accuracies)


def createModel(layers):
    pass

def main():
    trainingSet = Dataset.trainingSet()
    model = createModel()
    modelAccuracy = crossValidate(model,trainingSet)
    pass

if __name__ == '__main__':
    main()