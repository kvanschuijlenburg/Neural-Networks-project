# Neural-Networks-project
The data set used in this project is the [fer2013 data set](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data), which should be included as a .csv file in the folder "./fer2013/fer2013.csv" table. When an experiment is executed for the first time, the data set is read in the file Dataset.py. Normalization, calculating class weights, and augmention of the training data is then performed. Because this process is time consuming, the arrays containing the samples and class lables are stored as a numpy file. The next time an experiment is performed, this file will be loaded automatically.

## Parameter search
To run the hyper-parameters search, run the file ParameterSearch.py. In this file, the experiments performed in the project are included. After each search, a .csv file for the experiment is saved.

## Train baseline
The baseline can be trained by executing the BaselineMLP.py file. 

## Train model
To train the models with the optimal parameters found with the parameter search, run the file Train.py. For the deep and shallow models, using weighted loss and augmentation for class balancing, the models are trained. 

## Test model
By executing the file Test.py, the models generated during training the baseline, and the convolutional models are tested on the benchmark set. The test results are saved as a numpy array.

## Plotting results
### Parameter search
The plots of the hyper-parameter search, as well as the latex code for the table in the appendix of the report are plotted by executing the PlotParameterSearchResults.py file.

### Figures and experiment results
By running the file Utilities.py, the figures used in the report are plotted. Moreover, the top one and top two test accuracies are plotted in the terminal for each model.