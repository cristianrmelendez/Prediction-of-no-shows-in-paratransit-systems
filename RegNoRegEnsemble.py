from sklearn.metrics import accuracy_score, confusion_matrix

from MLAlgorithms import MLAlgorithms
from MLAlgorithms1 import MLAlgorithms1
import numpy as np


class RegNoRegEnsemble:

    REGModel = None
    NoREGModel = None

    def __init__(self):
        self.REGModel = MLAlgorithms(['Random Forest'], scaler='Normal Scaler', binary=True, dividePopulation=1)
        self.NoREGModel = MLAlgorithms1(['Random Forest'], scaler='Normal Scaler', binary=True, dividePopulation=0)

    def readAndPrepareData(self, filepath, nrows = None):
        self.REGModel.readAndPrepareData(filepath, numRows=nrows)
        self.NoREGModel.readAndPrepareData(filepath, numRows=nrows)

    def trainModels(self):
        self.REGModel.trainModels()
        self.NoREGModel.trainModels()


    def makePredictions(self):
        self.REGModel.makePredictions()
        self.NoREGModel.makePredictions()

    def printModelAccuracyAndConfusionMatrix(self):

        y_test = self.REGModel.y_test
        y_test = np.append(y_test, self.NoREGModel.y_test)

        predictions = self.REGModel.models[0][2]
        predictions =  np.append(predictions, self.NoREGModel.models[0][2])
        accuracies = ''
        accuracies = accuracies + " Prediction Accuracy: is %2.2f" % accuracy_score(y_test, predictions) + '\n'
        accuracies = accuracies + "Confusion Matrix:" + '\n'
        accuracies = accuracies + str(confusion_matrix(y_test, predictions)) + '\n'
        print(accuracies)







