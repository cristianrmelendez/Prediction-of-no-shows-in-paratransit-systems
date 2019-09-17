import pandas
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from feature_selection_tests import FeatureSelectionTests
import matplotlib.pyplot as plt
from demandDataTrasformer import transformDemandData

class MLAlgorithms:

    data = None
    trainData = None
    testData = None
    models = []
    scaler = []
    featureSelector = None
    interactions = False
    dividePopulation = None
    initialPath = None

    def __init__(self, modelList = None, scaler = None, featureSelector = None, binary = False, interactions = False, dividePopulation = None):

        if 'Logistic Regression' in  modelList:
            if binary:
                self.models.append(['Logistic Regression', LogisticRegression( solver='sag')])
            else:
                self.models.append(['Logistic Regression', LogisticRegression(multi_class='multinomial', solver='sag')])

        if 'Decision Tree' in modelList:
            self.models.append(['Decision Tree', DecisionTreeClassifier(class_weight='balanced')])
        if 'Random Forest' in modelList:
            self.models.append(['Random Forest', RandomForestClassifier(class_weight='balanced')])
        if 'Naive Bayes' in modelList:
            if binary:
                self.models.append(['Naive Bayes', GaussianNB()])
            else:
                self.models.append(['Naive Bayes', MultinomialNB()])
        if 'MLP Classifier' in modelList:
            self.models.append(['MLP Classifier', MLPClassifier(verbose=True, max_iter=2000, solver="adam")])
        if 'SVC' in modelList:
            self.models.append(['SVC', svm.SVC(kernel='rbf', C=0.5, gamma=1, probability=True, verbose=True)])
        if 'K Nearest Neighbors' in modelList:
            self.models.append(['K Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)])
        if 'AdaBoost' in modelList:
            self.models.append(['AdaBoost', AdaBoostClassifier()])
        if 'Gradient Boosting' in modelList:
            self.models.append(['Gradient Boosting', GradientBoostingClassifier()])
        if 'Voting' in modelList:
            self.models.append(['Voting Classifier', VotingClassifier(estimators=[('Logistic Regression', LogisticRegression(solver='sag')), ('Decision Tree', DecisionTreeClassifier(class_weight='balanced')), ('Random Forest', RandomForestClassifier(class_weight='balanced'))], voting='soft')])
        if 'Bagging' in modelList:
            self.models.append(['Bagging Classifier', BaggingClassifier(LogisticRegression())])


        if not self.models:
            raise Exception('No Models Selected!')

        if scaler:
            if scaler == 'Standard Scaler':
                self.scaler = ['Standard Scaler', preprocessing.StandardScaler()]
            elif scaler == 'MinMax Scaler' :
                self.scaler = ['MinMax Scaler', preprocessing.MinMaxScaler(feature_range=(0, 1))]
            elif scaler == 'Normal Scaler':
                self.scaler = ['Normal Scaler', preprocessing.Normalizer()]

        self.binary = binary

        self.featureSelector = featureSelector

        self.dividePopulation = dividePopulation

        self.initialPath = 'Results/BinaryModelResults/Normal Scaler/feature' + str(featureSelector) + '/interactions' + str(interactions) + '/division' + str(dividePopulation) + '/'

        if not os.path.exists(self.initialPath):
            os.makedirs(self.initialPath)

        if dividePopulation is not None and interactions:
            raise Exception('Can not use interactions when dividing population')
        elif interactions:
            self.interactions = interactions

        print(self.models)


    def readAndPrepareData(self, filepath, numRows = None):
        print('Reading Data...')
        data = pandas.read_csv(filepath, nrows=numRows)

        if self.binary:
            for i, row in data.iterrows():
                classi = getattr(row, "CLASS")
                if classi == 2 or classi == 3 or classi == 4:
                    data.set_value(i, 'CLASS', 1)
                else:
                    data.set_value(i, 'CLASS', 0)

        if self.interactions:
            for i, row in data.iterrows():
                reg = getattr(row, "SUBTYPE REG")
                dial = getattr(row, "PURPOSE 8")
                job = getattr(row, "PURPOSE 7")
                school = getattr(row, "PURPOSE 5")
                data.set_value(i, 'REGDIAL', reg*dial)
                data.set_value(i, 'REGJOB', reg * job)
                data.set_value(i, 'REGSCHOOL', reg * school)

        if self.dividePopulation is not None:
            if self.dividePopulation == 0:
                data = data.loc[data['SUBTYPE REG'] == 0]
            elif self.dividePopulation == 1:
                data = data.loc[data['SUBTYPE REG'] == 1]


        self.data = data


        if self.scaler:
            self.scaleData(self.scaler)

        print('Dividing Data...')
        # Divide into Inputs and Outputs
        (X, y) = self.divideIO()

        X.to_csv(str(self.initialPath) + 'X.csv')

        print('Spliting Data...')
        # Add part of this chunk to test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)






        # if self.scaler:
        #     x = self.scaleData(self.scaler, data)
        #
        # print('Spliting Data...')
        # # Add part of this chunk to test data
        # self.trainData, self.testData = self.train_test_split(data, 2016)
        #
        #
        #
        # print('Dividing Data...')
        # # Divide into Inputs and Outputs
        # (self.X_train, self.y_train) = self.divideIO(self.trainData)
        # (self.X_test, self.y_test) = self.divideIO(self.testData)




        # #
        # # if self.featureSelector:
        # #     feature_selection = FeatureSelectionTests(x, y)
        # #     if self.featureSelector is 'chi2':
        # #         feature_selection.run_chi2()
        # #     elif self.featureSelector is 'rfe':
        # #         feature_selection.run_rfe()
        #
        # if self.interactions:
        #     self.generateInteractions(self.X_train)
        #     self.generateInteractions(self.X_test)
        #
        # #IN TEST
        # self.DEMAND_X_TEST = self.X_test.copy()
        #
        # del self.X_test['DATE_DAY']
        # del self.X_test['DATE_MONTH']
        # del self.X_test['DATE_YEAR']
        # del self.X_train['DATE_DAY']
        # del self.X_train['DATE_MONTH']
        # del self.X_train['DATE_YEAR']


    def trainModels(self):
        for model in self.models:
            print("Training Model: " + model[0])
            model[1].fit(self.X_train, self.y_train)

    def makePredictions(self):
            for model in self.models:
                print('Making Predictions for model ' + model[0])
                model.append(model[1].predict(self.X_test))
                model.append(model[1].predict_proba(self.X_test))


    def printModelAccuracyAndConfusionMatrix(self):

        accuracies = ''
        for model in self.models:
            accuracies =  accuracies + model[0] + " Prediction Accuracy: is %2.2f" % accuracy_score(self.y_test, model[2]) + '\n'
            accuracies = accuracies +"Confusion Matrix:" + '\n'
            accuracies = accuracies + str(confusion_matrix(self.y_test, model[2])) + '\n'
        if self.binary:
            text_file = open(self.initialPath + 'Model Accuracies.txt', "w")
        else:
            text_file = open('Results/4ModelResults/' + self.scaler[0] + '/ModelAccuracies.txt', "w")
        text_file.write(accuracies)
        text_file.close()
        print(accuracies)


    def printModelClassificationProbabilities(self):
        for model in self.models:
            print(model[0] + " Probabilities:")
            print(model[3])

        # getROCCurves(y_test, [['Logistic Regression', logProbabilities], ['Decision Trees', treeProbabilities],['Random Forest', rfProbabilities], ['Naive Bayes', NBProbabilities], ['MLP', MLPProbabilities], ['SVC', SVCProbabilities], ['KNN', KNNProbabilities]])

    def autopilot(self, filepath, nrows = None):
        self.readAndPrepareData(filepath, numRows=nrows)
        self.trainModels()
        self.makePredictions()
        self.printModelAccuracyAndConfusionMatrix()
        # if self.binary:
        #     self.saveDataForDemandModel()

    def getROCCurves(self):
        initialSavePath = ''
        areas = ''
        if self.binary:
            l = 0
            k = 1
            initialSavePath = self.initialPath
        else:
            l = 1
            k = 5
            initialSavePath = 'Results/4ModelResults/' + self.scaler[0]
        j = 1
        for model in self.models:
            areas = areas + 'ROC Curve Model: ' + model[0] + '\n'
            for i in range(l, k):
                if self.binary:
                    fpr, tpr, _ = roc_curve(self.y_test, model[3][:, i], i)
                else:
                    fpr, tpr, _ = roc_curve(self.y_test, model[3][:, i-1], i)
                plt.figure(j)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr, label= 'Class ' + str(i))
                areas = areas + 'Class ' + str(i) + ' (Area = %0.2f)' % auc(fpr, tpr) + '\n'
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve: ' + model[0])
                plt.legend(loc='best')
                plt.savefig(initialSavePath +'/' + model[0] + 'ModelCurve.png');

            plt.figure(10)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('All Models Performed ROC curve')
            fpr, tpr, _ = roc_curve(self.y_test, model[3][:, 0], l)
            plt.plot(fpr, tpr, label=model[0] )
            plt.legend(loc='best')
            plt.plot([0, 1], [0, 1], 'k--')



            j += 1

        plt.savefig(initialSavePath +'/AllModelsCurve.png');
        print(areas)
        text_file = open(initialSavePath  + '/Areas.txt', "w")
        text_file.write(areas)
        text_file.close()

    def divideIO(self):
        if self.interactions:
            if self.featureSelector is None:
                x = self.data[
                    ['AGE', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR',
                     'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                     'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                     'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                     'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                     'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                     'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                     'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                     'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                     'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
                     'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                     'pct trips canceled_period', 'pct trips late canceled_period',
                     'pct trips no-show_period', 'REGDIAL', 'REGJOB', 'REGSCHOOL' ]].copy()
            #Top 25 features
            elif self.featureSelector == 1:
                x = self.data[
                    ['pct trips canceled_period','pct trips late canceled_period','pct trips no-show_period','PURPOSE 8','Dis 9',
                     'trips before reservation','ANTICIPATED','MALE','SUBTYPE DEM','Dis 14','SUBTYPE REG','AGE','Dis 36','PURPOSE 1',
                     'PURPOSE 12','PURPOSE 7','Dis 26','Dis 8','PURPOSE 10','Dis 6','Dis 3','Dis 37','Sunday','Dis 5','PURPOSE 4', 'REGDIAL', 'REGJOB', 'REGSCHOOL']].copy()
            # Top 25% features
            elif self.featureSelector == 2:
                x = self.data[
                    ['trips before reservation','pct trips canceled_period','PURPOSE 8','AGE','pct trips late canceled_period',
                     'Dis 9','ANTICIPATED','BLOCK','Dis 14','MALE','SUBTYPE REG','Dis 36','PURPOSE 12','Dis 26','PURPOSE 1',
                     'pct trips no-show_period','Dis 8','PURPOSE 7','PURPOSE 10','Sunday','Dis 37', 'REGDIAL', 'REGJOB', 'REGSCHOOL']].copy()
            # 0.05 p-value threshold
            elif self.featureSelector == 3:
                x = self.data[
                    ['AGE', 'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                     'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                     'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                     'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                     'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                     'MobAid SC', 'MobAid SR', 'MobAid SRE', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                     'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                     'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                     'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 14',
                     'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                     'pct trips canceled_period', 'pct trips late canceled_period',
                     'pct trips no-show_period', 'REGDIAL', 'REGJOB', 'REGSCHOOL']].copy()
        else:
            if self.dividePopulation != None:
                if self.featureSelector is None:
                    x = self.data[
                        ['AGE', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR',
                         'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                         'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                         'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO',
                         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                         'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
                         'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                         'pct trips canceled_period', 'pct trips late canceled_period',
                         'pct trips no-show_period']].copy()
                #Top 25 features
                elif self.featureSelector == 1:
                    x = self.data[
                        ['pct trips canceled_period','pct trips late canceled_period','pct trips no-show_period','PURPOSE 8','Dis 9',
                         'trips before reservation','ANTICIPATED','MALE','AGE','Dis 36','PURPOSE 1',
                         'PURPOSE 12','PURPOSE 7','Dis 26','Dis 8','PURPOSE 10','Dis 6','Dis 3','Dis 37','Sunday','Dis 5','PURPOSE 4']].copy()
                # Top 25% features
                elif self.featureSelector == 2:
                    x = self.data[
                        ['trips before reservation','pct trips canceled_period','PURPOSE 8','AGE','pct trips late canceled_period',
                         'Dis 9','ANTICIPATED','BLOCK','Dis 14','MALE','Dis 36','PURPOSE 12','Dis 26','PURPOSE 1',
                         'pct trips no-show_period','Dis 8','PURPOSE 7','PURPOSE 10','Sunday','Dis 37']].copy()
                # 0.05 p-value threshold
                elif self.featureSelector == 3:
                    x = self.data[
                        ['AGE', 'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                         'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                         'MobAid SC', 'MobAid SR', 'MobAid SRE',
                         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                         'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 14',
                         'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                         'pct trips canceled_period', 'pct trips late canceled_period',
                         'pct trips no-show_period']].copy()
            else:
                if self.featureSelector is None:
                    x = self.data[
                        ['AGE', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR',
                         'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18',
                         'Dis 19',
                         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25',
                         'Dis 26',
                         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34',
                         'Dis 35',
                         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8',
                         'Dis 9',
                         'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                         'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG',
                         'SUBTYPE SBY',
                         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                         'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
                         'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                         'pct trips canceled_period', 'pct trips late canceled_period',
                         'pct trips no-show_period']].copy()
                # Top 25 features
                elif self.featureSelector == 1:
                    x = self.data[
                        ['pct trips canceled_period', 'pct trips late canceled_period', 'pct trips no-show_period',
                         'PURPOSE 8', 'Dis 9',
                         'trips before reservation', 'ANTICIPATED', 'MALE', 'SUBTYPE DEM', 'Dis 14', 'SUBTYPE REG',
                         'AGE', 'Dis 36', 'PURPOSE 1',
                         'PURPOSE 12', 'PURPOSE 7', 'Dis 26', 'Dis 8', 'PURPOSE 10', 'Dis 6', 'Dis 3', 'Dis 37',
                         'Sunday', 'Dis 5', 'PURPOSE 4']].copy()
                # Top 25% features
                elif self.featureSelector == 2:
                    x = self.data[
                        ['trips before reservation', 'pct trips canceled_period', 'PURPOSE 8', 'AGE',
                         'pct trips late canceled_period',
                         'Dis 9', 'ANTICIPATED', 'BLOCK', 'Dis 14', 'MALE', 'SUBTYPE REG', 'Dis 36', 'PURPOSE 12',
                         'Dis 26', 'PURPOSE 1',
                         'pct trips no-show_period', 'Dis 8', 'PURPOSE 7', 'PURPOSE 10', 'Sunday', 'Dis 37']].copy()
                # 0.05 p-value threshold
                elif self.featureSelector == 3:
                    x = self.data[
                        ['AGE', 'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17',
                         'Dis 18', 'Dis 19',
                         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25',
                         'Dis 26',
                         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                         'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                         'MobAid SC', 'MobAid SR', 'MobAid SRE', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                         'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 14',
                         'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                         'pct trips canceled_period', 'pct trips late canceled_period',
                         'pct trips no-show_period']].copy()

                else:
                    raise Exception("No X")

        # Select Output
        y = self.data[['CLASS']].copy().values.ravel()

        return x, y

    def scaleData(self, scaler):

        self.data[['AGE',
              'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
              'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
              'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
              'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
              'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
              'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
              'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
              'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
              'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
              'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
              'pct trips canceled_period', 'pct trips late canceled_period',
              'pct trips no-show_period']] \
            = scaler[1].fit_transform(self.data[['AGE',
                                            'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16',
                                            'Dis 17', 'Dis 18', 'Dis 19',
                                            'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23',
                                            'Dis 24', 'Dis 25', 'Dis 26',
                                            'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32',
                                            'Dis 33', 'Dis 34', 'Dis 35',
                                            'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5',
                                            'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9'
            , 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                                            'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM',
                                            'SUBTYPE REG', 'SUBTYPE SBY',
                                            'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                                            'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6',
                                            'PURPOSE 7', 'PURPOSE 8',
                                            'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13',
                                            'PURPOSE 14',
                                            'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                                            'pct trips canceled_period', 'pct trips late canceled_period',
                                            'pct trips no-show_period']])

        return self.data



    def train_test_split(self, data, year):

        trainData = pandas.DataFrame()
        testData = pandas.DataFrame()

        trainData = trainData.append(data[data['DATE_YEAR'] < year])
        testData = testData.append(data[data['DATE_YEAR'] >= year])

        return(trainData, testData)

    def saveDataForDemandModel(self):
        # Saves data for demand model
        for model in self.models:
            data = self.DEMAND_X_TEST.loc[:, ['DATE_DAY', 'DATE_MONTH', 'DATE_YEAR']]
            data['y'] = self.y_test
            data['y_hat'] = pandas.Series(model[2], index=data.index)
            data['pr'] = pandas.Series(model[3][:, 0], index=data.index)

            data.to_csv(
                '../demandData/' + model[0] + '_' + self.scaler[0] + '_' + str(self.binary) + '_demandData.csv',
                index=False)