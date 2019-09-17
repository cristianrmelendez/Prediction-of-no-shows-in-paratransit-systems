from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt


def divideIO(data):
    ################################################################
    # This function must be updated is data file structure changes #
    ################################################################



    x = data[
        ['AGE',
         'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9'
            , 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
         'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
         'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
         'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
         'pct trips canceled_period', 'pct trips late canceled_period',
         'pct trips no-show_period']].copy()

    # Select Output
    y = data[['CLASS']].copy().values.ravel()

    return x, y


def makePredictions(testData, model):
    print('Making Predictions for model: ' + model[0])
    model.append(model[1].predict(testData))
    model.append(model[1].predict_proba(testData))



def scaleData(scaler, data, partial):
    print("Scaling data with: " + scaler[0])
    if partial:
        data[['BLOCK', 'ANTICIPATED', 'trips before reservation', 'pct performed', 'pct canceled', 'pct late canceled',
              'pct no-show']] \
            = scaler[1].fit_transform(data[['BLOCK', 'ANTICIPATED',
                                         'trips before reservation',
                                         'pct performed', 'pct canceled',
                                         'pct late canceled', 'pct no-show']])
    else:
        data[[ 'AGE',
                'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9'
                , 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                 'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
                'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                'pct trips canceled_period', 'pct trips late canceled_period',
                'pct trips no-show_period']] \
            = scaler[1].fit_transform(data[[ 'AGE',
                'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9'
                , 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                 'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14',
                'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
                'pct trips canceled_period', 'pct trips late canceled_period',
                'pct trips no-show_period']])

    return data


def printModelClassificationProbabilities(testData, model):
    print(model[0] + " Probabilities:")
    print(model[3])


def printModelAccuracyAndConfusionMatrix(testData, model):
    print(model[0] +" Prediction Accuracy: is %2.2f" % accuracy_score(testData, model[2]))
    print("Confusion Matrix:")
    print(confusion_matrix(testData, model[2]))



def trainModel(trainData_X, trainData_Y, model):
    print("Training Model: " + model[0])
    model[1].fit(trainData_X, trainData_Y)



def getROCCurves(outputs, modelsProbabilities):
    j = 1
    for modelProbabilities in modelsProbabilities:
        try:
            plt.figure(j)
            for i in range(1,5):
                fpr, tpr, _ = roc_curve(outputs, modelProbabilities[1][:, i-1], i)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr, label=i)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve: ' + modelProbabilities[0])
                plt.legend(loc='best')



            plt.figure(10)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('All Models Class 1 ROC curve')
            fpr, tpr, _ = roc_curve(outputs, modelProbabilities[1][:, 0], 1)
            plt.plot(fpr, tpr, label= modelProbabilities[0])

            plt.legend(loc='best')
            plt.plot([0, 1], [0, 1], 'k--')
        except:
            j+=1

    plt.show();

def getROCCurvesBinary(outputs, modelsProbabilities):
    j = 1
    for modelProbabilities in modelsProbabilities:
        try:


            for i in range(1,3):
                fpr, tpr, _ = roc_curve(outputs, modelProbabilities[1][:, i-1], i)
                plt.figure(j)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr, label=i)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve: ' + modelProbabilities[0])
                plt.legend(loc='best')

            plt.figure(10)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('All Models Class 1 ROC curve')
            fpr, tpr, _ = roc_curve(outputs, modelProbabilities[1][:, 0], 1)
            plt.plot(fpr, tpr, label=modelProbabilities[0])

            plt.legend(loc='best')
            plt.plot([0, 1], [0, 1], 'k--')
        except:
            print(modelProbabilities[0] + ' is None')

        j += 1

    plt.show();
