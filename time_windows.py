import pandas as pd
import numpy as np
from base_share_classifier import base_share_rand_classifier
import statsmodels.api as sm
from ols import ols_weight

import time
from datetime import date, timedelta

import warnings

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, \
    BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from dateutil.relativedelta import relativedelta




warnings.filterwarnings('ignore')

all_predicted = []
all_expected = []
all_share = []
all_real = []

isBinary = True  # set binary or multi class (default = True)
users = 1           # 0 = all, 1 = reg, 2 = non reg

print("READING DATA")
data = pd.read_csv('updated_data.csv', sep=',', header=0, nrows=None)
data.astype('float')
numOfTripsPerDay = list()
models = list()
results = list()
results.append(['date', 'predicted','expected', 'share',  'ensemble' ,'real'])

# if isBinary:
#     models.append(('LogisticRegression', LogisticRegression(solver='sag')))
#     # models.append(('GaussianNB', GaussianNB()))
# else:
#     models.append(('LogisticRegression', LogisticRegression(multi_class='multinomial', solver='sag')))
    # models.append(('MultinomialNB', MultinomialNB()))

# models.append(('KNearestNeighbors', KNeighborsClassifier(n_neighbors=5)))
# models.append(('DecisionTree', DecisionTreeClassifier(class_weight='balanced')))
models.append(('RandomForest', RandomForestClassifier(class_weight='balanced')))
#
# models.append(('SupportVector', SVC(kernel='rbf', C=0.5, gamma=1, probability=True)))
# # models.append(('MLPClassifier', MLPClassifier(max_iter=2000, solver="adam")))
# models.append(('AdaBoostClassifier', AdaBoostClassifier()))
# models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
# models.append(('BaggingClassifier', BaggingClassifier(LogisticRegression())))
# models.append(('VotingClassifier', VotingClassifier(
#     estimators=[('Logistic Regression', LogisticRegression(solver='sag')),
#                 ('Decision Tree', DecisionTreeClassifier(class_weight='balanced')),
#                 ('Random Forest', RandomForestClassifier(class_weight='balanced'))], voting='soft')))

# set data to binary (1 = performed, 0 = not-performed)
if isBinary:
    for index, row in data.iterrows():
        if row['CLASS'] != 1:
            data.set_value(index, 'CLASS', 0)

performedIndex = 1  # default (binary)
if not isBinary:
    performedIndex = 0


if users == 1:
    data = data.loc[data['SUBTYPE REG'] == 1]
elif users == 2:
    data = data.loc[data['SUBTYPE REG'] == 0]
else:
    pass

# t = (90, 180, 365, 730)  # time windows (in days)
t = (365,)  # time windows (in days)

d0 = date(2016, 1, 1)  # simulation start date
# D = (d0, )
D = list()

d = d0
delta = timedelta(days=1)
while d <= date(2016, 12, 31):
    D.append(d)
    d += delta


startTime = time.time()

#print("Date", "ShareDemand", "LogisticRegression", "RandomForest")

for k in t:
    for d in D:
        timeBefore = d - relativedelta(days=k)

        dataTrain = data.loc[(((data['DATE_YEAR'] == timeBefore.year) & (data['DATE_MONTH'] == timeBefore.month) & (data['DATE_DAY'] >= timeBefore.day)) | ((data['DATE_YEAR'] == timeBefore.year) & (data['DATE_MONTH'] > timeBefore.month)) | ((data['DATE_YEAR'] > timeBefore.year))) & (((data['DATE_YEAR'] == d.year) & (data['DATE_MONTH'] == d.month) & (data['DATE_DAY'] < d.day)) | ((data['DATE_YEAR'] == d.year) & (data['DATE_MONTH'] < d.month)) | (data['DATE_YEAR'] < d.year))]

        dataTest = data.loc[
            ((data['DATE_YEAR'] == d.year) & (data['DATE_MONTH'] == d.month) & (data['DATE_DAY'] == d.day))]

        Y_train = dataTrain[['CLASS']].copy()
        X_train = dataTrain[[
            'AGE', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR',
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
            'pct trips no-show_period']].copy()

        Y_validation = dataTest[['CLASS']].copy()
        X_validation = dataTest[[
            'AGE', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR',
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
            'pct trips no-show_period']].copy()

        # print("\nw =", k, ":", count, "training samples for day", d)

        results_str = str(d)


        try:

            # share demand calculation algorithm
            share = np.sum(Y_train['CLASS'] == performedIndex)/ Y_train.size



            # real demand
            real_demand = np.sum(Y_validation['CLASS'] == performedIndex)
            # print("share:", share)
            share_demand = Y_validation.size * share


            # print("share_demand:", share_demand)
            error = (np.absolute((share_demand - real_demand)) / real_demand) * 100

            if real_demand == 0:
                error = 0

            for name, model in models:
                # print("\n--"+name+"--")
                model.fit(X_train, Y_train.values.ravel())
                predictions = model.predict(X_validation)
                # print("\nAccuracy Score: %f" % accuracy_score(Y_validation, predictions))
                # print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
                # print(classification_report(Y_validation, predictions))

                performed_prob = model.predict_proba(X_validation)[:, performedIndex]
                real_demand = np.sum(Y_validation['CLASS'] == performedIndex)
                expected_demand = np.sum(performed_prob)
                predicted_demand = np.sum(predictions == performedIndex)

                # Calculating the Difference
                if real_demand == 0:  # Percentage of Difference Not Defined.
                    if expected_demand == 0:
                        diff_1 = 0
                    else:
                        diff_1 = 100

                    if predicted_demand == 0:
                        diff_2 = 0
                    else:
                        diff_2 = 100
                else:
                    # Difference Between Expected Value and Real Demand
                    diff_1 = (np.absolute((expected_demand - real_demand) / real_demand) * 100)
                    # Difference Between Predicted Class and Real Demand
                    diff_2 = (np.absolute((predicted_demand - real_demand) / real_demand) * 100)

                w = ols_weight(data, k, timeBefore, d - relativedelta(days=1))
                ensemble_demand = np.dot(w, [1, predicted_demand, expected_demand, share_demand])

                results.append([d, predicted_demand, expected_demand, share_demand, ensemble_demand, real_demand])

                results_str += " " + str(predicted_demand) + " " + str(expected_demand) + " " + str(
                    share_demand) + " " + str(ensemble_demand) + " " + str(real_demand)

            print(results_str)



        except Exception:
            print("caught exception")
            results_str += " Error: No Data for this window"
            print(results_str)


resultsDF = pd.DataFrame(results)
resultsDF.to_csv('TimeWindowsOlSReport.csv')




#
#
# G = list()
# O = all_real
# # G = [[1,2,3,2], [1,2,2,3]]
# # OLS weight calcs
# for i in range(0, len(all_predicted)):
#     G.append([1, all_predicted[i], all_expected[i], all_share[i]])
#
# w = list()
#
# for i in range(0, len(all_predicted)):
#     w.append(ols_weight(G[i:9+i], O[i:9+i]))
#
# ensemble = list()
#
# for i in range(0, len(all_predicted)):
#     ensemble.append((w[i][0] * 1 + all_predicted[i] * w[i][1] + all_expected[i]*w[i][2] + w[i][3] * all_share[i]))
#
# print(ensemble)
#
# data = list()
# data.append(['date', 'real','predicted', 'predicted difference',  'expected' ,'expected difference', 'share' ,'share difference', 'ensemble' ,'ensemble difference', 'random', 'random difference'])
# d = date(2016, 1, 1)
# count = 0
# delta = timedelta(days=1)
#
# print("noftripsperday:", len(all_real))
# print("w count:", len(w))
# print("ensemble count:", len(ensemble))
# print("all_real count:", len(all_real))
#
# while d <= date(2016, 12, 31):
#     try:
#         print(count, d)
#         # print(all_real[count])
#         rand = base_share_rand_classifier(all_real[count], w[count])[1]
#         data.append([d, all_real[count], all_predicted[count], abs(all_predicted[count] - all_real[count]) /
#                      all_real[count], all_expected[count], abs(all_expected[count] - all_real[count]) /
#                      all_real[count], all_share[count], abs(all_share[count] - all_real[count]) /
#                      all_real[count], ensemble[count], abs(ensemble[count] - all_real[count]) /
#                      all_real[count], rand, abs(rand - all_real[count]) / all_real[count]])
#         print([d, all_real[count], all_predicted[count], abs(all_predicted[count] - all_real[count]) /
#                all_real[count], all_expected[count], abs(all_expected[count] - all_real[count]) /
#                all_real[count], all_share[count], abs(all_share[count] - all_real[count]) /
#                all_real[count], ensemble[count], abs(ensemble[count] - all_real[count]) /
#                all_real[count], rand, abs(rand - all_real[count]) / all_real[count]])
#     except ZeroDivisionError:
#         data.append([d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
#     d += delta
#     count += 1
#
# results = pd.DataFrame(data)
# print(results)
