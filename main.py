from datetime import date
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta

from MLAlgorithms import MLAlgorithms
import numpy as np
import pandas
from RegNoRegEnsemble import RegNoRegEnsemble


# m = RegNoRegEnsemble()
# m.readAndPrepareData('updated_data.csv')
# m.trainModels()
# m.makePredictions()
# m.printModelAccuracyAndConfusionMatrix()
#
# models = MLAlgorithms(['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'MLP Classifier',
#                        'Gradient Boosting', 'Voting', 'Bagging', 'AdaBoost'], scaler='Normal Scaler', binary=True, dividePopulation=1)
#
# models.autopilot('updated_data.csv')
# models.getROCCurves()
#
# data = pandas.read_csv('updated_data.csv', nrows=None)
# print("1")
#
# for i, row in data.iterrows():
#     classi = getattr(row, "CLASS")
#     if classi == 2 or classi == 3 or classi == 4:
#         data.set_value(i, 'CLASS', 1)
#     else:
#         data.set_value(i, 'CLASS', 0)
#
# print("2")
#
# for i, row in data.iterrows():
#     reg = getattr(row, "SUBTYPE REG")
#     dial = getattr(row, "PURPOSE 8")
#     job = getattr(row, "PURPOSE 7")
#     school = getattr(row, "PURPOSE 5")
#     data.set_value(i, 'REGDIAL', reg * dial)
#     data.set_value(i, 'REGJOB', reg * job)
#     data.set_value(i, 'REGSCHOOL', reg * school)
# print("3")
# data[['AGE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
#               'pct trips canceled_period', 'pct trips late canceled_period',
#               'pct trips no-show_period']] \
#             = preprocessing.Normalizer().fit_transform(data[['AGE', 'BLOCK', 'ANTICIPATED', 'trips before reservation',
#                                             'pct trips canceled_period', 'pct trips late canceled_period',
#                                             'pct trips no-show_period']])
# print("4")
#
#
# trainInitDate = date(2014, 1, 1)
# trainFinDate = date(2016, 4, 30)
# testInitDate = date(2016, 5, 1)
# testFinDate = date(2017, 12, 31)
# print("5")
#
# dataTrain = data.loc[((data['DATE_YEAR'] >= 2014) & (data['DATE_YEAR'] < 2016)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 4)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 3))| ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 2)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 1))]
# print("6")
#
# dataTest = data.loc[((data['DATE_YEAR'] == 2016) & (data['DATE_MONTH'] == 5)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 6)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 7)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 8)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 9)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 10)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 11)) | ((data['DATE_YEAR'] == 2016)& (data['DATE_MONTH'] == 12)) | (data['DATE_YEAR'] ==2017)]
#
#
# # for i, row in data.iterrows():
# #     print(i)
# #     rowDate = date(int(getattr(row,'DATE_YEAR')), int(getattr(row,'DATE_MONTH')), int(getattr(row, 'DATE_DAY')))
# #     if rowDate >= trainInitDate and rowDate <= trainFinDate:
# #         dataTrain = dataTrain.append(row, ignore_index=True)
# #     elif rowDate >= testInitDate and rowDate <= testFinDate:
# #         dataTest = dataTest.append(row, ignore_index=True)
#
# print("Gothere")
#
# dataTrain.to_csv('dataTrain.csv')
# dataTest.to_csv('dataTest.csv')
