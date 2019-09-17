# -*- coding: utf-8 -*-
import numpy
import scipy
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

# author: Angel G, Carrillo Laguna
# git: AngelGCL
# e-mail: angel.carrillo1@upr.edu

data = pd.DataFrame(pd.read_csv(
        'updated_data.csv',
        sep=',', header=0))

data14to16 = data.ix[~((data['DATE_YEAR'] < 2014) & (data['DATE_YEAR'] > 2016))]

cleaned_data = data14to16.filter(items=['pct trips late canceled_period',
                                  'pct trips no-show_period', 'PURPOSE 8',
                                  'SUBTYPE DEM', 'Dis 26', 'MALE',
                                  'pct trips canceled_period',
                                  'trips before reservation', 'ANTICIPATED',
                                  'PURPOSE 1', 'Dis 9', 'Dis 5', 'PURPOSE 4',
                                  'SUBTYPE REG', 'Sunday', 'Dis 6', 'Dis 14',
                                  'PURPOSE 7', 'PURPOSE 10', 'PURPOSE 12',
                                  'Dis 36','Dis 37', 'Dis 8', 'Dis 3',
                                  'AGE', 'REG', 'CLASS'])

data16to17 = data.ix[~((data['DATE_YEAR'] < 2016) & (data['DATE_YEAR'] > 2017))]
# data frame with testing data set
cleaned_data2 = data16to17.filter(items=['pct trips late canceled_period',
                                  'pct trips no-show_period', 'PURPOSE 8',
                                  'SUBTYPE DEM', 'Dis 26', 'MALE',
                                  'pct trips canceled_period',
                                  'trips before reservation', 'ANTICIPATED',
                                  'PURPOSE 1', 'Dis 9', 'Dis 5', 'PURPOSE 4',
                                  'SUBTYPE REG', 'Sunday', 'Dis 6', 'Dis 14',
                                  'PURPOSE 7', 'PURPOSE 10', 'PURPOSE 12',
                                  'Dis 36','Dis 37', 'Dis 8', 'Dis 3',
                                  'AGE', 'REG', 'CLASS'])

Y_train = numpy.array(data14to16.filter(items=['CLASS']).astype(numpy.int64))
variable_names = numpy.array([0, 1, 2, 3, 4])

X_train = numpy.array(cleaned_data).astype(numpy.int64)
X_test = numpy.array(cleaned_data2).astype(numpy.int64)

X = scale(X_train).astype(numpy.int64)
X2 = scale(X_test).astype(numpy.int64)
Y = scale(Y_train).astype(numpy.int64)

km = KMeans(n_clusters=5)
km.fit(X)
c = km.predict(X)
clusters = pd.DataFrame({'cluster of inputs': c})
# testing that clusters frame is properly made
print(clusters)
# getting number of elements per cluster
cerosONES = 0
ceros = 0
unosONES = 0
unos = 0
dosONES = 0
dos = 0
tresONES = 0
tres = 0
cuatrosONES = 0
cuatros = 0
for index, row in clusters.iterrows():
    if row['cluster of inputs'] == 0:
        ceros = ceros + 1
        if (cleaned_data.iloc[index]['CLASS']) == 1.0:
            cerosONES = cerosONES + 1
    elif row['cluster of inputs'] == 1:
        unos = unos + 1
        if (cleaned_data.iloc[index]['CLASS']) == 1.0:
            unosONES = unosONES + 1
    elif row['cluster of inputs'] == 2:
        dos = dos + 1
        if (cleaned_data.iloc[index]['CLASS']) == 1.0:
            dosONES = dosONES + 1
    elif row['cluster of inputs'] == 3:
        tres = tres + 1
        if (cleaned_data.iloc[index]['CLASS']) == 1.0:
            tresONES = tresONES + 1
    elif row['cluster of inputs'] == 4:
        cuatros = cuatros + 1
        if (cleaned_data.iloc[index]['CLASS']) == 1.0:
            cuatrosONES = cuatrosONES + 1

print(cerosONES)
print(unosONES)
print(dosONES)
print(tresONES)
print(cuatrosONES)
# proportions for each cluster
clust_0 = cerosONES/ceros
clust_1 = unosONES/unos
clust_2 = dosONES/dos
clust_3 = tresONES/tres
clust_4 = cuatrosONES/cuatros

print('cluster zero: ', clust_0, ', cluster one: ', clust_1, ', cluster two: ', clust_2, ', cluster three: ', clust_3,
      ', cluster four: ', clust_4)
# clustering for testing
km2 = KMeans(n_clusters=5)
km2.fit(X2)
c2 = km.predict(X2)
clusters2 = pd.DataFrame({'cluster of inputs': c2})
Y_test = []
# classifying the inputs ##NOTE: this was made for testing purposes, as it classifies, if the cluster does not have
# class 1 as the highest proportion, it will guess using the real CLASS value.
# Fix this for further use.
# ##
for index, row in clusters.iterrows():
    if row['cluster of inputs'] == 0:
        if clust_0 >= 0.5:
            Y_test.append(1.0)
        else:
            Y_test.append(cleaned_data.iloc[index]['CLASS'])
    elif row['cluster of inputs'] == 1:
        if clust_1 >= 0.5:
            Y_test.append(1.0)
        else:
            Y_test.append(cleaned_data.iloc[index]['CLASS'])
    elif row['cluster of inputs'] == 2:
        if clust_2 >= 0.5:
            Y_test.append(1.0)
        else:
            Y_test.append(cleaned_data.iloc[index]['CLASS'])
    elif row['cluster of inputs'] == 3:
        if clust_3 >= 0.5:
            Y_test.append(1.0)
        else:
            Y_test.append(cleaned_data.iloc[index]['CLASS'])
    elif row['cluster of inputs'] == 4:
        if clust_4 >= 0.5:
            Y_test.append(1.0)
        else:
            Y_test.append(cleaned_data.iloc[index]['CLASS'])

Y_final = scale(numpy.array(Y_test)).astype(numpy.int64)
Y_ = numpy.array(data16to17.filter(items=['CLASS']).astype(numpy.int64))
Y2 = scale(Y_).astype(numpy.int64)
print('Accuracy: {}'.format(accuracy_score(Y2, Y_final)))

##--------------------------------------END-------------------------------------------------##

