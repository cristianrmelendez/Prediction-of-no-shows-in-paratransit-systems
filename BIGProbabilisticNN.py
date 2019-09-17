# 0. import the needed packages
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Read the data points
data = pd.read_csv('updated_data.csv', sep=',', header=0)

cleaned_data = data.filter(items=['pct late canceled_period', 'pct no-show_period', 'PURPOSE 8', 'SUBTYPE DEM', 'BLOCK',
                                  'FEMALE', 'pct canceled_period', 'trips before reservation', 'ANTICIPATED',
                                  'PURPOSE 1', 'Dis 9', 'MobAid BT', 'Monday', 'SUBTYPE REG', 'Tuesday', 'Dis 6',
                                  'PURPOSE 0', 'PURPOSE 10', 'Friday', 'Thursday', 'MobAid AD', 'Dis 3'])


# 2. Declare the needed variable
y = data['CLASS']
x = cleaned_data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

nnw = MLPClassifier(hidden_layer_sizes=(300, 300, 300), activation='logistic', max_iter=650, learning_rate='adaptive',
                    alpha=0.00001, solver='adam', verbose=True, random_state=21, tol=0.000000001, warm_start=True)

nnw.fit(x_train, y_train)
y_pred = nnw.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

#cm = confusion_matrix(y_test, y_pred)

#cm

print('the score is: ', accuracy)
