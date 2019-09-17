
#Works for any model

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm


from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix



#Create data frame from csv file
reservations =  pd.read_csv("C:/Users/jaime/Documents/updated_data.csv", na_filter=False)
reservations = reservations.sample(n = 200000, random_state = 10000)



toDel = ['ID', 'SCHEDULE STATUS', 'FUNDING SOURCE 1', 'Create_MONTH', 'Create_YEAR', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR', 'BIRTH_YEAR', 'FUNDING SOURCE 2', 'Create_DAY', 'SEX NOT SPECIFIED','SUBTYPE WCL', 'PURPOSE 14', 'MALE', 'Wednesday']
for col in toDel:
    del reservations[col]



X, y = reservations.loc[:,reservations.columns != 'CLASS'], reservations['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1337)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



def apply_classifier(model,modelName,X_train, y_train, X_test, y_test):
    startTime = time.monotonic()
    model.fit(X_train, y_train)
    print(modelName)
    print('Training accuracy:', model.score(X_train, y_train))
    print('Test accuracy:', model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    duration = time.monotonic() - startTime
    print('Running time:','{:.3f} minutes'.format( duration / 60.0 ))
    print('')



# First try our classifiers with all features.
models = []

models.append([RandomForestClassifier(n_estimators = 500, random_state = 8888),"Random Forest"])
models.append([GaussianNB(),"Naive Bayes"])
models.append([LogisticRegression(penalty = 'l1', C = 0.01),"Logistic Regression"])
#SVM has a complexity of O(n_samples^2 * n_features)
models.append([svm.SVC(kernel='rbf', C=0.5, gamma=1),"SVM"])
#KNN has a complexity of O(n_samples^2 * n_features) or O[D N *log(N)] when k is specified
models.append([KNeighborsClassifier(n_neighbors = 5),"KNN"])

for model,modelName in models:
    apply_classifier(model,modelName,X_train_std, y_train, X_test_std, y_test)

