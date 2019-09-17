
# coding: utf-8

# In[1]:


#Import Library
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[3]:


#Create data frame from csv file
reservations =  pd.read_csv("C:/Users/jaime/Documents/preprocessed-data.csv", na_filter=False)


# In[4]:


X, y = reservations.loc[:,reservations.columns != 'CLASS'], reservations['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1337)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[5]:


def apply_classifiers(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)
    print('KNN:')
    print('Training accuracy:', knn.score(X_train, y_train))
    print('Test accuracy:', knn.score(X_test, y_test))
    print('')
    
    forest = RandomForestClassifier(n_estimators = 500, random_state = 8888)
    forest.fit(X_train, y_train)
    print('Random Forest:')
    print('Training accuracy:', forest.score(X_train, y_train))
    print('Test accuracy:', forest.score(X_test, y_test))
    print('')
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print('Naive Bayes:')
    print('Training accuracy:', nb.score(X_train, y_train))
    print('Test accuracy:', nb.score(X_test, y_test))
    print('')
    
    lr = LogisticRegression(penalty = 'l1', C = 0.1)
    lr.fit(X_train, y_train)
    print('Logistic Regression:')
    print('Training accuracy:', lr.score(X_train, y_train))
    print('Test accuracy:', lr.score(X_test, y_test))
    print('')


# In[ ]:


# First try our classifiers with all features.
print('Testing with all features...')
apply_classifiers(X_train_std, y_train, X_test_std, y_test)

