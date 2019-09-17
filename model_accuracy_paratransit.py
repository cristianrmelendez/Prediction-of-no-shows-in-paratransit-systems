import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "train_data.csv"
names = ['ID', 'Purpose id', 'Funding Source1', 'Funding Source2', 'Mobility Aid1', 'Mobility Aid2', 'Mobility Aid3',
         'Disability1', 'Disability2', 'Disability3', 'Disability4', 'Disability5', 'Disability6', 'Disability7',
         'Disability8', 'Gender', 'Birth Date', 'Create_DAY', 'Create_MONTH', 'Create_YEAR', 'DATE_DAY', 'DATE_MONTH',
         'DATE_YEAR', 'DAYS_OF_ANTICIPATION', 'BIRTH_DAY', 'BIRTH_MONTH', 'BIRTH_YEAR', 'CLASS']

df = pd.read_csv(url)

df = df.drop(columns=['ID', 'Birth Date', 'Create_DAY', 'Create_MONTH', 'Create_YEAR', 'DATE_DAY', 'DATE_MONTH',
                      'DATE_YEAR', 'BIRTH_DAY', 'BIRTH_MONTH'])

df.astype('float')

# Split-out validation dataset
array = df.values
X = array[:, 0:17]
Y = array[:, 17]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
models = list()
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset

print("\n--LOGISTIC REGRESSION--")
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print("\nAccuracy Score: %f" % accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("\n--NAIVE BAYES--")
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print("\nAccuracy Score: %f" % accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print("\n--DECISION TREE CLASSIFIER--")
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print("\nAccuracy Score: %f" % accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
