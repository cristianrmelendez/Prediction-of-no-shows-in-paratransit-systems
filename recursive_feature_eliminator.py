import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier

print("READING DATA")
data = pd.read_csv('updated_data.csv', sep=',', header=0)

y = data['CLASS']
x = data.filter(items=['Dis 1', 'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
         'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
         'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
         'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
         'MobAid AB', 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
         'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
         'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
         'PURPOSE 0', 'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
         'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13',
         'FEMALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation', 'pct performed_period', 'pct canceled_period',
         'pct late canceled_period', 'AGE'])

startTime = time.time()

models = [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier()]

print("FITTING MODELS")
for model in models:
    print()
    print(model.__class__.__name__)
    # Run RFE for the current model
    rfe = RFE(model, 50)
    fit = rfe.fit(x, y)

    headers = list(x)
    selected = list()

    for i, f in enumerate(fit.support_):
        if f:
            selected.append(headers[i])
            print(headers[i])


endTime = time.time()
print("Elapsed Time:", endTime - startTime)