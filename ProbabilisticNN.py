import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the data points
from sklearn.preprocessing import StandardScaler, Normalizer

print("READING DATA")
data = pd.read_csv('updated_data.csv', sep=',', header=0)

selected_features = data.filter(items=['Dis 1', 'Dis 10', 'Dis 11', 'Dis 12', 'Dis 15', 'Dis 17', 'Dis 18', 'Dis 19', 'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 24', 'Dis 25', 'Dis 26', 'Dis 27', 'Dis 29', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35', 'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 40', 'Dis 5', 'Dis 7', 'Dis 8', 'MobAid AB', 'MobAid AR', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG', 'MobAid SC', 'MobAid TO', 'SUBTYPE SBY', 'Sunday', 'PURPOSE 0', 'PURPOSE 3', 'PURPOSE 6', 'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 13'])


# Declare the needed variable
y = data['CLASS']
x = selected_features

print("SPLITTING DATA")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=27)

print("NORMALIZING DATA")
scaler = Normalizer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# apply same transformation to test data
x_test = scaler.transform(x_test)

print("FITTING DATA")
nnw = MLPClassifier(hidden_layer_sizes=(50, ), solver="adam", max_iter=2000, alpha=0.0001,
                    verbose=True, warm_start=True, tol=0.000000001)

nnw.fit(x_train, y_train)
y_pred = nnw.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print('\nAccuracy Score:', accuracy)

time.sleep(1)
