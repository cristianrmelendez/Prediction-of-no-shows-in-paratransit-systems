import statsmodels.api as sm
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
import numpy as np

def ols_weight(data, k, d0, d1):
    model = RandomForestClassifier(class_weight='balanced')
    G = []
    O = []
    d = d0

    while d <= d1:

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
            share = np.sum(Y_train['CLASS'] == 1) / Y_train.size
            # real demand
            real_demand = np.sum(Y_validation['CLASS'] == 1)
            # print("share:", share)
            share_demand = Y_validation.size * share
            # print("share_demand:", share_demand)
            error = (np.absolute((share_demand - real_demand)) / real_demand) * 100

            if real_demand == 0:
                error = 0


                # print("\n--"+name+"--")
            model.fit(X_train, Y_train.values.ravel())
            predictions = model.predict(X_validation)
                # print("\nAccuracy Score: %f" % accuracy_score(Y_validation, predictions))
                # print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
                # print(classification_report(Y_validation, predictions))

            performed_prob = model.predict_proba(X_validation)[:, 1]
            real_demand = np.sum(Y_validation['CLASS'] == 1)
            expected_demand = np.sum(performed_prob)
            predicted_demand = np.sum(predictions == 1)

            G.append([predicted_demand, expected_demand, share_demand])

            O.append(real_demand)

        except Exception:
            G.append([0,0,0])
            O.append(0)

        d = d + relativedelta(days= 1)
    G = sm.add_constant(G)
    modelOLS = sm.OLS(O,G)
    results = modelOLS.fit()
    return results.params