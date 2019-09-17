import time
import operator

import pandas
from sklearn import preprocessing
from sklearn.feature_selection import *

from util_functions import divideIO, scaleData

# score_fns = (f_classif, chi2, f_regression, mutual_info_classif, mutual_info_regression)
score_fns = (f_classif, chi2)

# change path to the corresponding data csv file
testDataFilePath = 'updated_data.csv'

# data normalizer
normalScaler = preprocessing.Normalizer()

# data scaling
print("READING DATA")
data = pandas.read_csv(testDataFilePath)
print("SPLITTING DATA")
(x, y) = divideIO(data)
# print("NORMALIZING DATA")
# scaledData_x = scaleData(normalScaler, x, False)

# SelectKBest test using every score function
print("\nRunning SelectKBest Tests\n")
for scoreFn in score_fns:
    print(scoreFn.__name__)

    startTime = time.time()

    best_features_kbest = []

    kBest = SelectKBest(score_func=scoreFn, k=25)
    kBest.fit(x, y)

    idxs_selected = kBest.get_support(indices=True)
    print(idxs_selected)

    for j, value in enumerate(x.columns.values):
        if j in idxs_selected:
            best_features_kbest.append(
                {"index": j, "feature": value, "p-val": kBest.pvalues_[j], "score": kBest.scores_[j]})

    best_features_kbest.sort(key=operator.itemgetter("score"), reverse=True)

    values = list()

    for f in best_features_kbest:
        print(str(f["feature"]))
        values.append(f["feature"])

    print(values)

    endTime = time.time()

    print("Test runtime:", endTime-startTime)

    print()

# SelectPercentile test using every score function
print("\nRunning SelectPercentile Tests\n")
for scoreFn in score_fns:
    print(scoreFn.__name__)

    startTime = time.time()

    best_features_percentiles = []

    percentiles = SelectPercentile(score_func=scoreFn, percentile=25)
    percentiles.fit(x, y)

    idxs_selected = percentiles.get_support(indices=True)
    print(idxs_selected)

    for j, value in enumerate(x.columns.values):
        if j in idxs_selected:
            best_features_percentiles.append(
                {"index": j, "feature": value, "p-val": percentiles.pvalues_[j], "score": percentiles.scores_[j]})

    best_features_percentiles.sort(key=operator.itemgetter("score"), reverse=True)

    values = list()

    for f in best_features_percentiles:
        print(str(f["feature"]))
        values.append(f["feature"])

    print(values)

    endTime = time.time()

    print("Test runtime:", endTime - startTime)

    print()

# SelectFpr test using every score function
print("\nRunning SelectFpr Tests\n")
for scoreFn in score_fns:
    print(scoreFn.__name__)

    startTime = time.time()

    best_features_fpr = []

    fpr = SelectFpr(score_func=scoreFn)
    fpr.fit(x, y)

    idxs_selected = fpr.get_support(indices=True)
    print(idxs_selected)

    values = list()

    for i, pval in enumerate(fpr.pvalues_):
        if pval > 0.05:
            print("discarted feature:", x.columns.values[i], "with p-value:", pval)
        else:
            values.append(x.columns.values[i])

    for j, value in enumerate(x.columns.values):
        if j in idxs_selected:
            best_features_fpr.append\
                ({"index": j, "feature": value, "p-val": fpr.pvalues_[j], "score": fpr.scores_[j]})

    best_features_fpr.sort(key=operator.itemgetter("score"), reverse=True)

    print()
    print(values)

    for f in values:
        print(f)

    print()

    endTime = time.time()

    print("Test runtime:", endTime - startTime)

    print()

# SelectFdr test using every score function
print("\nRunning SelectFdr Tests\n")
for scoreFn in score_fns:
    print(scoreFn.__name__)

    startTime = time.time()

    best_features_fdr = []

    fdr = SelectFdr(score_func=scoreFn)
    fdr.fit(x, y)

    idxs_selected = fdr.get_support(indices=True)
    print(idxs_selected)

    values = list()

    for i, pval in enumerate(fdr.pvalues_):
        if pval > 0.05:
            print("discarted feature:", x.columns.values[i], "with p-value:", pval)
        else:
            values.append(x.columns.values[i])

    for j, value in enumerate(x.columns.values):
        if j in idxs_selected:
            best_features_fdr.append(
                {"index": j, "feature": value, "p-val": fdr.pvalues_[j], "score": fdr.scores_[j]})

    best_features_fdr.sort(key=operator.itemgetter("score"), reverse=True)

    print()
    print(values)

    for f in values:
        print(f)

    print()

    endTime = time.time()

    print("Test runtime:", endTime - startTime)

    print()

# SelectFwe test using every score function
print("\nRunning SelectFwe Tests\n")
for scoreFn in score_fns:
    print(scoreFn.__name__)

    startTime = time.time()

    best_features_fwe = []

    fwe = SelectFwe(score_func=scoreFn)
    fwe.fit(x, y)

    idxs_selected = fwe.get_support(indices=True)
    print(idxs_selected)

    values = list()

    for i, pval in enumerate(fwe.pvalues_):
        if pval > 0.05:
            print("discarted feature:", x.columns.values[i], "with p-value:", pval)
        else:
            values.append(x.columns.values[i])

    for j, value in enumerate(x.columns.values):
        if j in idxs_selected:
            best_features_fwe.append(
                {"index": j, "feature": value, "p-val": fwe.pvalues_[j], "score": fwe.scores_[j]})

    best_features_fwe.sort(key=operator.itemgetter("score"), reverse=True)

    print()
    print(values)

    for f in values:
        print(f)

    print()

    endTime = time.time()

    print("Test runtime:", endTime - startTime)

    print()
