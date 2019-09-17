# Chi^2 Test for correlation and feature selection
import operator

from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, rfe, RFE
from sklearn.linear_model import LogisticRegression


class FeatureSelectionTests:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.p_values = []
        self.best_features = []
        self.transformed = None

    def run_chi2(self):
        print("\nChi^2 feature selection (select top 25 features)")
        test = SelectKBest(score_func=chi2, k=25)

        self.transformed = DataFrame(test.fit_transform(self.x, self.y))

        idxs_selected = test.get_support(indices=True)

        print(idxs_selected)

        for i, value in enumerate(self.x.columns.values):
            if i in idxs_selected:
                self.best_features.append({"index": i, "feature": value, "p-val": test.pvalues_[i], "score": test.scores_[i]})
                # print(i, ":", value, "| p-val=", test.pvalues_[i], "| score=", test.scores_[i])

        self.p_values = test.pvalues_

        self.best_features.sort(key=operator.itemgetter("score"), reverse=True)

        for f in self.best_features:
            print(f)

    def get_pvals(self):
        return self.p_values

    def get_best_features(self):
        return self.best_features

    def get_transformed(self):
        return self.transformed

    def run_rfe(self):
        print("\nRecursive Feature Elimination (select top 25 features)")

        model = LogisticRegression()

        rfe = RFE(model, 25)

        fit = rfe.fit(self.x, self.y)

        print("Num Features: %d") % fit.n_features_
        print("Selected Features: %s") % fit.support_
        print("Feature Ranking: %s") % fit.ranking_

    # def run_pca(self):
    #     print("\nPrincipal Component Analysis (select top 25 features)")
    #     pca = PCA(n_components=25)
    #
    #     newDf = DataFrame(pca.fit_transform(self.x, self.y))
    #
    #     print(newDf.columns.values)
