#import/export model
import pandas as pd

from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter
model_name = './models/outlier.joblib'


def train_outlier_model(X_train, y_train, iteration=0):
    # printCsv(X_train, y_train)

    print("y_train:", len(y_train))

    # clf = LocalOutlierFactor(novelty=True)
    clf = IsolationForest()
    # clf.fit(X_train, y_train)
    # clf = BalancedBaggingClassifier(
    #     n_estimators=5,
    #     base_estimator=outlier.SVC(probability=True),
    #     random_state=0,
    #     n_jobs=-1)
    print(clf)
    clf.fit(X_train, y_train)

    save_model(clf, iteration)


def save_model(model, iteration=0):
    joblib.dump(model, "{}_{}".format(model_name, iteration))


def load_outlier_model(iteration=0):
    clf = joblib.load("{}_{}".format(model_name, iteration))
    return clf