#import/export model
import pandas as pd

from sklearn.externals import joblib
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter
# model_name = './models/svm_text.joblib'

# model_name = './models/svm_rdf.joblib'
model_name = './models/svm.joblib'


def train_svm_model(X_train, y_train):
    # printCsv(X_train, y_train)

    print("y_train:", len(y_train))

    # clf = LocalOutlierFactor(novelty=True)
    # clf = IsolationForest()
    clf = svm.SVC(probability=True)
    # clf.fit(X_train, y_train)
    # clf = BalancedBaggingClassifier(
    #     n_estimators=5,
    #     base_estimator=svm.SVC(probability=True),
    #     random_state=0,
    #     n_jobs=-1)
    print(clf)
    clf.fit(X_train, y_train)

    save_model(clf)


def test_svm_model(X_test, y_test):
    print("test_svm_model")
    print(y_test)
    print("X_test:", X_test.shape)
    clf = load_svm_model()

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    # Model Precision: What proportion of positive identifications was actually correct?
    # Model Recall: What proportion of actual positives was identified correctly?

    return y_pred


def save_model(model):
    joblib.dump(model, model_name)


def load_svm_model():
    clf = joblib.load(model_name)
    return clf


def printCsv(df, labels):
    temp = df.copy()
    temp['label'] = labels
    writer = pd.ExcelWriter(
        './sample/clusters/svm.xlsx',
        engine='xlsxwriter',
        options={'strings_to_urls': False})
    temp.to_excel(writer)
    writer.save()
