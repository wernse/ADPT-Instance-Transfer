#import/export model
import pandas as pd

from sklearn.externals import joblib
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# model_name = './models/svm_text.joblib'

# model_name = './models/svm_rdf.joblib'
model_name = './models/dt.joblib'


def train_tree_model(X_train, y_train):
    classifier = BalancedBaggingClassifier(
        n_estimators=5,
        base_estimator=DecisionTreeClassifier(),
        random_state=0,
        n_jobs=-1)
    classifier.fit(X_train, y_train)
    save_model(classifier)


def save_model(model):
    joblib.dump(model, model_name)


def load_tree_model():
    clf = joblib.load(model_name)
    return clf