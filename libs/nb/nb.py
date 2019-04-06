import pandas as pd

from sklearn.externals import joblib
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.ensemble import EasyEnsembleClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.svm import LinearSVC
from imblearn.ensemble import BalancedBaggingClassifier

model_name = './models/nb.joblib'
model_name_2 = './models/vocab.joblib'
import time


def train_nb_model(X_train, y_train, vectorize=False, iteration=0):
    # printCsv(print_df, "train")
    print("train_nb_model", iteration)
    start_time = time.time()

    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        # min_df=5,
        norm='l2',
        encoding='latin-1',
        ngram_range=(1, 2),
        stop_words='english')
    features = tfidf.fit_transform(X_train).toarray()
    labels = y_train
    print(features.shape)

    # from sklearn.feature_selection import chi2
    # import numpy as np
    # N = 10
    # a = labels == False
    # features_chi2 = chi2(features, a)
    # indices = np.argsort(features_chi2[0])
    # feature_names = np.array(tfidf.get_feature_names())[indices]
    # unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    # bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    # printCsv(pd.DataFrame(feature_names), "tfidf")
    # print("# asd")
    # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(
    #     unigrams[0:N])))
    # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(
    #     unigrams[-N:])))
    # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(
    #     bigrams[0:N])))
    # print("  . Most correlated bigrams:\n. {}".format('\n. '.join(
    #     bigrams[-N:])))
    # print("Training")
    tfidf_train = tfidf.fit(X_train)
    save_model_2(tfidf_train, iteration)
    bow_features = tfidf_train.transform(X_train)

    text_clf = BalancedBaggingClassifier(
        n_estimators=5, base_estimator=LinearSVC(), random_state=0)
    #train
    text_clf = text_clf.fit(bow_features, y_train)
    save_model(text_clf, iteration)
    print("modle saved")

    if vectorize:
        text_clf = Pipeline([('vect',
                              TfidfVectorizer(
                                  sublinear_tf=True,
                                  norm='l2',
                                  ngram_range=(1, 2),
                                  stop_words='english')), ('clf',
                                                           LinearSVC())])
        text_clf = text_clf.fit(X_train, y_train)

        save_model(text_clf)

    # vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    # vectorizer_train = vectorizer.fit(X_train)
    # X = vectorizer_train.transform(X_train)
    # save_model_2(vectorizer_train)
    # sparse = csr_matrix(normalised_data.values)
    # bow_features = hstack((sparse, X))

    # text_clf = MultinomialNB()
    # #train
    # text_clf = text_clf.fit(bow_features, y_train)
    # save_model(text_clf)
    # elapsed_time = time.time() - start_time
    # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def test_nb_model(X_test, y_test):
    print_df = pd.DataFrame(X_test).copy()
    print_df["labels"] = y_test
    printCsv(print_df, "test")
    print("X_test:", X_test.shape)
    print("X_test not_del",
          pd.DataFrame(y_test).mask('tweet_deleted', False).shape[0])
    print("X_test del",
          pd.DataFrame(y_test).mask('tweet_deleted', True).shape[0])

    clf = load_nb_model()
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # vectorizer = load_nb_model_2()
    # X = vectorizer.transform(X_test)
    # sparse = csr_matrix(normalised_data.values)
    # bow_features = hstack((sparse, X))
    # clf = load_nb_model()

    # #Predict the response for test dataset
    # y_pred = clf.predict(bow_features)

    # Model Accuracy: how often is the classifier correct?
    # Model Precision: precision is the amount of true pos out of the (false pos + true pos)
    # Model Recall: recall is the amount of true pos out of (false pos + true pos)
    print(
        metrics.classification_report(
            y_test, y_pred, target_names=['Legitimate', 'Phising']))
    return y_pred


def save_model(model, iteration=0):
    joblib.dump(model, "{}_{}".format(model_name, iteration))


def save_model_2(model, iteration=0):
    joblib.dump(model, "{}_{}".format(model_name_2, iteration))


def load_nb_model(iteration=0):
    clf = joblib.load("{}_{}".format(model_name, iteration))
    return clf


def load_nb_model_2(iteration=0):
    clf = joblib.load("{}_{}".format(model_name_2, iteration))
    return clf


# counter vector
def test():
    X_train = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    y_train = [True, False, False, True]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_train)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    df_1 = pd.DataFrame([10, 20, 30, 40], columns=['a'])
    df_result = pd.concat([df, df_1], axis=1)
    sparse = csr_matrix(df_1.values)
    sparse_1 = csr_matrix(df_result.values)
    print(X)
    print("")
    print(sparse)
    print("")
    c = hstack((sparse, X))
    print(c.toarray())
    print("")
    print(sparse_1)

    text_clf = MultinomialNB()
    text_clf = text_clf.fit(c, y_train)
    print(text_clf)

    y_pred = text_clf.predict(c)
    print(y_pred)


def printCsv(df, name=""):
    writer = pd.ExcelWriter(
        './sample/clusters/nb_{}.xlsx'.format(name),
        engine='xlsxwriter',
        options={'strings_to_urls': False})
    df.to_excel(writer)
    writer.save()