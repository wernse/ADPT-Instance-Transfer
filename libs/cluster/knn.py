# 4. train KNN on the dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from imblearn.ensemble import BalancedBaggingClassifier

model_name = './models/knn.joblib'


def train_knn_model(df_formatted, true_labels, iteration=0):
    classifier = BalancedBaggingClassifier(
        n_estimators=5,
        base_estimator=KNeighborsClassifier(n_neighbors=5),
        random_state=0,
        n_jobs=-1)
    classifier.fit(df_formatted, true_labels)
    save_model(classifier, iteration)


def test_knn_model(df_formatted):
    clf = load_knn_model()
    #Predict the response for test dataset
    y_pred = clf.predict(df_formatted)
    return y_pred


def save_model(model, iteration):
    joblib.dump(model, "{}_{}".format(model_name, iteration))


def load_knn_model(iteration=0):
    clf = joblib.load("{}_{}".format(model_name, iteration))
    return clf