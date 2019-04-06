import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.externals import joblib
import hdbscan
from sklearn.cluster import DBSCAN

model_name = './models/hdbscan.joblib'


def train_hdbscan(data):
    cluster = hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True)
    cluster.fit(data)
    save_model(cluster)


def test_hdbscan(data):
    cluster_loaded = load_model()
    labels, strengths = hdbscan.approximate_predict(cluster_loaded, data)
    print(labels, strengths)

    # db = DBSCAN(eps=0.25, min_samples=30)
    # db.fit(data)
    # labels = db.labels_
    #get db count
    cluster_groups = {}
    for i in labels:
        if cluster_groups.get(i):
            cluster_groups[i] = cluster_groups[i] + 1
        else:
            cluster_groups[i] = 1
    print("cluster_groups", cluster_groups)

    # Number of clusters in labels, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, cluster_groups


def save_model(model):
    joblib.dump(model, model_name)


def load_model():
    clf = joblib.load(model_name)
    return clf


# Generate sample data
def dbscan(data, labels_true, eps, min_samples):
    print(data)
    dbscan = DBSCAN(eps, min_samples)
    # dbscan = hdbscan.HDBSCAN(min_cluster_size=30)
    dbscan.fit(data)
    print("dbscan fitted")
    labels = dbscan.labels_
    print(labels)

    #get db count
    cluster_groups = {}
    for i in labels:
        if cluster_groups.get(i):
            cluster_groups[i] = cluster_groups[i] + 1
        else:
            cluster_groups[i] = 1
    print("cluster_groups", cluster_groups)

    # Number of clusters in labels, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, cluster_groups