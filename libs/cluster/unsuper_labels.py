import math
import hdbscan
import pandas as pd
from libs.drift.kl_divergence import calc_pair_divergence
from libs.cluster.dbscan import dbscan, test_hdbscan


def generate_clusters(df):
    """ Runs an algorithm to select the min cluster size to only select 3 clusters """

    df_size = df.shape[0]
    print(df_size)
    n_clusters = 0
    percent_min_pts = 0.105
    min_clusters = 3
    while (n_clusters != min_clusters):
        print("percent_min_pts", percent_min_pts)
        min_cluster_pts = math.floor(df_size * percent_min_pts)
        print("min_cluster_pts", min_cluster_pts)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_pts)
        print(df.head())
        clusterer.fit(df)
        cluster_groups = {}
        labels = clusterer.labels_
        for i in labels:
            if cluster_groups.get(i):
                cluster_groups[i] = cluster_groups[i] + 1
            else:
                cluster_groups[i] = 1
        print("cluster_groups", cluster_groups)
        n_clusters = len(set(labels))
        print("n_clusters", n_clusters)
        multiplier = abs(n_clusters - min_clusters) * 0.001
        print("multiplier", multiplier)
        if n_clusters > min_clusters:
            percent_min_pts += multiplier
        else:
            percent_min_pts -= multiplier
        print("percent_min_pts", percent_min_pts)
    return labels


def generate_clusters_n(df, tweet_deleted):
    """ Runs an algorithm to select the min cluster size to only select 3 clusters """
    # cluster_labels, n_clusters = dbscan(normalised_df, true_labels, 0.25, 30)
    # print("normalised_df.head()", normalised_df.head())
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusterer.fit(df)
    labels = clusterer.labels_
    cluster_groups = {}
    for i in labels:
        if cluster_groups.get(i):
            cluster_groups[i] = cluster_groups[i] + 1
        else:
            cluster_groups[i] = 1
    print("cluster_groups", cluster_groups)
    df["cluster"] = labels
    df["tweet_deleted"] = tweet_deleted
    cluster_results = list()
    for cluster_no in cluster_groups.keys():
        print("++++++++++")
        print("cluster_no", cluster_no)
        cluster_result = list()
        cluster_result.append(cluster_no)

        cluster = df.mask('cluster', cluster_no)
        print(cluster_no, " :")
        tweet_deleted = cluster.mask('tweet_deleted', True).shape[0]
        not_tweet_deleted = cluster.mask('tweet_deleted', False).shape[0]
        print("deleted_df len:", tweet_deleted)
        print("not_deleted_df len:", not_tweet_deleted)


def generate_metrics(df_n_phishing_dist, df, labels):
    df['cluster'] = labels
    cluster_groups = {}
    for i in labels:
        if cluster_groups.get(i):
            cluster_groups[i] = cluster_groups[i] + 1
        else:
            cluster_groups[i] = 1
    print("cluster_groups", cluster_groups)

    #split the df into labelled data
    deleted = df.mask('tweet_deleted', True)
    not_deleted = df.mask('tweet_deleted', False)
    print("deleted_df len:", deleted.shape[0])
    print("not_deleted_df len:", not_deleted.shape[0])

    cluster_0 = df.mask('cluster', 0)
    cluster_1 = df.mask('cluster', 1)

    #Calculate the KL Divergence of the clusters
    divergence_0 = calc_pair_divergence(df_n_phishing_dist, cluster_0)
    divergence_1 = calc_pair_divergence(df_n_phishing_dist, cluster_1)
    print("divergence_0", divergence_0)
    print("cluster_0.mask('tweet_deleted', True).shape[0]",
          cluster_0.mask('tweet_deleted', True).shape[0])
    print("cluster_0.mask('tweet_deleted', False).shape[0]",
          cluster_0.mask('tweet_deleted', False).shape[0])
    print("divergence_1", divergence_1)
    print("cluster_1.mask('tweet_deleted', True).shape[0]",
          cluster_1.mask('tweet_deleted', True).shape[0])
    print("cluster_1.mask('tweet_deleted', False).shape[0]",
          cluster_1.mask('tweet_deleted', False).shape[0])
    non_phishing_cluster, non_phishing_divergence = (
        0, divergence_0) if divergence_0 < divergence_1 else (1, divergence_1)
    phishing_cluster, phishing_divergence = (
        1, divergence_1) if divergence_0 < divergence_1 else (0, divergence_0)

    non_phishing_cluster = 0 if cluster_0.shape[0] < cluster_1.shape[0] else 1
    phishing_cluster = 1 if cluster_0.shape[0] < cluster_1.shape[0] else 0

    print("non_phishing_cluster", non_phishing_cluster)
    print("phishing_cluster", phishing_cluster)
    cluster_lists = list()
    for cluster_no in cluster_groups.keys():
        print("cluster_no", cluster_no)
        print(cluster_groups[cluster_no])
        cluster = df.mask('cluster', cluster_no)
        tweet_deleted = cluster.mask('tweet_deleted', True).shape[0]
        not_tweet_deleted = cluster.mask('tweet_deleted', False).shape[0]
        print("deleted_df len:", tweet_deleted)
        print("not_deleted_df len:", not_tweet_deleted)
        label = 'non-phishing' if non_phishing_cluster == cluster_no else "phishing"
        if cluster_no == -1:
            label = ''
        print("label", label)
        cluster_obj = {
            'cluster': cluster_no,
            'total': cluster.shape[0],
            'tweet_deleted': tweet_deleted,
            'not_tweet_deleted': not_tweet_deleted,
            'tweet_deleted_percent': tweet_deleted / cluster.shape[0],
            'not_tweet_deleted_percent': not_tweet_deleted / cluster.shape[0],
            'label': label,
            'non_phishing_divergence': non_phishing_divergence,
            'phishing_divergence': phishing_divergence
        }
        cluster_lists.append(cluster_obj)

    return pd.DataFrame(cluster_lists), non_phishing_cluster, phishing_cluster
