import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# #############################################################################
# Generate sample data
iris_data, labels_true = load_iris(True)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(iris_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print("true labels", labels_true)
print("pred labels", labels)

labels_df = pd.DataFrame(labels)

groups = labels_df.groupby([0]).groups
print(len(groups[-1]))
# print(labels_df)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(
#     labels_true, labels))
# print("Adjusted Mutual Information: %0.3f" %
#       metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
#     iris_data, labels))

# # #############################################################################
# # Plot result
# import matplotlib.pyplot as plt

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [
#     plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
# ]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = iris_data[class_member_mask & core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         'o',
#         markerfacecolor=tuple(col),
#         markeredgecolor='k',
#         markersize=14)

#     xy = iris_data[class_member_mask & ~core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         'o',
#         markerfacecolor=tuple(col),
#         markeredgecolor='k',
#         markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# # plt.show()