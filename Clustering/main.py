import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

from utils import *
from datasets import *
from classifiers import *
from metrics import *

from agglomerative_clustering import AgglomerativeClustering
from dbscan import DBSCAN

X, y = read_dataset(dataset='Iris')

print("--- AgglomerativeClustering ---")
model = AgglomerativeClustering(n_clusters=3, verbose=False, linkage='complete', distance_metric='l1')
cluster_pred = model.fit_predict(X)
print("adjusted_rand_score", metrics.adjusted_rand_score(y, cluster_pred))
print(" normalized_mutual_info_score", normalized_mutual_info_score(y, cluster_pred))

print("--- DBSCAN ---")
cluster_pred = DBSCAN(eps=1, MinPts=5).fit_predict(X)
print("adjusted_rand_score", metrics.adjusted_rand_score(y, cluster_pred))
print(" normalized_mutual_info_score", normalized_mutual_info_score(y, cluster_pred))
