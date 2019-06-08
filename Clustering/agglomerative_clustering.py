import numpy as np

from utils import *

class AgglomerativeClustering:
    def __init__(self, n_clusters = 2, linkage='single', distance_metric='l2', verbose=True):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.verbose = verbose
        
    def fit_predict(self, X):
        # 1. Compute pairwise distance
        pair_distance = np.zeros((len(X), len(X)))
        if self.verbose:
            print("--Compute pairwise distance")
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                pair_distance[i, j] = compute_distance(X[i], X[j], metric=self.distance_metric)
                pair_distance[j, i] = pair_distance[i, j]
        if self.verbose:
            print("--Agglomerative Clustering")
        cluster_assignment = np.array([i for i in range(len(X))])
        exist_cluster = np.array(list(set(cluster_assignment)))
        n_cluster = len(exist_cluster)
        while n_cluster > self.n_clusters:
            # 2. Find pair between cluster with minimum distance  
            merge_idx = (-1, -1)
            min_cluster_distance = 1e+5
            for i in range(len(exist_cluster)):
                for j in range(i+1, len(exist_cluster)):
                    cluster_i = exist_cluster[i]
                    cluster_j = exist_cluster[j]
                    cluster_i_member = X[cluster_assignment[cluster_assignment==cluster_i]]
                    cluster_j_member = X[cluster_assignment[cluster_assignment==cluster_j]]
                    
                    cluster_distance = 1e-5 if self.linkage == 'single' else 0
                    cnt = 0
                    
                    for member_i in cluster_i_member:
                        for member_j in cluster_j_member:
                            distance = compute_distance(member_i, member_j, metric=self.distance_metric)
                            if self.linkage == 'single':
                                cluster_distance = min(cluster_distance, distance)
                            if self.linkage == 'complete':
                                cluster_distance = max(cluster_distance, distance)
                            if self.linkage == 'average':
                                cluster_distance += distance
                                cnt += 1
                    if self.linkage == 'average':
                        cluster_distance /= cnt
                    if cluster_distance < min_cluster_distance:
                        min_cluster_distance = cluster_distance
                        merge_idx = (cluster_i, cluster_j)
            
            # 3. Merge two cluster 
            i, j = merge_idx
            cluster_assignment[cluster_assignment==j] = i
            exist_cluster = np.array(list(set(cluster_assignment)))
            n_cluster = len(exist_cluster)
            if self.verbose:
                print("Merge :", merge_idx, "\tNum of Cluster:", n_cluster)
                print("")
        # 4. Reassign Cluster index
        for i in range(self.n_clusters):
            cluster_assignment[cluster_assignment == exist_cluster[i]] = i
        if self.verbose:
            print(cluster_assignment)
        return cluster_assignment

def plot_data(X, y):
    for c in range(len(set(y))):
        plt.scatter(X[y==c][:,0], X[y==c][:,1])
#     plt.show()

def test():
    import matplotlib.pyplot as plt
    X, y = generate_data(n_data=50)
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    model = AgglomerativeClustering(n_clusters=2, verbose=False, linkage='single', distance_metric='l1')
    cluster_pred = model.fit_predict(X)
    plot_data(X, cluster_pred)
    plt.title("Single Linkage")

    plt.subplot(122)
    model = AgglomerativeClustering(n_clusters=2, verbose=False, linkage='complete', distance_metric='l1')
    cluster_pred = model.fit_predict(X)
    plot_data(X, cluster_pred)
    plt.title("Complete Linkage")
    plt.show()


if __name__ == '__main__':
    test()