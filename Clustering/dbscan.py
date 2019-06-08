import numpy as np

from utils import *

class DBSCAN:
    def __init__(self, eps, MinPts, distance_metric='l2'):
        self.eps = eps
        self.MinPts = MinPts
        self.distance_metric = distance_metric
        
    def fit_predict(self, X):
        X = X.copy()
        n_data = len(X)
        cluster_assignment = [0 for i in range(n_data)]
        C = 0
        for Pi in range(n_data):
            # If have visited, skip
            if cluster_assignment[Pi] != 0:
                continue
            # If Num of Neihgbor less than MinPts, mark as noise
            NeighborIdxs = self.regionQuery(X, Pi)
            if len(NeighborIdxs) < self.MinPts:
                cluster_assignment[Pi] = -1
            else:
                C += 1
                self.growCluster(X, cluster_assignment, Pi, NeighborIdxs, C)
        return np.array(cluster_assignment)
                
    def regionQuery(self, X, Pi):
        NeighborIdxs = []
        for Pn in range(0, len(X)):
            if compute_distance(X[Pi], X[Pn], self.distance_metric) < self.eps:
                NeighborIdxs.append(Pn)
        return NeighborIdxs
    
    def growCluster(self, X, cluster_assignment, Pi, NeighborIdxs, C):
        cluster_assignment[Pi] = C
        for Pn in NeighborIdxs:
            # If neightbor is noise 
            if cluster_assignment[Pn] == -1:
                cluster_assignment[Pn] = C
                
            # If neightbor is not visted yet 
            elif cluster_assignment[Pn] == 0:
                cluster_assignment[Pn] = C
                new_NeighborIdxs = self.regionQuery(X, Pn)
                if len(new_NeighborIdxs) >= self.MinPts:
                    NeighborIdxs += new_NeighborIdxs


def plot_data(X, y):
    import matplotlib.pyplot as plt
    for c in list(set(y)):
        if c == -1:
            plt.scatter(X[y==c][:,0], X[y==c][:,1], label='noise', c='black')
        else:
            plt.scatter(X[y==c][:,0], X[y==c][:,1], label=str(c))
    plt.legend(loc='best')

def test_DBSCAN():
    import matplotlib.pyplot as plt
    X, y = generate_data()

    plt.figure(figsize=(10,10))
    plt.subplot(221)
    eps = 1
    MinPts = 20
    plt.title("eps=%.2f, MinPts=%d"%(eps,MinPts))
    cluster_pred = DBSCAN(eps=eps, MinPts=MinPts).fit_predict(X)
    plot_data(X, cluster_pred)

    plt.subplot(222)
    eps = 1
    MinPts = 10
    plt.title("eps=%.2f, MinPts=%d"%(eps,MinPts))
    cluster_pred = DBSCAN(eps=eps, MinPts=MinPts).fit_predict(X)
    plot_data(X, cluster_pred)

    plt.subplot(223)
    eps = 1
    MinPts = 5
    plt.title("eps=%.2f, MinPts=%d"%(eps,MinPts))
    cluster_pred = DBSCAN(eps=eps, MinPts=MinPts).fit_predict(X)
    plot_data(X, cluster_pred)

    plt.subplot(224)
    eps = 1
    MinPts = 1
    plt.title("eps=%.2f, MinPts=%d"%(eps,MinPts))
    cluster_pred = DBSCAN(eps=eps, MinPts=MinPts).fit_predict(X)
    plot_data(X, cluster_pred)
    plt.show()
if __name__ == '__main__':
    test_DBSCAN()