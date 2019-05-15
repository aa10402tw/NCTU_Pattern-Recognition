import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def transform(self, X):
        X_high = np.copy(X)
        mean_mat = np.tile(self.mean_vec, (X.shape[0],1))
        diff_mat = X_high - mean_mat
        # Project from high to low
        X_low = np.matmul(diff_mat, self.W)
        return np.real(X_low)
    
    def fit(self, X):
        X_high = np.copy(X)
        mean_vec = np.mean(X_high, 0)
        mean_mat = np.tile(mean_vec, (X.shape[0],1))
        diff_mat = X_high - mean_mat
        cov_mat = np.cov(diff_mat.T)
        self.mean_vec = mean_vec
        
        # Compute eigenpairs of cov mat
        eigenValues, eigenVectors = np.linalg.eig(cov_mat)
        idx = eigenValues.argsort()[::-1]   
        W = eigenVectors[:,idx][:, :self.n_components]
        W = W * -1 
        self.W = W
        return self