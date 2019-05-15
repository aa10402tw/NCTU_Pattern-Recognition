import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)

class LDA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = 0
        self.std = 1
    
    def transform(self, X):
        X_high = np.copy(X)
        X_high = (X_high - self.mean) / self.std 
        # Project from high to low
        X_low = np.matmul(X_high, self.W)
        return np.real(X_low)
        
    def fit(self, X, Y):
        N, dim = X.shape
        X_high = np.copy(X)
        self.mean = X_high.mean()
        self.std = X_high.std()
        X_high = (X_high - self.mean) / self.std 
        
        # Compute mean for each class (mj, nj)
        mean_vectors = []
        for c in set(Y):
            mean_vectors.append( np.mean(X_high[Y==c], axis=0) )
        self.mean_vectors = mean_vectors
        
        # Compute within-class scatter
        SW = np.zeros( (dim,dim) )
        for c, mv in zip(set(Y), mean_vectors):
            within_class_scattter = np.zeros((dim, dim))
            for xi in X_high[Y==c]:
                xi = xi.reshape(-1, 1) # make vec to mat
                mj = mv.reshape(-1, 1) # make vec to mat
                within_class_scattter += np.matmul(xi-mj, (xi-mj).T)
            SW += within_class_scattter
    
        # Compute between-class scatter
        SB = np.zeros( (dim,dim) )
        m = np.mean(X_high, axis=0).reshape(-1, 1)
        for c, mv in zip(set(Y), mean_vectors):
            nj = X_high[Y==c].shape[0]
            mj = mv.reshape(-1, 1) # make vec to mat
            SB += nj * np.matmul((mj-m), (mj-m).T)
            
        # Compute W using first k eigenvetor of inv(SW)*SB
        mat = np.dot(np.linalg.pinv(SW), SB)
        eigenValues, eigenVectors = np.linalg.eig(mat)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        W = np.real(eigenVectors[:, 0:self.n_components])
        W /= np.linalg.norm(W, axis=0)
        self.W = W
        return self


# lda = LDA(n_components=2)
# X_low_lda = lda.fit(X_train, Y_train).transform(X_train)