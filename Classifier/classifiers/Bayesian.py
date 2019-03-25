import numpy as np
from utils import *

class BaysianClassifier():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use maximimun likeliehood estimation 
        N, D = X.shape
        num_classes = len(set(y))
        X_c = [X[y==c] for c in range(num_classes)]
        self.priors = [ np.count_nonzero(y==c)/N for c in range(num_classes)]
        self.means = np.zeros((num_classes, D))
        self.covs = np.zeros((num_classes, D, D))
        
        # u_head = (1/m) * sigma (x_i)
        for c in range(num_classes):
            self.means[c] = np.mean(X_c[c], axis=0)
        # cov_head = (1/m) * (x_i-u_head)(x_i-u_head) ^ T
        for x, c in zip(X, y):
            u = self.means[c].reshape(-1,1)
            x = x.reshape(-1,1)
            self.covs[c] += np.matmul((x-u), (x-u).T)
        for c in range(num_classes):
            self.covs[c] /= len(X_c[c])
        return self
    
    def predict(self, X):
        num_classes, num_featrues = self.means.shape
        log_priors = np.log(self.priors)
        Y_pred = []
        for x in X:
            log_likelihoods = np.zeros(num_classes)
            for c in range(num_classes):
                mean, cov = self.means[c], self.covs[c]
                log_likelihoods[c] += log_multivariate_gaussian_pdf(x, mean, cov)
            log_posteriors = log_priors + log_likelihoods
            y_pred = np.argmax(log_posteriors)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def get_discriminant_function(self):
        pass

    def __str__(self):
        return "BaysianClassifier"


def test_Baysian():
    
    def generate_data(n=100, n_dims=2):
        pass

    X, y = generate_data()
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)

    model = BaysianClassifier()
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
    y_pred = model.predict(X_test)
    print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))

if __name__ == '__main__':
    #test_Baysian()
    x = 1
