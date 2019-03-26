import numpy as np
from utils import *

class NaiveBayesClassifier():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use Maximum Likelihood Estimation
        N, D = X.shape
        num_classes = len(set(y))
        self.priors = [ np.count_nonzero(y==c)/N for c in range(num_classes)]
        X_c = [X[y==c] for c in range(num_classes)]
        self.means = np.zeros((num_classes, D))
        self.stds = np.zeros((num_classes, D))
        eplison = 1e-12
        for c in range(num_classes):
            self.means[c, :] = np.mean(X_c[c], axis=0)
            self.stds[c, :] = np.std(X_c[c], axis=0) + eplison # avoid division by zero
        return self
    
    def predict(self, X):
        num_classes, num_featrues = self.means.shape
        log_priors = np.log(self.priors)
        Y_pred = []
        for x in X:
            log_likelihoods = np.zeros(num_classes)
            for c in range(num_classes):
                for d in range(num_featrues):
                    mean = self.means[c, d]
                    std = self.stds[c, d] 
                    log_likelihoods[c] += log_gaussian_pdf(x[d], mean, std)

            log_posteriors = log_priors + log_likelihoods
            y_pred = np.argmax(log_posteriors)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def predict_prob(self, X):
        num_classes, num_featrues = self.means.shape
        log_priors = np.log(self.priors)
        Y_pred_prob = []
        for x in X:
            log_likelihoods = np.zeros(num_classes)
            for c in range(num_classes):
                for d in range(num_featrues):
                    mean = self.means[c, d]
                    std = self.stds[c, d] 
                    log_likelihoods[c] += log_gaussian_pdf(x[d], mean, std)
            log_posteriors = log_priors + log_likelihoods
            y_pred_prob = np.exp(log_posteriors)
            Y_pred_prob.append(y_pred_prob/sum(y_pred_prob))
        return np.array(Y_pred_prob)

    def get_discriminant_function(self):
        pass

    def __str__(self):
        return "NaiveBayesClassifier"


def test_NaiveBayes():
    
    def generate_data(n=100, n_dims=2):
        x0 = np.random.normal(loc=0.0, scale=1, size=(n//2, n_dims))
        y0 = [ 0 for i in range(n//2) ]
        x1 = np.random.normal(loc=5.0, scale=1, size=(n//2, n_dims))
        y1 = [ 1 for i in range(n//2) ]
        X = list(x0) + list(x1)
        y = y0 + y1
        return np.array(X), np.array(y)

    X, y = generate_data()
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)

    model = NaiveBayesClassifier()
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))

    y_pred = model.predict(X_test)
    print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))

if __name__ == '__main__':
    test_NaiveBayes()
