import numpy as np
from utils import *

class BaysianClassifier():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use Maximum Likelihood Estimation
        pass
        return self
    
    def predict(self, X):
    	pass
        return Y_pred

    def get_discriminant_function(self):
        pass


def test_Baysian():
    
    def generate_data(n=100, n_dims=2):
    	pass
        return np.array(X), np.array(y)

    X, y = generate_data()
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)

    model = BaysianClassifier()
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
    y_pred = model.predict(X_test)
    print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))

if __name__ == '__main__':
    test_Baysian()
