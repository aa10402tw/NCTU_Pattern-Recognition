import numpy as np
from utils import *
import matplotlib.pyplot as plt
class PerceptronClassifier():
    def __init__(self, max_epochs=100):
        self.max_epochs = max_epochs
        self.weights_history = []
    
    def fit(self, X, labels):
        N, D = X.shape
        Y = labels.copy()
        Y[Y==0] = -1
        
        self.w = np.zeros((D,1))
        num_correct = 0
        isPerfect = False
        lr = 0.1
        for epochs in range(self.max_epochs):
            lr *= 0.5
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                if y * np.dot(self.w.T, x) <= 0:
                    self.w += lr * (y*x)
                    self.weights_history.append(self.w.copy())
                    num_correct = 0
                else:
                    num_correct += 1
                    if num_correct > N:
                        isPerfect = True
            if isPerfect:
                break
        return self
    
    def predict(self, X):
        y_pred = []
        for x in X:
            x = x.reshape(-1, 1)
            if np.dot(self.w.T, x) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)

    def get_w(self):
        return self.w
    
    def get_w_histroy(self):
        return self.weights_history
    
    def get_discriminant_function(self):
        pass

    def __str__(self):
        return "PerceptronClassifier"

def generatingLinearData(n=100, n_dims=2, require_w=False):
    xs = np.random.uniform(low=-1, high=1, size=(n, n_dims))
    ys = []
    w = np.random.uniform(-1, 1, size=(n_dims, 1))
    for x in xs:
        x = x.reshape(-1, 1)
        y = 1 if np.dot(w.T, x) >= 0 else 0
        ys.append(y)
    X = np.array(xs)
    y = np.array(ys)
    if require_w:
        return X, y, w
    else:
        return X, y

def plot_result(X, y, w=None):
    # 0
    for c in range(0, 2):
        xs = [p[0] for p in X[y == c]]
        ys = [p[1] for p in X[y == c]]
        plt.scatter(xs, ys, label=str(c))
    if w is not None:
        plot_decision_boundary(w)
    plt.xlim(-1,1), plt.ylim(-1,1)
    #plt.legend(loc='best')

def plot_decision_boundary(w):
    xs = np.linspace(-1, 1, 1000)
    ys = []
    w1, w2 = w
    for x in xs:
        y = -(w1 * x) / w2 
        ys.append(y)
    plt.plot(xs, ys, c='red', label='Decision Boundary')

def test_Perceptron():
    
    X, y, w = generatingLinearData(n=100, n_dims=2, require_w=True)
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)
    plt.subplot(121)
    plot_result(X, y, w)

    model =  PerceptronClassifier()
    model = model.fit(X_train, y_train)
    w = model.get_w()
    plt.subplot(122)
    plot_result(X, y, w)
    plt.show()

    y_pred = model.predict(X_train)
    print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
    y_pred = model.predict(X_test)
    print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))

if __name__ == '__main__':
    test_Perceptron()
