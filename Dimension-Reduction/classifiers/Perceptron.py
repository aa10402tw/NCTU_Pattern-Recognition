import numpy as np
from utils import *
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class PerceptronClassifier():
    def __init__(self, max_epochs=100):
        self.max_epochs = max_epochs
        self.weights_history = []
    
    def add_bias_term(self, X):
        N, D = X.shape
        bias_term = np.ones((N,1))
        D = D + 1
        X = np.hstack((X, bias_term))
        return X
    
    def fit(self, inputs, labels, lr=0.1, gamma=0.95):
        X = self.add_bias_term(inputs)
        N, D = X.shape
        Y = labels.copy()
        Y[Y==0] = -1
        
        best_acc = 0.0
        best_w = np.random.randn(D,1) / 10
        self.w = np.random.randn(D,1) / 10
    
        lr = lr
        for epochs in range(self.max_epochs):
            lr *= gamma
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                if y * np.dot(self.w.T, x) <= 0:
                    self.w += lr * (y*x)
                    self.weights_history.append(self.w.copy())
            # Pocket, keep best
            y_pred = self.predict(inputs)
            acc = accuracy(labels, y_pred)
            if acc == 1:
                break
            elif acc > best_acc:
                best_acc = acc
                best_w = self.w.copy()
        self.w = best_w
        return self
    
    def predict(self, X):
        X = self.add_bias_term(X)
        y_pred = []
        for x in X:
            x = x.reshape(-1, 1)
            if np.dot(self.w.T, x) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)
    
    def predict_prob(self, X):
        X = self.add_bias_term(X)
        outs = []
        y_pred = np.zeros((X.shape[0], 2))
        for x in X:
            x = x.reshape(-1, 1)
            out = np.dot(self.w.T, x)
            outs.append(out)
        out = np.array(outs).reshape(-1)
        out /= out.max()
        y_pred[:, 1] = sigmoid(out)
        y_pred[:, 0] = 1 - y_pred[:, 1]
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
