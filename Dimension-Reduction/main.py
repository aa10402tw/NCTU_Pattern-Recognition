import numpy as np
import matplotlib.pyplot as plt

from utils import *
from datasets import *
from classifiers import *
from metrics import *

def train_and_test_model(model, X, y, dataset_name='', verbose=True):
    if verbose:
        print('='*40 + "\n(Dataset: {}) \n(Classifier: {})\n".format(dataset_name, str(model)) + '-'*40)
    X, y = read_dataset_BreastCancer()
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_acc = accuracy(y_train, y_pred)
    if verbose:
        print('Train Accuracy: %.2f %%' % (train_acc * 100))

    y_pred = model.predict(X_test)
    test_acc = accuracy(y_test, y_pred)
    if verbose:
        print('Test Accuracy:  %.2f %%' % (test_acc * 100))
        print('='*40)
    return train_acc, test_acc

datasets = ['BreastCancer', 'Banknote']
models = [BayesianClassifier(), NaiveBayesClassifier()]

n_figures = len(datasets)
plt.figure(figsize=(5*n_figures, 5))

for i, dataset in enumerate(datasets):
    plt.subplot(1, n_figures, i+1)
    X, y = read_dataset(dataset=dataset)
    (X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.5)
    for model in models:
        model = model.fit(X_train, y_train)
        y_pred_prob = model.predict_prob(X_test)
        y_pred_prob = y_pred_prob[:, 1].reshape(-1)
        FPR, TPR = ROC(y_test, y_pred_prob, label=str(model))
        plt.title(dataset)
plt.show()


# # BreastCancer (binary-classes classification)
# X, y = read_dataset_BreastCancer()

# model = NaiveBayesClassifier()
# train_and_test_model(model, X, y, "BreastCancer")

# model = BaysianClassifier()
# train_and_test_model(model, X, y, "BreastCancer")

# model = PerceptronClassifier()
# train_and_test_model(model, X, y, "BreastCancer")

# # Iris (3-classes classification)
# X, y = read_dataset_Iris()
# model = NaiveBayesClassifier()
# train_and_test_model(model, X, y, "Iris")

# model = BaysianClassifier()
# train_and_test_model(model, X, y, "Iris")


# # Glass (3-classes classification)
# X, y = read_dataset_Glass()
# model = NaiveBayesClassifier()
# train_and_test_model(model, X, y, "Glass")

# model = BaysianClassifier()
# train_and_test_model(model, X, y, "Glass")

