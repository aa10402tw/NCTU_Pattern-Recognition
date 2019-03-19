import numpy as np
import matplotlib.pyplot as plt

from utils import *
from datasets import *
import classifiers
from classifiers.NaiveBayes import *

# BreastCancer (binary-classes classification)
print('='*20 + "\n(Dataset: {}) \n(Classifier: {})\n".format("BreastCancer", "NaiveBayes") + '-'*20)
X, y = read_dataset_BreastCancer()
(X_train, y_train), (X_test, y_test) = split_data(X, y, split_ratio=0.9)
model = NaiveBayesClassifier()
model = model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
y_pred = model.predict(X_test)
print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))
print('='*20)


# Iris (3-classes classification)
print('\n')
print('='*20 + "\n(Dataset: {}) \n(Classifier: {})\n".format("Iris", "NaiveBayes") + '-'*20)
X, y = read_dataset_IRIS()
(X_train, y_train), (X_test, y_test) = split_data(X, y)
model = NaiveBayesClassifier()
model = model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
y_pred = model.predict(X_test)
print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))
print('='*20)


# Glass (3-classes classification)
print('\n')
print('='*20 + "\n(Dataset: {}) \n(Classifier: {})\n".format("Glass", "NaiveBayes") + '-'*20)
X, y = read_dataset_Glass()
(X_train, y_train), (X_test, y_test) = split_data(X, y)
model = NaiveBayesClassifier()
model = model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print('Train Accuracy: %.2f %%' % (accuracy(y_train, y_pred) * 100))
y_pred = model.predict(X_test)
print('Test Accuracy: %.2f %%' % (accuracy(y_test, y_pred) * 100))
print('='*20)