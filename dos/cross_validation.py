import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape

# validation set/Development set
# Test set/holdout set

for i in range(100):
    X_train, X_dev, y_train, y_dev = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True)



