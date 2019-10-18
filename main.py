from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.io import arff
import pandas as pd
import numpy as np


def readDataFile(path):
    data = []
    file = open(path)
    for line in file:
        temp = line.replace("\n", '').replace('B', '0').replace('M', '1').split(',')
        data.append([float(i) for i in temp])

    return np.array(data)

def scale(X):
    """The following function is used to scale the features
    @X This is the feature vector of type numpy"""
    scalar = StandardScaler().fit(X)
    return scalar.transform(X)


def removeCorr(X, cutoff):
    """The following function is used to remove the highly correlated variables
    from the dataset. It uses 'spearman' correlation to find the variables.
    @X This is the feature vector of type DataFrame
    @cutoff This is used to remove the variables that greater or equal to the value
    @return It returns numpy object """

    temp = X.copy()
    corr_matrix = temp.corr(method='spearman').abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_index = [column for column in upper_triangle.columns if any(upper_triangle[column] > cutoff)]
    temp = temp.drop(drop_index, axis=1)
    return temp.to_numpy()

def knn(X, y):
    """The following function is used to perform KN for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns KNeighborsClassifier object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'n_neighbors': (5,6,7,8,9,10,11,12),
    'weights': ('uniform', 'distance'),
    'algorithm': ('auto'),
    'p':(1,2)
    }"""

    temp_cls_ = KNeighborsClassifier()

    parameters = {
    'n_neighbors': (5,6,7,8,9,10,11,12),
    'weights': ('uniform', 'distance'),
    'p':(1,2),
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = KNeighborsClassifier(**param_tuner_.best_params_).fit(X, y)
    return cls


def adaBoost(classifier, X, y):
    """The following function is used to perform adaBoost for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns AdaBoostClassifier object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'n_estimators': (10,20,30,40,50,60,70,80,90,100,200,300)
    }"""

    temp_cls_ = AdaBoostClassifier()

    parameters = {
    'n_estimators': (10,20,30,40,50,60,70,80,90,100,200,300),
    'base_estimator': [classifier],
    'random_state': [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = AdaBoostClassifier(**param_tuner_.best_params_).fit(X, y)
    return cls


def randomForest(X, y):
    """The following function is used to perform random forest for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns RandomForestClassifier object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'max_depth': (20,30,40,50,100,150,200),
    'n_estimators': (10,20,30,40,50,60,70,80,90,100,200,300)
    }"""

    temp_cls_ = RandomForestClassifier()

    parameters = {
    'max_depth': (50,100,150,200),
    'min_samples_split': (50,40,30,20,10,2),
    'n_estimators': (100, 150, 200, 350),
    'random_state' : [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = RandomForestClassifier(**param_tuner_.best_params_).fit(X, y)
    return cls


def decisionTree(X, y):
    """The following function is used to perform decision tree for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns DecisionTreeClassifier object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'max_depth': (50,100,150,200),
    'min_samples_split': (50,40,30,20,10,2,1),
    'random_state' : [0]
    }"""

    temp_cls_ = DecisionTreeClassifier()

    parameters = {
    'max_depth': (50,100,150,200),
    'min_samples_split': (50,40,30,20,10,2),
    'random_state' : [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = DecisionTreeClassifier(**param_tuner_.best_params_).fit(X, y)
    return cls




def logisticRegression(X, y):
    """The following function is used to perform logistic regression for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns LogisticRegression object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'penalty': ('l1', 'l2', 'elasticnet'),
    'dual': (True, False),
    'C': [1,4,10],
    'solver': ('newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’),
    'random_state' : [0]
    }"""

    temp_cls_ = LogisticRegression()

    parameters = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'random_state': [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = LogisticRegression(**param_tuner_.best_params_).fit(X, y)
    return cls



def svmClassifier(X, y):
    """The following function is used to perform SVM for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns SVC object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'degree': [2,3,4,5,6]
    'C': [1,4,10],
    'gamma': ['auto', 'scale']
    }"""

    temp_cls_ = SVC()

    parameters = {
    'kernel': ('linear', 'rbf'),
    'C': [1,2,3,4],
    'degree': [2,3,4],
    'gamma': ['auto', 'scale'],
    'random_state' : [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters)
    param_tuner_.fit(X, y)
    cls = SVC(**param_tuner_.best_params_).fit(X, y)
    return cls



arff_dataset, meta = arff.loadarff('/Users/sandeepchowdaryannabathuni/desktop/project/Diabetic_Retinopathy_Debrecen/messidor_features.arff')
dataset = np.array(arff_dataset.tolist(), dtype=np.int8)
X_original = dataset[:, 0:18]
y = dataset[:, 19]

#X_original = scale(X_original)

#Divide the dataset into two parts.
#a) Training data (75%)
#b) Testing data (25%)
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

#support vector machine
#Feature normalization degraded the performance
#print('support vector machine (accuracy): ', accuracy_score(svmClassifier(X_train, y_train).predict(X_test), y_test))

#logistic Regression
#It gave better accuracy after feature normalization by 1%
#print('logistic regression (accuracy): ', accuracy_score(logisticRegression(X_train, y_train).predict(X_test), y_test))

#decision tree
#Feature normalization has no effect
#print('decision tree (accuracy): ', accuracy_score(decisionTree(X_train, y_train).predict(X_test), y_test))

#random forest
#Feature normalization has no effect
#print('random forest (accuracy): ', accuracy_score(randomForest(X_train, y_train).predict(X_test), y_test))

#AdaBoostClassifier
#Feature normalization has no effect
#print('Adaboost (accuracy): ', accuracy_score(adaBoost(DecisionTreeClassifier(), X_train, y_train).predict(X_test), y_test))

#K-neighbors
#Feature normalization degraded the performance
#print('K-neighbors (accuracy): ', accuracy_score(knn(X_train, y_train).predict(X_test), y_test))
