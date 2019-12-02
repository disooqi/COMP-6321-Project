from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.stats import reciprocal
from sklearn.svm import SVC
from scipy.io import arff
import pandas as pd
import numpy as np


def preprocessFeatures(X, index_, auto=False):
    """ The following function is used to convert categorical data into numerical data

    @X This is the feature vector of type DataFrame
    @index_ This is the list of columns that need to be converted to numerical data
    @auto To auto detect the categorical columns and convert them to numerical data
    @return DataFrame
    example: Suppose a column has two types of categorical data which includes 'a' and 'b'
    ----col----
        a
        a
        b

    This will be transformed to

    ----a----b----
        1    0
        1    0
        0    1 """

    output = pd.DataFrame(index=X.index)

    if(auto is True):
        for col, col_data in X.iteritems():
            if(col_data.dtype == object):
                col_data = pd.get_dummies(col_data)
            output = output.join(col_data, how='left', lsuffix='_left', rsuffix='_right')

    else:
        for col, col_data in X.iteritems():
            if(col in index_):
                col_data = pd.get_dummies(col_data)
            output = output.join(col_data, how='left', lsuffix='_left', rsuffix='_right')

    return output


def merge(X):
    """This function is used to merge multioutput vector to single vector
    @X This is the input multioutput vector
    @return numpy array
    """

    size = X.shape[1]
    output = []

    for i in X:
        count = 1
        for j in i:
            if j==1:
                output.append(count)
                break
            count+=1

    return np.array(output)


def kFold(X, y, model, n_split_=None):
    """The following function is used to perform cross validation across different
    partitions of data and return average accuracyself.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    n_split_ Default value is 10"""

    accuracy = []
    if n_split_ is None:
        n_split_ = 10

    fold = KFold(n_splits=n_split_, random_state=0, shuffle=False)

    for train_indices, test_indices in fold.split(X):
        X_train, y_train = (X[train_indices], y[train_indices])
        X_test, y_test = (X[test_indices], y[test_indices])
        cls = model.fit(X_train, y_train)
        accuracy.append(accuracy_score(cls.predict(X_test), y_test))

    return sum(accuracy)/len(accuracy)


def readDataFile(path, delimiter,replace_=None):
    """"The following functiion is used to extract the data from the readDataFile
    @path This is the file path of type string
    @delimiter This is used to split the data
    @replace This is a dict to replace the characters in data """
    data = []
    file = open(path)
    for line in file:
        temp = line
        temp.replace('\n', '')
        if replace_ is not None:
            for key, value in replace_.items():
                temp = temp.replace(key, value)

        data.append(list(map(float,temp.split(delimiter))))

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

def knn(X, y, cv_size = 5):
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

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_


def adaBoost(classifier, X, y, cv_size = 5):
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

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_


def randomForest(X, y, cv_size = 5):
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

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_


def decisionTree(X, y, cv_size = 5):
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

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_




def logisticRegression(X, y, cv_size = 5):
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

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_



def svmClassifier(X, y, cv_size = 5):
    """The following function is used to perform SVM for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @ cv_size The k-fold size
    @return It returns SVC object and accuracy score value

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
    'C': reciprocal(1,100).rvs(5),
    'gamma': reciprocal(1,100).rvs(5),
    'random_state' : [0]
    }

    param_tuner_ = RandomizedSearchCV(temp_cls_, parameters, cv=cv_size, n_iter=16)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls, param_tuner_.best_score_


# Diabetic Retinopathy##
arff_dataset, meta = arff.loadarff('/Users/sandeepchowdaryannabathuni/desktop/project/datasets/messidor_features.arff')
dataset = np.array(arff_dataset.tolist(), dtype=np.int8)

X_original = dataset[:, 0:18]
X_scaled = scale(X_original)
y = dataset[:, 19]

#X_original = scale(X_original)

#Divide the dataset into two parts.
#a) Training data (75%)
#b) Testing data (25%)
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.25, random_state=0)


#support vector machine
support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)
support_vector_scaled, cv_score_scaled = svmClassifier(X_train_scaled, y_train_scaled)

print('#'*10, 'Diabetic Retinopathy', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

print('support vector machine (accuracy with scaling): ', accuracy_score(support_vector_scaled.predict(X_test_scaled), y_test_scaled))
print('Cross validation (with scaling):', cv_score_scaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')

################################################################################################################################################################################


# Breast Cancer wisconsin
path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/wdbc.data'
replace_ = {
    'M' : '0',
    'B' : '1'
}
data = readDataFile(path,',', replace_)
X_original = data[:,2:]
X_scaled = scale(X_original)
y = data[:,1]

#Divide the dataset into two parts.
#a) Training data (75%)
#b) Testing data (25%)
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.25, random_state=0)


#support vector machine
support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)
support_vector_scaled, cv_score_scaled = svmClassifier(X_train_scaled, y_train_scaled)

print('#'*10, 'Breast Cancer wisconsin', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

print('support vector machine (accuracy with scaling): ', accuracy_score(support_vector_scaled.predict(X_test_scaled), y_test_scaled))
print('Cross validation (with scaling):', cv_score_scaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')


################################################################################################################################################################################


#seismic bumps

path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/seismic-bumps.arff'
data, meta = arff.loadarff(path)
temp = data.tolist()
data = pd.DataFrame(temp)

utf_8_data = pd.DataFrame(index=data.index)

for col, col_data in data.iteritems():
    if(col_data.dtype == object):
        col_data = data[col].str.decode('utf-8')
    utf_8_data = utf_8_data.join(col_data)

X_original = preprocessFeatures(utf_8_data, [0,1,2,7], False).to_numpy()

y = np.array(list(map(int, X_original[:,24])))
X_original = X_original[:,0:24]

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)
support_vector_scaled, cv_score_scaled = svmClassifier(X_train_scaled, y_train_scaled)

print('#'*10, 'Seismic bumps', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

print('support vector machine (accuracy with scaling): ', accuracy_score(support_vector_scaled.predict(X_test_scaled), y_test_scaled))
print('Cross validation (with scaling):', cv_score_scaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')


################################################################################################################################################################################

#Adult


path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/adult.data'
dataset = pd.read_csv(path, header=None)

dataset = preprocessFeatures(dataset, [], True).to_numpy()
X_original = dataset[:,: dataset.shape[1] - 1]
y = dataset[:,dataset.shape[1] -2]

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'Adult', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')


############################################################################################################################################################################

ThoraricSurgery


path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/ThoraricSurgery.arff'

data, meta = arff.loadarff(path)
temp = data.tolist()
data = pd.DataFrame(temp)

utf_8_data = pd.DataFrame(index=data.index)

for col, col_data in data.iteritems():
    if(col_data.dtype == object):
        col_data = data[col].str.decode('utf-8')
    utf_8_data = utf_8_data.join(col_data)

X_original = preprocessFeatures(utf_8_data, [0,3,4,5,6,7,8,9,10,11,12,13,14,16], False).to_numpy()

y = np.array(list(map(int, X_original[:,37])))
X_original = X_original[:,0:37]

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)


support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'ThoraricSurgery', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')

############################################################################################################################################################################

#Yeast

path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/yeast.data'

data = np.genfromtxt(path, dtype=str)

dataset = preprocessFeatures(pd.DataFrame(data), [0], False).to_numpy()

y = dataset[:, dataset.shape[1] - 1]
X_original = dataset[:, 0:dataset.shape[1] - 1]


X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)


support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'Yeast', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')

############################################################################################################################################################################

#Faults

path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/Faults.NNA'

dataset = np.loadtxt(path, delimiter='\t')

y = merge(dataset[:, dataset.shape[1]-7:])
X_original = StandardScaler().fit(dataset[:,:dataset.shape[1]-7]).transform(dataset[:,:dataset.shape[1]-7])

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'Fault', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')


############################################################################################################################################################################

#default_of_credit_card_clients
path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/default_of_credit_card_clients.xls'

dataset = pd.read_excel(path, skiprows=[1])

data = dataset.to_numpy()
y = data[:, data.shape[1] -1]
X_original = data[:, 1: data.shape[1] -1]



X_original = StandardScaler().fit(X_original).transform(X_original)

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.4, random_state=0)

support_vector_scaled, cv_score_scaled = svmClassifier(X_train, y_train)

print('#'*10, 'default_of_credit_card_clients', '#'*10)

print('\n')

print('support vector machine (accuracy with scaling): ', accuracy_score(support_vector_scaled.predict(X_test), y_test))
print('Cross validation (with scaling):', cv_score_scaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')

############################################################################################################################################################################

#German

path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/german.data'

data = np.genfromtxt(path, dtype=str)

dataset = preprocessFeatures(pd.DataFrame(data), [0,2,3,5,6,8,9,11,13,14,16,18,19], False).to_numpy()

y = dataset[:, dataset.shape[1] - 1]
X_original = dataset[:, 0:dataset.shape[1] - 1]


X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'Yeast', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')

############################################################################################################################################################################

#Australian

path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/australian.dat'

data = pd.DataFrame(np.loadtxt(path))

dataset = preprocessFeatures(data, [0,3,4,5,7,8,10,11], False).to_numpy()

y = dataset[:, dataset.shape[1] - 1]
X_original = dataset[:, 0:dataset.shape[1] - 1]


X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0)

support_vector_unscaled, cv_score_unscaled = svmClassifier(X_train, y_train)

print('#'*10, 'Australian', '#'*10)

print('\n')

print('support vector machine (accuracy without scaling): ', accuracy_score(support_vector_unscaled.predict(X_test), y_test))
print('Cross validation (without scaling):', cv_score_unscaled)

print('\n')

#decision tree
decision_tree, cv_score = decisionTree(X_train, y_train)

print('decision tree:', accuracy_score(decision_tree.predict(X_test), y_test))
print('Cross validation:', cv_score)

print('\n')
