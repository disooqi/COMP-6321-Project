from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import reciprocal
from sklearn.svm import SVR
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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




def scale(X):
    """The following function is used to scale the features
    @X This is the feature vector of type numpy"""
    scalar = StandardScaler().fit(X)
    return scalar.transform(X)





def svmRegressor(X, y, cv_size = 5):
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

    #score = make_scorer(mean_squared_error)

    temp_cls_ = SVR()

    parameters = {
    'C': np.arange(1,3, 1),
    'degree': np.arange(2,5),
    'kernel': ('rbf', 'poly')
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    #param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size, scoring=score)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls


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

    temp_cls_ = DecisionTreeRegressor()

    parameters = {
    'max_depth': (50,100,150,200),
    'min_samples_split': (50,40,30,20,10,2),
    'random_state' : [0]
    }

    param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls

def decisionTreeMulti(X, y):
    """The following function is used to perform decision tree for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns MultiOutputRegressor object

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'max_depth': (50,100,150,200),
    'min_samples_split': (50,40,30,20,10,2,1),
    'random_state' : [0]
    }"""

    temp_ = DecisionTreeRegressor()


    parameters = {
    'estimator__max_depth': (50,100,150,200),
    'estimator__min_samples_split': (50,40,30,20,10,2),
    'estimator__random_state' : [0]
    }

    param_tuner_ = GridSearchCV(MultiOutputRegressor(temp_), param_grid=parameters)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls


def svmRegressorMultiple(X, y):
    """The following function is used to perform SVM for classification.
    @X This is the feature vector of type numpy
    @y labels of type numpy
    @return It returns MultiOutputRegressor object and accuracy score value

    This function uses GridSearchCV to tune the parameters for the best performance
    The parameters are as follows:
    {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'degree': [2,3,4,5,6]
    'C': [1,4,10],
    'gamma': ['auto', 'scale']
    }"""

    #score = make_scorer(mean_squared_error)

    temp_cls_ = SVR()

    parameters = {
    'estimator__C': np.arange(1,3,1),
    'estimator__degree': np.arange(2,5),
    'estimator__kernel': ('rbf', 'poly')
    }

    param_tuner_ = GridSearchCV(MultiOutputRegressor(temp_cls_), param_grid=parameters)
    #param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size)
    #param_tuner_ = GridSearchCV(temp_cls_, parameters, cv=cv_size, scoring=score)
    param_tuner_.fit(X, y)
    cls = param_tuner_.best_estimator_.fit(X, y)
    return cls


# # Wine
#
# path_red = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/winequality-red.csv'
# path_white = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/winequality-white.csv'
# data_red = pd.read_csv(path_red, delimiter=';').to_numpy()
# data_white = pd.read_csv(path_white, delimiter=';').to_numpy()
#
#
# X_original_red = data_red[:,:data_red.shape[1]- 1]
# X_original_red = StandardScaler().fit(X_original_red).transform(X_original_red)
# X_original_red = np.concatenate((np.ones((X_original_red.shape[0],1)), X_original_red) , axis=1)
# y_red = data_red[:,data_red.shape[1] - 1]
#
#
# X_original_white = data_white[:,:data_white.shape[1]- 1]
# X_original_white = np.concatenate((np.zeros((X_original_white.shape[0],1)), X_original_white) , axis=1)
#
# X_original_white = StandardScaler().fit(X_original_white).transform(X_original_white)
# y_white = data_white[:,data_white.shape[1] - 1]
#
# X_original = np.concatenate((X_original_red, X_original_white), axis=0)
# y = np.concatenate((y_red, y_white), axis=0)
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# support_vector = svmRegressor(X_train, y_train, 8)
# #support_vector = SVR().fit(X_train, y_train)
#
# print('#'*10, 'Wine', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')


# DecisionTreeRegressor


#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')


################################################################################################################################################################################

# # qsar_aquatic_toxicity
#
# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/qsar_aquatic_toxicity.csv'
# data = pd.read_csv(path, delimiter=';', header=None).to_numpy()
#
# X_original = data[:, :data.shape[1] - 1]
# y = data[:, data.shape[1] - 1]
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# support_vector = svmRegressor(X_train, y_train)
#
# print('#'*10, 'qsar_aquatic_toxicity', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
# # DecisionTreeRegressor
#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

################################################################################################################################################################################
#student

# path_train = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/student-por.csv'
# path_test = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/student-mat.csv'
# data_train = pd.read_csv(path_train, delimiter=';')
#
# data_train = preprocessFeatures(data_train, [], True).to_numpy()
# X_train = data_train[:, :data_train.shape[1] - 3]
# y_train = data_train[:, data_train.shape[1] - 3:]
#
#
#
# data_test = pd.read_csv(path_test, delimiter=';')
#
# data_test = preprocessFeatures(data_test, [], True).to_numpy()
# X_test = data_test[:, :data_test.shape[1] - 3]
# y_test = data_test[:, data_test.shape[1] - 3:]
#
#
#
# #support_vector = MultiOutputRegressor(SVR()).fit(X_train, y_train)
# support_vector = svmRegressorMultiple(X_train, y_train)
#
# print('#'*10, 'student', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
# #DecisionTreeRegressor
#
# tree = decisionTreeMulti(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')
#

################################################################################################################################################################################

#Concrete_Data

# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/Concrete_Data.xls'
# dataset = pd.read_excel(path, skiprows=[1]).to_numpy()
#
# X_original = dataset[:, :dataset.shape[1]-1]
# y = dataset[:, dataset.shape[1]-1]
# X_original = scale(X_original)
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# #support_vector = SVR().fit(X_train, y_train)
# support_vector = svmRegressor(X_train, y_train)
#
# print('#'*10, 'Concrete_Data', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
#
# # DecisionTreeRegressor
#
#
# #tree = MultiOutputRegressor(DecisionTreeRegressor(min_samples_split=150)).fit(X_train, y_train)
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')



################################################################################################################################################################################

#parkinsons

# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/parkinson_train.txt'
# dataset = np.loadtxt(path, delimiter=',')
#
# X_original = dataset[:, :dataset.shape[1]-1]
# y = dataset[:, dataset.shape[1]-1]
# X_original = scale(X_original)
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# #support_vector = SVR().fit(X_train, y_train)
# support_vector = svmRegressor(X_train, y_train)
#
# print('#'*10, 'parkinsons', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
#
# # DecisionTreeRegressor
#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

################################################################################################################################################################################

# #Bike
#
# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/hour.csv'
# dataset = pd.read_csv(path).to_numpy()[:,2:]
#
# X_original = dataset[:, :dataset.shape[1]-1]
# y = dataset[:, dataset.shape[1]-1]
# X_original = scale(X_original)
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# #support_vector = SVR().fit(X_train, y_train)
# support_vector = svmRegressor(X_train, y_train)
#
# print('#'*10, 'Bike', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
#
# # DecisionTreeRegressor
#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

################################################################################################################################################################################

#Facebook
# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/dataset_Facebook.csv'
# dataset = pd.read_csv(path, delimiter=';')
#
# y = dataset['Total Interactions']
# dataset = dataset.drop(['Total Interactions'], axis=1)
# cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
# ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
# num_si_step = ('si', SimpleImputer(strategy='constant'))
# sc_step = ('sc', StandardScaler())
# oe_step = ('le', OrdinalEncoder())
# bin_si_step = ('si', SimpleImputer(strategy='most_frequent'))
#
# cat_pipe = Pipeline([cat_si_step, ohe_step])
# num_pipe = Pipeline([num_si_step, sc_step])
# bin_pipe = Pipeline([bin_si_step, oe_step])
#
# transformers = [
#     ('cat', cat_pipe, ['Type', 'Category', 'Post Month', 'Post Weekday', 'Post Hour']),
#     ('num', num_pipe, ['Page total likes', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
#                        'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
#                        'Lifetime Post Impressions by people who have liked your Page', 'comment', 'like', 'share']),
#     ('bin', bin_pipe, ['Paid']),
# ]
# ct = ColumnTransformer(transformers=transformers)
# X_original = ct.fit_transform(dataset)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# print(X_train)
# #support_vector = SVR().fit(X_train, y_train)
# support_vector = svmRegressor(X_train, y_train)
#
# print('#'*10, 'Facebook', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
#
# # DecisionTreeRegressor
#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

################################################################################################################################################################################

# #gpu
# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/sgemm_product.csv'
# dataset = pd.read_csv(path, delimiter=',')
#
# y = dataset[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']]
# X_original = dataset.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis=1)
#
# X_original = scale(X_original)
#
# X_original = X_original[:10000,:]
# y = y[:10000]
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
# support_vector = svmRegressorMultiple(X_train, y_train)
# #support_vector = MultiOutputRegressor(SVR()).fit(X_train, y_train)
#
# print('#'*10, 'gpu', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
# #DecisionTreeRegressor
#
# tree = decisionTreeMulti(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

###############################################################################################################################################################################

# # communities
#
# path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/communities.data'
#
# dataset = pd.read_csv(path, header=None).replace('?', np.NaN)
# dataset = dataset.convert_objects(convert_numeric=True)
#
# for i in dataset.columns:
#     if dataset[i].dtype in [np.float64, np.int64]:
#         dataset[i] = dataset[i].fillna(dataset[i].mean())
#
#
# dataset = preprocessFeatures(dataset, [], True).to_numpy()
#
# X_original = dataset[:, :dataset.shape[1]-1]
# y = dataset[:, dataset.shape[1]-1]
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)
#
#
# support_vector = svmRegressor(X_train, y_train)
# #support_vector = SVR().fit(X_train, y_train)
#
# print('#'*10, 'communities', '#'*10)
#
# print('\n')
#
# print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
# print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
# print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
# print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))
#
# print('\n')
#
#
#
# # DecisionTreeRegressor
#
# tree = decisionTree(X_train, y_train)
# print('\n')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')

###############################################################################################################################################################################

# merck

path_one = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/ACT2_competition_training.csv'
path_two = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/ACT4_competition_training.csv'

dataset_one = pd.read_csv(path_one)
dataset_two = pd.read_csv(path_two)

frames = [dataset_one, dataset_two]

dataset = pd.concat(frames)

print('Initial shape : ',dataset.shape)
corr_matrix = dataset.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] >= 0.6)]

dataset = dataset.drop(dataset[to_drop], axis=1)
dataset = dataset.drop('MOLECULE', 1)

print('After drop shape : ',dataset.shape)

for i in dataset.columns:
    if dataset[i].dtype in [np.float64, np.int64]:
        dataset[i] = dataset[i].fillna(dataset[i].mean())

print('Cleared NaN')
dataset = dataset.to_numpy()
X_original = dataset[:, :dataset.shape[1]-1]
y = dataset[:, dataset.shape[1]-1]


X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.45, random_state=0, shuffle=True)


#support_vector = svmRegressor(X_train, y_train)
support_vector = SVR().fit(X_train, y_train)

print('#'*10, 'Merck', '#'*10)

print('\n')

print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))

print('\n')



# DecisionTreeRegressor

tree = decisionTree(X_train, y_train)
print('\n')

print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))

print('\n')
