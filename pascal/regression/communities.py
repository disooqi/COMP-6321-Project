import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct


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

    if (auto is True):
        for col, col_data in X.iteritems():
            if (col_data.dtype == object):
                col_data = pd.get_dummies(col_data)
            output = output.join(col_data, how='left', lsuffix='_left', rsuffix='_right')

    else:
        for col, col_data in X.iteritems():
            if (col in index_):
                col_data = pd.get_dummies(col_data)
            output = output.join(col_data, how='left', lsuffix='_left', rsuffix='_right')

    return output


path = 'data/regression/communities-and-crime-data-set/communities.data'

dataset = pd.read_csv(path, header=None).replace('?', np.NaN)
dataset = dataset.convert_objects(convert_numeric=True)

for i in dataset.columns:
    if dataset[i].dtype in [np.float64, np.int64]:
        dataset[i] = dataset[i].fillna(dataset[i].mean())

dataset = preprocessFeatures(dataset, [], True).to_numpy()

X_original = dataset[:, :dataset.shape[1] - 1]
y = dataset[:, dataset.shape[1] - 1]

X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.25, random_state=0, shuffle=True)

support_vector = LinearRegression(X_train, y_train)
# support_vector = SVR().fit(X_train, y_train)

print('#' * 10, 'communities', '#' * 10)

print('\n')

print('support vector machine (r2 on training): ', support_vector.score(X_train, y_train))
print('support vector machine (MAE on train): ', mean_absolute_error(support_vector.predict(X_train), y_train))
print('support vector machine (MSE on train): ', mean_squared_error(support_vector.predict(X_train), y_train))
print('support vector machine (MAE on test): ', mean_absolute_error(support_vector.predict(X_test), y_test))
print('support vector machine (MSE on test): ', mean_squared_error(support_vector.predict(X_test), y_test))

print('\n')

# DecisionTreeRegressor

tree = GaussianProcessRegressor(X_train, y_train)
print('\n')

print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))

print('\n')
