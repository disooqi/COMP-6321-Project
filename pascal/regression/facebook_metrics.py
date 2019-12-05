import os
import pandas as pd
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ConstantKernel, DotProduct, WhiteKernel

dataset = pd.read_csv(r'data/regression/facebook-metrics/dataset_Facebook.csv', delimiter=';')
y = StandardScaler().fit_transform(dataset.pop('Total Interactions').values.reshape(-1, 1)).ravel()

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
num_si_step = ('si', SimpleImputer(strategy='constant'))
sc_step = ('sc', StandardScaler())
oe_step = ('le', OrdinalEncoder())
bin_si_step = ('si', SimpleImputer(strategy='most_frequent'))

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([bin_si_step, oe_step])

transformers = [
    ('cat', cat_pipe, ['Type', 'Category', 'Post Month', 'Post Weekday', 'Post Hour']),
    ('num', num_pipe, ['Page total likes', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                       'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                       'Lifetime Post Impressions by people who have liked your Page', 'comment', 'like', 'share']),
    ('bin', bin_pipe, ['Paid']),
]

ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed.shape)

linear_r_pipe = Pipeline([
    ('X_transform', ct),
    ('linear_r', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

linear_r_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    # 'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    # 'mlp__alpha': [1e-1, 1e-3, 1e-5],
    # 'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}
k1 = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
k2 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
k3 = DotProduct() + WhiteKernel()
k4 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

gp_pipe = Pipeline([
    ('X_transform', ct),
    ('gp',
     GaussianProcessRegressor(alpha=0.0, copy_X_train=True, kernel=DotProduct(sigma_0=1) + WhiteKernel(noise_level=1))),
])

gp_param_grid = {
    'X_transform__num__si__strategy': ['mean'],
    # 'gp__alpha': [1e-01, 1e-03, 1e-05],
    # 'gp__kernel': [DotProduct() + WhiteKernel(), k1, k2, k3, k4],
    # 'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

scoring = 'r2'
gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
gs_linear_r.fit(dataset, y)
print("##################### Linear Regression for FacebookM Dataset ############################")
print('CV Score', gs_linear_r.best_score_)
print('Training accuracy', gs_linear_r.score(dataset, y))
print("##################### GaussianProcessRegressor for FacebookM Dataset #####################")
gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, return_train_score=True, scoring=scoring)
gs_gp.fit(dataset, y)
print('CV Score', gs_gp.best_score_)
print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR FACEBOOK DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR FACEBOOK DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     handler.write('#' * 120)
#     handler.write('\n')


#
#
#
# #Facebook
# path = 'data/regression/facebook-metrics/dataset_Facebook.csv'
# #path = '/Users/sandeepchowdaryannabathuni/desktop/project/datasets/dataset_Facebook.csv'
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
# support_vector = LinearRegression().fit(X_train, y_train)
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
# #
# tree = GaussianProcessRegressor().fit(X_train, y_train)
# print('\n', '#########################')
#
# print('DecisionTreeRegressor (r2 on training): ', tree.score(X_train, y_train))
# print('DecisionTreeRegressor (MAE on train): ', mean_absolute_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MSE on train): ', mean_squared_error(tree.predict(X_train), y_train))
# print('DecisionTreeRegressor (MAE on test): ', mean_absolute_error(tree.predict(X_test), y_test))
# print('DecisionTreeRegressor (MSE on test): ', mean_squared_error(tree.predict(X_test), y_test))
#
# print('\n')
