import os
import pandas as pd
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

dataset = pd.read_csv(r'data/regression/bike-sharing/hour.csv')

dataset.drop(['instant', 'dteday'], axis=1, inplace=True)

"""
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
"""
# y = StandardScaler().fit_transform(dataset.pop('cnt').values.reshape(-1, 1))
# y = y.flatten()
y = dataset.pop('cnt').values

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='constant'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['season', 'mnth', 'weekday', 'weathersit', 'hr']),
    ('num', num_pipe, ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']),
    ('bin', bin_pipe, ['yr', 'holiday', 'workingday']),
]

ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed.shape)

linear_r_pipe = Pipeline([
    ('X_transform', ct),
    ('linear_r', LinearRegression(fit_intercept=True)),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

linear_r_param_grid = {
    'X_transform__num__si__strategy': ['mean'],
    # 'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    # 'mlp__alpha': [1e-1, 1e-3, 1e-5],
    # 'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp_pipe = Pipeline([
    ('X_transform', ct),
    ('gp', GaussianProcessRegressor()),
])

gp_param_grid = {
    'X_transform__num__si__strategy': ['mean'],
    # 'gp__alpha': [1e-01, 1e-03, 1e-05],
    # 'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

scoring = 'r2'
gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
gs_linear_r.fit(dataset, y)
print("##################### Linear Regression for BikeSharing Dataset ############################")
print('CV Score', gs_linear_r.best_score_)
print('Training accuracy', gs_linear_r.score(dataset, y))
print("##################### GaussianProcessRegressor for BikeSharing Dataset #####################")
gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, return_train_score=True, scoring=scoring)
gs_gp.fit(dataset, y)
print('CV Score', gs_gp.best_score_)
print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR BIKE_SHARING DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR BIKE_SHARING DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     handler.write('#' * 120)
#     handler.write('\n')
