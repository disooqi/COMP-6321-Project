import os
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.svm import LinearSVC

dataset = pd.read_csv(r'data/regression/SGEMM-GPU-kernel-performance/sgemm_product_dataset/sgemm_product.csv')
y = StandardScaler().fit_transform(dataset.loc[:, 'Run1 (ms)':'Run4 (ms)'])
dataset.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis=1, inplace=True)
print(dataset.shape)
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(dataset, y)
# print(clf.feature_importances_)
clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', LinearRegression())
])
# clf.fit(dataset, y)
print(clf)

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
#     ('cat', cat_pipe, ['MWG', 'NWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB', 'VWM', 'VWN']),
#     ('num', num_pipe, []),
#     ('bin', bin_pipe, ['KWG', 'KWI', 'STRM', 'STRN', 'SA', 'SB']),
# ]
#
# ct = ColumnTransformer(transformers=transformers)
# # X_transformed = ct.fit_transform(dataset)
# # print(X_transformed.shape)
#
# linear_r_pipe = Pipeline([
#     ('X_transform', ct),
#     ('linear_r', LinearRegression(fit_intercept=True)),
# ])
#
# kf = KFold(n_splits=5, shuffle=True)
# # cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()
#
# linear_r_param_grid = {
#     'X_transform__num__si__strategy': ['mean', 'median'],
#     # 'mlp__solver': ['sgd', 'adam', 'lbfgs'],
#     # 'mlp__alpha': [1e-1, 1e-3, 1e-5],
#     # 'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
#     # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
# }
# k1 = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
#     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
# k2 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
#     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
# k3 = DotProduct() + WhiteKernel()
# k4 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp_pipe = Pipeline([
#     ('X_transform', ct),
#     ('gp', MultiOutputRegressor(GaussianProcessRegressor())),
# ])
#
# gp_param_grid = {
#     'X_transform__num__si__strategy': ['mean', 'median'],
#     #'gp__alpha': [0.0, 1e-01, 1e-03, 1e-05],
#     #'gp__kernel': [k3]
#     # 'rf__n_estimators': range(1, 20),
#     # 'rf__criterion': ['mse', 'mae'],
# }
#
# scoring = 'r2'
# gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
# gs_linear_r.fit(dataset, y)
# print("##################### Linear Regression for Sgemm Dataset ############################")
# print('CV Score', gs_linear_r.best_score_)
# print('Training accuracy', gs_linear_r.score(dataset, y))
# print("##################### GaussianProcessRegressor for Sgemm Dataset #####################")
# gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, return_train_score=True, scoring=scoring)
# gs_gp.fit(dataset, y)
# print('CV Score', gs_gp.best_score_)
# print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR STUDENT_PERFORMANCE DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR STUDENT_PERFORMANCE DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     handler.write('#' * 120)
#     handler.write('\n')
