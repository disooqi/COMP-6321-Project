'''
http://archive.ics.uci.edu/ml/datasets/Student+Performance
http://www3.dsi.uminho.pt/pcortez/student.pdf
'''
import os
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

## data/regression/student-performance/student/student-mat.csv

dataset_r = pd.read_csv(r'data/regression/wine-quality/winequality-red.csv',
                        delimiter=';', )  # header=None, usecols=[3,6]
dataset_w = pd.read_csv(r'data/regression/wine-quality/winequality-white.csv', delimiter=';')
# dataset.pop('ID')
dataset_r['red?'] = True
dataset_w['red?'] = False
# print(dataset_r)
# print(dataset_w)

dataset = pd.concat([dataset_w, dataset_r, ], ignore_index=True)
y = dataset.pop('quality').values

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, []),
    ('num', num_pipe, ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                       'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']),
    ('bin', bin_pipe, ['red?']),
]
ct = ColumnTransformer(transformers=transformers)

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='constant'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed.shape)

linear_r_pipe = Pipeline([
    ('X_transform', ct),
    ('linear_r', LinearRegression(fit_intercept=True)),
])

kf = KFold(n_splits=5, shuffle=False)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

linear_r_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    # 'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    # 'mlp__alpha': [1e-1, 1e-3, 1e-5],
    # 'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp_pipe = Pipeline([
    ('X_transform', ct),
    ('gp', GaussianProcessRegressor()),
])

gp_param_grid = {
    'X_transform__num__si__strategy': ['mean'],
    # 'gp__alpha': [0.1],
    # 'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

scoring = 'r2'
gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
gs_linear_r.fit(dataset, y)
print("##################### Linear Regression for WineQ Dataset ############################")
print('CV Score', gs_linear_r.best_score_)
print('Training accuracy', gs_linear_r.score(dataset, y))
print("##################### GaussianProcessRegressor for WineQ Dataset #####################")
gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, scoring=scoring)
gs_gp.fit(dataset, y)
print('CV Score', gs_gp.best_score_)
print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR WINE_QUALITY DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     # handler.write(f'{gs.grid_scores_}\n')
#     # pd.DataFrame(gs.cv_results_).to_html(f'{file_name}-cv_results_.html')
#     # handler.write(f'{pd.DataFrame(gs.cv_results_)}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR WINE_QUALITY DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     # handler.write(f'{gs.grid_scores_}\n')
#     # pd.DataFrame(gs.cv_results_).to_html(f'{file_name}-cv_results_.html')
#     # handler.write(f'{pd.DataFrame(gs.cv_results_)}\n')
#     handler.write('#'*120)
#     handler.write('\n')
