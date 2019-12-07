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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, WhiteKernel

# MERCK_FILE = r'..\..\..\data\regression\merck-molecular-activity\TrainingSet\ACT4_competition_training.csv'
# with open(MERCK_FILE) as f:
#     cols = f.readline().rstrip('\n').split(',') # Read the header line and get list of column names
#     # Load the actual data, ignoring first column and using second column as targets.
#     X = np.loadtxt(MERCK_FILE, delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8)
#     y = np.loadtxt(MERCK_FILE, delimiter=',', usecols=[1], skiprows=1)

dataset = pd.read_csv(r'data/regression/merck-molecular-activity/TrainingSet/ACT4_competition_training.csv')
dataset.drop(['MOLECULE'], axis=1, inplace=True)
y = StandardScaler().fit_transform(dataset.pop('Act').values.reshape(-1, 1)).ravel()

bins = list()
drops = list()
cats = list()
for i, col in enumerate(dataset.columns):
    # print(dataset[col].value_counts())
    if len(dataset[col].value_counts()) == 1:
        drops.append(col)
        # print(col, '=> min:', min(dataset[col].values), '--  max:', max(dataset[col].values), end='')
        # print(' --  values count', len(dataset[col].value_counts()))
    elif len(dataset[col].value_counts()) == 2:
        bins.append(col)
    else:
        cats.append(col)
else:
    dataset.drop(drops, axis=1, inplace=True)

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
num_si_step = ('si', SimpleImputer(strategy='constant'))
sc_step = ('sc', StandardScaler())

bin_oe_step = ('le', OrdinalEncoder())
bin_si_step = ('si', SimpleImputer(strategy='most_frequent'))

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([bin_si_step, ])

transformers = [
    ('cat', cat_pipe, []),
    ('num', num_pipe, cats),
    ('bin', bin_pipe, bins),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed)

linear_r_pipe = Pipeline([
    ('X_transform', ct),
    ('linear_r', LinearRegression(fit_intercept=True)),
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
    ('gp', GaussianProcessRegressor()),
])

gp_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    # 'gp__alpha': [0.0, 1e-01, 1e-03, 1e-05],
    # 'gp__kernel': [k4]
    # 'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

scoring = 'r2'
gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
gs_linear_r.fit(dataset, y)
print("##################### Linear Regression for Sgemm Dataset ############################")
print('CV Score', gs_linear_r.best_score_)
print('Training accuracy', gs_linear_r.score(dataset, y))
print("##################### GaussianProcessRegressor for Sgemm Dataset #####################")
gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, return_train_score=True, scoring=scoring)
gs_gp.fit(dataset, y)
print('CV Score', gs_gp.best_score_)
print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR MERCK_MOLECULAR DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR MERCK_MOLECULAR DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     handler.write('#' * 120)
#     handler.write('\n')
