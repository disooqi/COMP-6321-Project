"""

https://www.kaggle.com/c/MerckActivity/overview/evaluation
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error


# MERCK_FILE = r'..\..\..\data\regression\merck-molecular-activity\TrainingSet\ACT4_competition_training.csv'
# with open(MERCK_FILE) as f:
#     cols = f.readline().rstrip('\n').split(',') # Read the header line and get list of column names
#     # Load the actual data, ignoring first column and using second column as targets.
#     X = np.loadtxt(MERCK_FILE, delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8)
#     y = np.loadtxt(MERCK_FILE, delimiter=',', usecols=[1], skiprows=1)

dataset = pd.read_csv(r'..\..\..\data\regression\merck-molecular-activity\TrainingSet\ACT4_competition_training.csv')
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
bin_pipe = Pipeline([bin_si_step,])

transformers = [
    ('cat', cat_pipe, []),
    ('num', num_pipe, cats),
    ('bin', bin_pipe, bins),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed)


mlp_pipe = Pipeline([
    ('X_transform', ct),
    ('mlp', MLPRegressor()),
])

kf = KFold(n_splits=5, shuffle=True, )
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

mlp_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'mlp__solver': [ 'adam', 'lbfgs'],
    'mlp__alpha': [1e-1, 1e-3, 1e-5],
    'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}

rf_pipe = Pipeline([
    ('X_transform', ct),
    ('rf', RandomForestRegressor()),
])

rf_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

ml_pipe = mlp_pipe
ml_param_grid = mlp_param_grid

scoring = 'r2'
gs = GridSearchCV(ml_pipe, ml_param_grid, cv=kf, return_train_score=True, scoring=scoring)
gs.fit(dataset, y)

file_name = os.path.splitext(os.path.basename(__file__))[0]
with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
    handler.write(f'The CV best {scoring} score: {gs.best_score_:.4f}\n')
    handler.write(f'The train set {scoring} score: {gs.score(dataset, y):.4f}\n')
    handler.write(f'{gs.best_params_}\n')
    handler.write(f'{gs.best_estimator_}\n')
    # handler.write(f'{gs.grid_scores_}\n')
    # pd.DataFrame(gs.cv_results_).to_html(f'{file_name}-cv_results_.html')
    # handler.write(f'{pd.DataFrame(gs.cv_results_)}\n')
    handler.write('#'*120)
    handler.write('\n')
