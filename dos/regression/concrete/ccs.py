"""
http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete
Research, Vol. 28, No. 12, pp. 1797-1808 (1998).
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

#  data/regression/concrete-compressive-strength/Concrete_Data.xls
dataset = pd.read_excel(r'..\..\..\data\regression\concrete-compressive-strength\Concrete_Data.xls', sheet_name=0)
y = dataset.pop('Concrete compressive ')

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
    ('cat', cat_pipe, []),
    ('num', num_pipe, ['Cement (component 1)(kg in a m^3 mixture)',
                       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
                       'Fly Ash (component 3)(kg in a m^3 mixture)',
                       'Water  (component 4)(kg in a m^3 mixture)',
                       'Superplasticizer (component 5)(kg in a m^3 mixture)',
                       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
                       'Fine Aggregate (component 7)(kg in a m^3 mixture)',
                       'Age (day)',
                       ]),
    ('bin', bin_pipe, []),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed)

mlp_pipe = Pipeline([
    ('X_transform', ct),
    ('mlp', MLPRegressor()),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

mlp_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'mlp__solver': ['sgd', 'adam', 'lbfgs'],
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
