'''
__created__= '31 Oct 2019'
__developer__ = 'disooqi@gmail.com'
'''

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

dataset = pd.read_csv(r'..\..\data\Steel_plates_faults\Faults.NNA', header=None, delimiter='	',
    names=['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
           'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
           'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
           'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
           'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas', 'Pastry',
           'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])

# y = dataset[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]
y = dataset.loc[:, 'Pastry':'Other_Faults']
dataset.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], axis=1, inplace=True)

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['Outside_Global_Index']),
    ('num', num_pipe, ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
                       'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
                       'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
                       'Edges_X_Index', 'Edges_Y_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
                       'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']),
    ('bin', bin_pipe, ['TypeOfSteel_A300', 'TypeOfSteel_A400']),
]

ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed)

ml_pipe = Pipeline([
    ('X_transform', ct),
    ('mlp', MLPClassifier(activation='identity', solver='sgd', alpha=1e-1, hidden_layer_sizes=(4, 4))),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    'mlp__alpha': [1e-1, 1e-3, 1e-5],
    'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}

knn_pipe = Pipeline([
    ('X_transform', ct),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
])

# knn_pipe.fit(dataset, y)
# print(f'All data score: {knn_pipe.score(dataset, y)}')

knn_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'knn__n_neighbors': range(1, 10),
}

gs = GridSearchCV(knn_pipe, knn_param_grid, cv=kf)
gs.fit(dataset, y)
print(gs.best_params_)
print('The CV best score:', gs.best_score_)
# print(pd.DataFrame(gs.cv_results_))

print(f'The train set score: {gs.score(dataset, y)} ')

