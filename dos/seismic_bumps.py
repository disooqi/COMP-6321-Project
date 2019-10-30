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


data, meta = arff.loadarff(r'..\data\seismic-bumps\seismic-bumps.arff')
dataset = pd.DataFrame(data)

y = LabelEncoder().fit_transform(dataset.pop('class').values)

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['seismic', 'seismoacoustic', 'ghazard']),
    ('num', num_pipe, ['genergy', 'gpuls', 'gdenergy', 'gdpuls', 'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5',
                       'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy']),
    ('bin', bin_pipe, ['shift']),
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

