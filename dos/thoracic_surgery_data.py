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

data, meta = arff.loadarff(r'..\data\thoracic_surgery_data\ThoraricSurgery.arff')
dataset = pd.DataFrame(data)

y = LabelEncoder().fit_transform(dataset.pop('Risk1Yr').values)

si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
# In dev and test data, new values will raise an error due to handle_unknown='error'
# Set handle_unknown='ignore', will mask OOV and missing value problems in dev and test data
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
oe_step = ('le', OrdinalEncoder())

num_si_step = ('si', SimpleImputer(strategy='mean'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['DGN', 'PRE6', 'PRE14']),
    ('num', num_pipe, ['PRE4', 'PRE5', 'AGE']),
    ('bin', bin_pipe, ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)

ml_pipe = Pipeline([
    ('X_transform', ct),
    ('mlp', MLPClassifier()),
])


kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    'mlp__alpha': [1e-1, 1e-3, 1e-5],
    'mlp__hidden_layer_sizes': [(5, 2), (4, 3), (4, 4), (5, 5)],

}

knn_pipe = Pipeline([
    ('X_transform', ct),
    ('knn', KNeighborsClassifier(n_neighbors=8)),
])

knn_pipe.fit(dataset, y)
print(f'All data score: {knn_pipe.score(dataset, y)}')

knn_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'knn__n_neighbors': range(1, 10),
}

gs = GridSearchCV(knn_pipe, knn_param_grid, cv=kf)
gs.fit(dataset, y)
print(gs.best_params_)
print(gs.best_score_)
print(pd.DataFrame(gs.cv_results_))
