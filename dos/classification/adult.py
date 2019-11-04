import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV


dataset = pd.read_csv(r'..\..\data\adult\adult.data', header=None, skipinitialspace=True,
                      names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                             'hours-per-week', 'native-country', 'target'], na_values=['?'], keep_default_na=False)

testset = pd.read_csv(r'..\..\data\adult\adult.test', header=None, skipinitialspace=True,
                      names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                             'hours-per-week', 'native-country', 'target'],
                      skiprows=1, na_values=['?'], keep_default_na=False)
# print(dataset.dtypes.head(20))
# print(np.array([dt.kind for dt in dataset.dtypes]))


y = LabelEncoder().fit_transform(dataset.pop('target').values)
y_test = LabelEncoder().fit_transform(testset.pop('target').values)

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']),
    ('num', num_pipe, ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']),
    ('bin', bin_pipe, ['sex']),
]
ct = ColumnTransformer(transformers=transformers)
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

gs = GridSearchCV(ml_pipe, param_grid, cv=kf)
gs.fit(dataset, y)
print(gs.best_params_)
print('The CV best score:', gs.best_score_)
# print(pd.DataFrame(gs.cv_results_))

print(f'The train set score: {gs.score(dataset, y)} ')
print(f'The test set score: {gs.score(testset, y_test)} ')


