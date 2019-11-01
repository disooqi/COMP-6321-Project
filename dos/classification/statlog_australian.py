import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV


dataset = pd.read_csv(r'..\..\data\statlog\australian\australian.dat', header=None, delimiter=' ',
                      names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
                             'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'])

y = LabelEncoder().fit_transform(dataset.pop('A15').values)

si_step = ('si', SimpleImputer(strategy='constant', fill_value=-1))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, ['A4', 'A5', 'A6', 'A12']),
    ('num', num_pipe, ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']),
    ('bin', bin_pipe, ['A1', 'A8', 'A9', 'A11']),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)


ml_pipe = Pipeline([
    ('X_transform', ct),
    ('mlp', MLPClassifier(activation='relu', solver='adam', alpha=1e-3, hidden_layer_sizes=(4, 4))),
])


kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    'mlp__alpha': [1e-1, 1e-3, 1e-5],
    'mlp__hidden_layer_sizes': [(5, 2), (4, 3), (4, 4), (5, 5)],
    'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],

}

knn_pipe = Pipeline([
    ('X_transform', ct),
    ('knn', KNeighborsClassifier(n_neighbors=9)),
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

