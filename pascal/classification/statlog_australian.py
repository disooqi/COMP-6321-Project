import pandas as pd
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

dataset = pd.read_csv(r'data/statlog/australian/australian.dat', header=None, delimiter=' ',
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

lr_pipe = Pipeline([
    ('X_transform', ct),
    ('lr', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1500)),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    'lr__penalty': ['l2'],
    'lr__solver': ['saga', 'lbfgs', 'liblinear', 'sag', 'newton-cg'],
}
lr_pipe.fit(dataset, y)
nb_pipe = Pipeline([
    ('X_transform', ct),
    ('lr', GaussianNB(priors=None, var_smoothing=1e-01)),
])

nb_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],

}
print(
    "########################################## LOGISTIC REGRESSION ON STATLOG_AUSTRALIAN DATA ########################################## ")
grid_search_lr = GridSearchCV(lr_pipe, param_grid, cv=kf)
grid_search_lr.fit(dataset, y)
print(grid_search_lr.best_params_)
print('The CV best score:', grid_search_lr.best_score_)
print(f'The train set score: {grid_search_lr.score(dataset, y)} ')
print()
print(
    "########################################## NAIVE BAYES STATLOG_AUSTRALIAN DATA ########################################## ")
grid_search_nb = GridSearchCV(nb_pipe, nb_param_grid, cv=kf)
grid_search_nb.fit(dataset, y)
print(grid_search_nb.best_params_)
print('The CV best score:', grid_search_nb.best_score_)
print(f'The train set score: {grid_search_nb.score(dataset, y)} ')
