import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

dataset = pd.read_csv(r'data/breast-cancer-wisconsin/wdbc.data', header=None)  # header=None, usecols=[3,6]

# print(dataset[1].value_counts())

dataset.pop(0)
y = LabelEncoder().fit_transform(dataset.pop(1).values)

si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='mean'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    # ('cat', cat_pipe, ['DGN', 'PRE6', 'PRE14']),
    ('num', num_pipe,
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
      28, 29]),
    # ('bin', bin_pipe, ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)

lr_pipe = Pipeline([
    ('X_transform', ct),
    ('lr', LogisticRegression(penalty='l1', solver='liblinear', max_iter=800)),
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
    ('lr', GaussianNB(priors=None, var_smoothing=1e-09)),
])

nb_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],

}
print(
    "########################################## LOGISTIC REGRESSION ON BREAST_CANCER DATA ########################################## ")
grid_search_lr = GridSearchCV(lr_pipe, param_grid, cv=kf)
grid_search_lr.fit(dataset, y)
print(grid_search_lr.best_params_)
print('The CV best score:', grid_search_lr.best_score_)
print(f'The train set score: {grid_search_lr.score(dataset, y)} ')
print()
print(
    "########################################## NAIVE BAYES BREAST_CANCER DATA ########################################## ")
grid_search_nb = GridSearchCV(nb_pipe, nb_param_grid, cv=kf)
grid_search_nb.fit(dataset, y)
print(grid_search_nb.best_params_)
print('The CV best score:', grid_search_nb.best_score_)
print(f'The train set score: {grid_search_nb.score(dataset, y)} ')
