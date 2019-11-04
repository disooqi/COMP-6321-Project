import pandas as pd
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

dataset = pd.read_csv(r'data/yeast/yeast.data', header=None, delimiter=' ', skipinitialspace=True,
                      names=['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl',
                             'pox', 'vac', 'nuc', 'target'])

##############################
#### Remove Dublicates #######
##############################
dataset.pop('Sequence Name')
y = LabelEncoder().fit_transform(dataset.pop('target').values)

# print(dataset.dtypes.head(20))
# print(np.array([dt.kind for dt in dataset.dtypes]))


num_si_step = ('si', SimpleImputer(strategy='median'))
sc_step = ('sc', StandardScaler())

num_pipe = Pipeline([num_si_step, sc_step])

transformers = [
    ('num', num_pipe, ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']),
]

ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)

lr_pipe = Pipeline([
    ('X_transform', ct),
    ('lr', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1500, multi_class='ovr')),
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
    "########################################## LOGISTIC REGRESSION ON YEAST DATA ########################################## ")
grid_search_lr = GridSearchCV(lr_pipe, param_grid, cv=kf)
grid_search_lr.fit(dataset, y)
print(grid_search_lr.best_params_)
print('The CV best score:', grid_search_lr.best_score_)
print(f'The train set score: {grid_search_lr.score(dataset, y)} ')
print()
print("########################################## NAIVE BAYES YEAST DATA ########################################## ")
grid_search_nb = GridSearchCV(nb_pipe, nb_param_grid, cv=kf)
grid_search_nb.fit(dataset, y)
print(grid_search_nb.best_params_)
print('The CV best score:', grid_search_nb.best_score_)
print(f'The train set score: {grid_search_nb.score(dataset, y)} ')
