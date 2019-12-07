import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

#  data/default_of_credit_card_clients/default_of_credit_card_clients.xls
dataset = pd.read_excel(r'..\..\data\default_of_credit_card_clients\default_of_credit_card_clients.xls', skiprows=1)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# C:\Users\disoo\Documents\COMP-6321-Project\data\cifar-10-batches-py\data_batch_1
dc = unpickle(r'C:\Users\disoo\Documents\COMP-6321-Project\data\cifar-10-batches-py\data_batch_1')

# dataset.pop('ID')
# y = LabelEncoder().fit_transform(dataset.pop('default payment next month').values)
#
# cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
# ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
# oe_step = ('le', OrdinalEncoder())
# num_si_step = ('si', SimpleImputer(strategy='median'))
# sc_step = ('sc', StandardScaler())
#
# cat_pipe = Pipeline([cat_si_step, ohe_step])
# num_pipe = Pipeline([num_si_step, sc_step])
# bin_pipe = Pipeline([oe_step])
#
# transformers = [
#     ('cat', cat_pipe, ['EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']),
#     ('num', num_pipe, ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
#                        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']),
#     ('bin', bin_pipe, ['SEX']),
# ]
# ct = ColumnTransformer(transformers=transformers)
# # X_transformed = ct.fit_transform(dataset)
# # print(X_transformed)
#
# ml_pipe = Pipeline([
#     ('X_transform', ct),
#     ('mlp', DecisionTreeClassifier()),
# ])
#
# kf = KFold(n_splits=5, shuffle=True)
# # cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()
#
# param_grid = {
#     'X_transform__num__si__strategy': ['mean', 'median'],
#     'mlp__solver': ['sgd', 'adam', 'lbfgs'],
#     'mlp__alpha': [1e-1, 1e-3, 1e-5],
#     'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
#     'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
# }
#
# gs = GridSearchCV(ml_pipe, param_grid, cv=kf, scoring='accuracy')
# gs.fit(dataset, y)
# print(gs.best_params_)
# print('The CV best score:', gs.best_score_)
# # print(pd.DataFrame(gs.cv_results_))
#
# print(f'The train set score: {gs.score(dataset, y)} ')

