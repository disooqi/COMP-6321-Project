import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data, meta = arff.loadarff(r'..\data\thoracic_surgery_data\ThoraricSurgery.arff')
dataset = pd.DataFrame(data)

y = LabelEncoder().fit_transform(dataset.pop('Risk1Yr').values)
print(y)

si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
# In dev and test data, new values will raise an error due to handle_unknown='error'
# Set handle_unknown='ignore', will mask OOV and missing value problems in dev and test data
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='error'))
oe_step = ('le', OrdinalEncoder())

num_si_step = ('si', SimpleImputer(strategy='median'))
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
X_transformed = ct.fit_transform(dataset)
print(X_transformed.shape)

