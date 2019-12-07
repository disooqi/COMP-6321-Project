import os
import pandas as pd
import sklearn
from scipy.io import arff
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.svm import LinearSVC

#  data/regression/communities-and-crime-data-set/communities.names
#  data/regression/communities-and-crime-data-set/communities.data
data, meta = arff.loadarff(r'data/regression/communities-and-crime-data-set/communities.arff')
dataset = pd.DataFrame(data)

''' # Dropping non-predictive features
-- state: US state (by number) - not counted as predictive above, but if considered, should be considered nominal (nominal)
-- county: numeric code for county - not predictive, and many missing values (numeric)
-- community: numeric code for community - not predictive and many missing values (numeric)
-- communityname: community name - not predictive - for information only (string)
-- fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests - not predictive (numeric)
'''
dataset.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1, inplace=True)
y = dataset.pop('ViolentCrimesPerPop').values
dataset.drop(['LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
              'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
              'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits',
              'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
              'LemasGangUnitDeploy', 'PolicBudgPerPop'], axis=1, inplace=True)

## Search for
# what should happen when most of the data in column is missing? [options: new value not in the column, drop the column]
# what do we do in case of many duplicates in the data? [drop, keep]
'''
OtherPerCap  1
LemasSwornFT  1675
LemasSwFTPerPop  1675
LemasSwFTFieldOps  1675
LemasSwFTFieldPerPop  1675
LemasTotalReq  1675
LemasTotReqPerPop  1675
PolicReqPerOffic  1675
PolicPerPop  1675
RacialMatchCommPol  1675
PctPolicWhite  1675
PctPolicBlack  1675
PctPolicHisp  1675
PctPolicAsian  1675
PctPolicMinor  1675
OfficAssgnDrugUnits  1675
NumKindsDrugsSeiz  1675
PolicAveOTWorked  1675
PolicCars  1675
PolicOperBudg  1675
LemasPctPolicOnPatr  1675
LemasGangUnitDeploy  1675
PolicBudgPerPop  1675
'''
cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value=-99))  # This is for training
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))  # This is for testing
oe_step = ('le', OrdinalEncoder())
num_si_step = ('si', SimpleImputer(strategy='constant'))
sc_step = ('sc', StandardScaler())

cat_pipe = Pipeline([cat_si_step, ohe_step])
num_pipe = Pipeline([num_si_step, sc_step])
bin_pipe = Pipeline([oe_step])

transformers = [
    ('cat', cat_pipe, []),
    ('num', num_pipe, list(dataset.columns)),
    ('bin', bin_pipe, []),
]
ct = ColumnTransformer(transformers=transformers)
# X_transformed = ct.fit_transform(dataset)
# print(X_transformed.shape)

linear_r_pipe = Pipeline([
    ('X_transform', ct),
    ('linear_r', LinearRegression(fit_intercept=True)),
])

kf = KFold(n_splits=5, shuffle=True)
# cv_score = cross_val_score(ml_pipe, dataset, y, cv=kf).mean()

linear_r_param_grid = {
    'X_transform__num__si__strategy': ['mean', 'median'],
    # 'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    # 'mlp__alpha': [1e-1, 1e-3, 1e-5],
    # 'mlp__hidden_layer_sizes': [(10,), (20,), (5, 2), (4, 3), (4, 4)],
    # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
}
k1 = 1 ** 2 * RBF(length_scale=100) + WhiteKernel(noise_level=1)
# k2 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
#     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
# k3 = DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)
# k4 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp_pipe = Pipeline([
    ('X_transform', ct),
    ('gp', GaussianProcessRegressor(kernel=k1)),
])

gp_param_grid = {
    'X_transform__num__si__strategy': ['median'],
    # 'gp__alpha': [1e-01, 1e-03, 1e-05],
    # 'gp__kernel': [k2]
    # 'rf__n_estimators': range(1, 20),
    # 'rf__criterion': ['mse', 'mae'],
}

scoring = 'r2'
gs_linear_r = GridSearchCV(linear_r_pipe, linear_r_param_grid, cv=kf, scoring=scoring)
gs_linear_r.fit(dataset, y)
print("##################### Linear Regression for Sgemm Dataset ############################")
print('CV Score', gs_linear_r.best_score_)
print('Training accuracy', gs_linear_r.score(dataset, y))
print("##################### GaussianProcessRegressor for Communities Dataset #####################")
gs_gp = GridSearchCV(gp_pipe, gp_param_grid, cv=kf, scoring=scoring)
gs_gp.fit(dataset, y)
print('CV Score', gs_gp.best_score_)
print('Training accuracy', gs_gp.score(dataset, y))

# file_name = os.path.splitext(os.path.basename(__file__))[0]
# with open(f'{file_name}-result.txt', mode='a', encoding='utf-8') as handler:
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LINEAR REGRESSION FOR STUDENT_PERFORMANCE DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_linear_r.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_linear_r.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_linear_r.best_params_}\n')
#     handler.write(f'{gs_linear_r.best_estimator_}\n')
#     handler.write(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GAUSSIAN PROCESS FOR STUDENT_PERFORMANCE DATA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
#     handler.write(f'The CV best {scoring} score: {gs_gp.best_score_:.4f}\n')
#     handler.write(f'The train set {scoring} score: {gs_gp.score(dataset, y):.4f}\n')
#     handler.write(f'{gs_gp.best_params_}\n')
#     handler.write(f'{gs_gp.best_estimator_}\n')
#     handler.write('#' * 120)
#     handler.write('\n')
