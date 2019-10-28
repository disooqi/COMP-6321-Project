import numpy as np
from scipy.io import arff
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# 'C:\Users\disoo\Documents\COMP-6321-Project\data\diabetic_retinopathy\messidor_features.arff'
dataset, meta = arff.loadarff(r'..\data\diabetic_retinopathy\messidor_features.arff')

target = list()
X = None
for sample in dataset:
    sample_features = sample.tolist()
    target.append(sample_features[-1])
    arr = np.array(sample_features[:-1], dtype=np.float16)
    X = np.r_['0,2', X, arr] if X is not None else arr
else:
    y = np.array(target, dtype=np.int8)


'''
Multi-Layer Perceptron (MLP) 
'''
mlp_01 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 7), random_state=5)
pline = make_pipeline(StandardScaler(), mlp_01)
cv_scores = cross_val_score(pline, X, y, cv=5, scoring='accuracy')
print(f'Multi-Layer Perceptron (MLP) DEV Accuracy: {cv_scores.mean():0.2f} and the 95% confidence interval of the score estimate is  '
      f'{cv_scores.std()*2:0.2f}')
# print("Accuracy: " % (scores.mean(), scores.std() * 2))

scaled_X = StandardScaler().fit_transform(X)
mlp_02 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=5)
mlp_02.fit(scaled_X, y)
pred = mlp_02.predict(scaled_X)
print('Multi-Layer Perceptron (MLP) Training Acc: ', np.around(np.mean(pred == y), 2), " ... ", mlp_02.score(scaled_X, y))


'''
Nearest Neighbors
'''
n_neighbors = 70
neigh_01 = KNeighborsClassifier(n_neighbors=n_neighbors)
pline = make_pipeline(StandardScaler(), neigh_01)
cv_scores = cross_val_score(pline, X, y, cv=5, scoring='accuracy')
print(f'Nearest Neighbors Classifier DEV Accuracy: {cv_scores.mean():0.2f} and the 95% confidence interval of the score estimate is  '
      f'{cv_scores.std()*2:0.2f}')
# print("Accuracy: " % (scores.mean(), scores.std() * 2))

scaled_X = StandardScaler().fit_transform(X)
neigh_02 = KNeighborsClassifier(n_neighbors=n_neighbors)
neigh_02.fit(scaled_X, y)
pred = neigh_02.predict(scaled_X)
print('Nearest Neighbors Classifier Training Acc: ', np.around(np.mean(pred == y), 2), " ... ", neigh_02.score(scaled_X, y))

# kf = KFold(n_splits=5, shuffle=True)
# train_agg = 0
# test_agg = 0
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#
#     y_train, y_test = y[train_index], y[test_index]  #
#
#     lrc = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
#     lrc.fit(X_train, y_train)
#
#     train_pred = lrc.predict(X_train)
#     test_pred = lrc.predict(X_test)
#     train_agg += np.around(np.mean(train_pred == y_train), 2) * 100
#     test_agg += np.around(np.mean(test_pred == y_test), 2) * 100
# else:
#
#     print('Training Avg: ', train_agg / 5, '%')
#     print('Testing Avg: ', test_agg / 5, '%')
