import numpy as np
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Data path to diabetic_retinopathy
data, meta = arff.loadarff(r'data/diabetic_retinopathy/messidor_features.arff')
target = list()
X = np.zeros((19,), dtype='float32')

for e in data:
    a = e.tolist()
    target.append(a[-1])
    arr = np.asarray(a[:-1], dtype='float32')
    X = np.r_['0,2', X, arr]
    #print(type(arr), arr.shape, arr)
else:
    y = np.array(target)  # dtype='int8'
    X = np.delete(X, 0, 0)

kf = KFold(n_splits=5, shuffle=True)
train_agg = 0
test_agg = 0
for train_index, test_index in kf.split(X):
   # print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   lrc = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
   lrc.fit(X_train, y_train)

   train_pred = lrc.predict(X_train)
   test_pred = lrc.predict(X_test)
   train_agg += np.around(np.mean(train_pred == y_train), 2) * 100
   test_agg += np.around(np.mean(test_pred == y_test), 2) * 100
else:
    print('Training Avg: ', train_agg/5, '%')
    print('Testing Avg: ', test_agg/5, '%')

# print(y)
# lrc = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
# lrc.fit(X, y)
#
# train_pred = lrc.predict(X)
# print(train_pred)
# print('Training Acc: ',np.around(np.mean(train_pred == y)*100,2), '%')