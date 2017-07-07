import os
from time import time

import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from preprocessData import getDataXY

trainX, trainY, testX, testY, validX, validY = getDataXY()
print len(trainX), len(trainY), len(testX), len(testY), len(validX), len(validY)

X = trainX
y = trainY
# clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=800)
# clf = tree.DecisionTreeClassifier()
# clf = NearestCentroid(metric='manhattan')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(13, 5), random_state=1,
                    verbose=True, max_iter=100, early_stopping=True)

if not os.path.isfile('NNP0.61.pkl'):
    clf.fit(X, y)
else:
    clf = joblib.load('NNP0.61.pkl')

if not os.path.isfile('NNP0.61.pkl'):
    joblib.dump(clf, 'NNP0.61.pkl')

start = time()
valid_result = clf.predict(validX)
end = time()
print end - start
valid_accuracy = np.mean(valid_result.ravel() == validY.ravel()) * 100
print valid_accuracy

# def get_accuracy(predict, true):
#     right_num = 0
#     total_num = 0
#     for pre, tru in zip(predict, true):
#         total_num += 1
#         if pre == tru:
#             right_num += 1
#     print right_num, total_num
#     return right_num / 0.1 / total_num
#
#
# start = time()
# print get_accuracy(clf.predict(validX), validY)
# end = time()
# print end - start, 's'
