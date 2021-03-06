import os
from time import time

from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from preprocessData import getDataXY

trainX, trainY, testX, testY, validX, validY = getDataXY()
print len(trainX), len(trainY), len(testX), len(testY), len(validX), len(validY)

retrain = 1
X = trainX
y = trainY
# clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=800)
# clf = tree.DecisionTreeClassifier()
# clf = NearestCentroid(metric='manhattan')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(13, 5), random_state=1,
                    verbose=True, max_iter=100, early_stopping=True)

if not os.path.isfile('NNP.pkl') or retrain == 1:
    clf.fit(X, y)
else:
    clf = joblib.load('NNP.pkl')

if not os.path.isfile('NNP.pkl') or retrain == 1:
    joblib.dump(clf, 'NNP.pkl')

# print clf.predict(testX[0:10])
# print testY[0:10]

valid_result = clf.predict(validX)


def get_accuracy(predict, true):
    right_num = 0
    total_num = 0
    for pre, tru in zip(predict, true):
        total_num += 1
        if pre == tru:
            right_num += 1
    print right_num, total_num
    return right_num / 0.1 / total_num


start = time()
print get_accuracy(clf.predict(validX), validY)
end = time()
print end - start, 's'
