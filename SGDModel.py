import os
from time import time
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from preprocessData import getDataXY

trainX, trainY, testX, testY, validX, validY = getDataXY()
# print len(trainX), len(trainY), len(testX), len(testY), len(validX), len(validY)

X = trainX
y = trainY
clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=800)

if not os.path.isfile('sgd0.54.pkl'):
    clf.fit(X, y)
else:
    clf = joblib.load('sgd0.54.pkl')

if not os.path.isfile('sgd0.54.pkl'):
    joblib.dump(clf, 'sgd0.54.pkl')

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
    return right_num / 0.1 / total_num

start = time()
print get_accuracy(clf.predict(validX), validY)
end = time()
print end - start, 's'
