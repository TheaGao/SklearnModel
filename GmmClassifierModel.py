# coding=utf-8
import numpy as np
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

from preprocessData import getDataXY

trainX, trainY, testX, testY, validX, validY = getDataXY()

X_train = trainX
y_train = trainY
X_test = testX
y_test = testY
X_valid = validX
y_valid = validY

# 转换为0，1
trainY_num = np.arange(len(trainY))
for lab in range(len(trainY)):
    if 'nosing' == trainY[lab]:
        trainY_num[lab] = 0
    else:
        trainY_num[lab] = 1
testY_num = np.arange(len(testY))
for lab in range(len(testY)):
    if 'nosing' == testY[lab]:
        testY_num[lab] = 0
    else:
        testY_num[lab] = 1
validY_num = np.arange(len(validY))
for lab in range(len(validY)):
    if 'nosing' == validY[lab]:
        validY_num[lab] = 0
    else:
        validY_num[lab] = 1

y_train = trainY_num
y_test = testY_num
y_valid = validY_num

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                                    covariance_type=covar_type, init_params='wc', n_iter=2))
                   for covar_type in ['spherical'])

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100

    y_valid_pred = classifier.predict(X_valid)
    valid_accuracy = np.mean(y_valid_pred.ravel() == y_valid.ravel()) * 100


    print train_accuracy, test_accuracy,valid_accuracy
