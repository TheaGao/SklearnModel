import os
import numpy as np
from baseZhang import class_encoder_to_number
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

from preprocessData import getDataXY

trainX, trainY, testX, testY, validX, validY = getDataXY()

encoder_path = 'encoder.pkl'
if not os.path.isfile(encoder_path):
    encoder = class_encoder_to_number(trainY)
    joblib.dump(encoder, 'encoder.pkl')
else:
    encoder = joblib.load(encoder_path)

trainY = encoder.transform(trainY)
testY = encoder.transform(testY)
validY = encoder.transform(validY)

n_classes = len(np.unique(trainY))

# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                                             covariance_type=cov_type, max_iter=200, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([trainX[trainY == i].mean(axis=0)
                                     for i in range(n_classes)])
    model_path = name + '.pkl'

    if not os.path.isfile(model_path):
        # Train the other parameters using the EM algorithm.
        estimator.fit(trainX)
        joblib.dump(estimator, model_path)
    else:
        estimator = joblib.load(model_path)

    y_train_pred = estimator.predict(trainX)
    print name
    train_accuracy = np.mean(y_train_pred.ravel() == trainY.ravel()) * 100
    print 'Train accuracy: %.1f' % train_accuracy
    y_valid_pred = estimator.predict(validX)
    valid_accuracy = np.mean(y_valid_pred.ravel() == validY.ravel()) * 100
    print  'Test accuracy: %.1f' % valid_accuracy
