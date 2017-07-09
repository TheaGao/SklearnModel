import os

import numpy as np
from baseZhang import class_encoder_to_number
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

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

dt_model_path = 'Models/dt0.58.pkl'
dt_model = joblib.load(dt_model_path)

nb_model_path = 'Models/NB0.59.pkl'
nb_model = joblib.load(nb_model_path)

nc_model_path = 'Models/NC0.57.pkl'
nc_model = joblib.load(nc_model_path)

nnp_model_path = 'Models/NNP0.61.pkl'
nnp_model = joblib.load(nnp_model_path)

sgd_model_path = 'Models/sgd0.54.pkl'
sgd_model = joblib.load(sgd_model_path)

voting_clf = VotingClassifier(estimators=[
    ('dt', dt_model), ('nb', nb_model),
    ('nc', nc_model), ('nnp', nnp_model), ('sgd', sgd_model)], voting='hard')
#
# encoder_path = 'voting.pkl'
# if not os.path.isfile(encoder_path):
voting_clf = voting_clf.fit(trainX, trainY)
#     joblib.dump(encoder, 'voting.pkl')
# else:
#     voting_clf = joblib.load(encoder_path)

voting_test_result = voting_clf.predict(testX)
voting_valid_result = voting_clf.predict(validX)

test_accuracy = np.mean(voting_test_result.ravel() == testY.ravel()) * 100
valid_accuracy = np.mean(voting_valid_result.ravel() == validY.ravel()) * 100

print test_accuracy, valid_accuracy
