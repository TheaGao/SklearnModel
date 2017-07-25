import os

import numpy
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
from preprocessData import getDataXY
from preprocessData import get_accuracy

retrain = 1
model_path = "modelDNN.json"
model_weights = "modelDNN.h5"
seed = 1007
numpy.random.seed(seed)
# load dataset
trainX, trainY, testX, testY, validX, validY = getDataXY()

print len(trainY), len(trainX), len(testX), len(testX), len(validX), len(validY)
# standScaler = StandardScaler()
# standScaler.fit(trainX)
# trainX = standScaler.transform(trainX)
# testX = standScaler.transform(testX)
# validX = standScaler.transform(validX)
# split into input (X) and output (Y) variables
X = testX[:10000]
Y = testY[:10000]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(300, input_dim=13, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if os.path.isfile(model_path) and not retrain == 1:
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights)
    print("Loaded model from disk")
else:
    model = create_baseline()
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    model.fit(X, encoded_Y, batch_size=1000, validation_split=0.2, callbacks=callbacks, )
    # save model

    # serialize model to JSON

    if not os.path.isfile(model_path) or retrain == 1:
        model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights)
    print("Saved model to disk")
    # later...

    predictY = model.predict_classes(validX)
    predictY_class = model.predict_classes(validX)
    predictY_proba = model.predict_proba(validX)
    print 'predict:', predictY[:10]
    print  'class:', predictY_class[:10]
    print 'proba:', predictY_proba[:10]
    print 'type preY:', type(predictY[0])
    encoded_Y_valid = encoder.transform(validY)
    print 'encoded_y:', encoded_Y_valid[:10]
    print get_accuracy(encoded_Y_valid, predictY_class)


    # # evaluate model with standardized dataset
    # estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=500, verbose=0)
    # kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    # print results
    # print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
