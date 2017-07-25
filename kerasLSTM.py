import os

import numpy
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, model_from_json

# fix random seed for reproducibility
from preprocessData import getDataXY


# load dataset
trainX, trainY, testX, testY, validX, validY = getDataXY()


def preDataStep(dataX, dataY):
    finalX = []
    finalY = []
    step = 7
    for item in range(0, len(dataX), step):
        if step == list(dataY[item:item + step]).count('sing'):
            finalX.append(dataX[item:item + step])
            finalY.append(1)

        elif step == list(dataY[item:item + step]).count('nosing'):
            finalX.append(dataX[item:item + step])
            finalY.append(0)

    print 'nosing', finalY.count(0)
    print 'sing', finalY.count(1)

    finalX = numpy.array(finalX)
    finalY = numpy.array(finalY)

    return finalX, finalY


trainX, trainY = preDataStep(trainX[:100000], trainY[:100000])
validX, validY = preDataStep(validX, validY)
testX, testY = preDataStep(testX, testY)

# create the model
model_path = 'lstm.josn'
model_weights = 'lstm.h5'
retrain = 1

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
    numpy.random.seed(24)
    model = Sequential()
    model.add(LSTM(100, input_shape=(7, 13), return_sequences=True))
    model.add(LSTM(25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=64, nb_epoch=10,
              callbacks=callbacks)
    if not os.path.isfile(model_path) or retrain == 1:
        model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights)
    print("Saved model to disk")
# Final evaluation of the model
scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
