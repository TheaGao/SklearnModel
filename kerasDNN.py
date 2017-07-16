import os

import numpy
from dataprocessing import get_data
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearnModel import get_accuracy

trainX, trainY, testX, testY, validX, validY = get_data()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

trainX = testX[:10000]
trainY = testY[:10000]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(trainY)
encoded_Y = encoder.transform(trainY)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=13, init='normal', activation='relu'))
    model.add(Dropout(0.0075))
    model.add(Dense(120, init='normal', activation='relu'))
    model.add(Dropout(0.0075))
    model.add(Dense(90, init='normal', activation='relu'))
    model.add(Dropout(0.0075))
    model.add(Dense(60, init='normal', activation='relu'))
    model.add(Dropout(0.0075))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dropout(0.0075))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if os.path.isfile('kerasDNN.json'):
    json_file = open('kerasDNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('kerasDNN.h5')
    print("Saved model to disk")
else:
    model = create_baseline()
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    model.fit(trainX, encoded_Y, batch_size=1000, validation_split=0.2, callbacks=callbacks)

if not os.path.isfile('kerasDNN.json'):
    model_json = model.to_json()
    with open("kerasDNN.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('kerasDNN.h5')

predictY = model.predict_classes(validX)
encoder_valid_Y = encoder.transform(validY)
percentDNN = get_accuracy(validX, encoder_valid_Y, predictY)

print percentDNN
