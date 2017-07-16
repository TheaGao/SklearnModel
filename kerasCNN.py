import numpy
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from preprocessData import getDataXY


def preDataStep(dataX, dataY):
    finalX = []
    finalY = []
    step = 30
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
    finalX = numpy.reshape(finalX, (-1, 1, step, 13))

    return finalX, finalY


def build_model():
    print 'create model'
    model = Sequential()
    nb_filter_1, nb_row_1, nb_col_1, drop_rate = 13, 1, 4, 0.2
    #1
    model.add(Conv2D(nb_filter_1, nb_row_1, nb_col_1, input_shape=(1, 30, 13), activation='relu'))
    model.add(Dropout(drop_rate))
    #2
    model.add(MaxPooling2D(pool_size=(1, 2)))
    # shapes: [?,1,(2582+1-nb_col_1+1)/2,nb_filter_1]
    model.add(Dropout(drop_rate))
    #3
    model.add(Conv2D(63, 1, 4, activation='relu'))
    model.add(Dropout(drop_rate))
    #4
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(drop_rate))
    #5
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(drop_rate))
    #6
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(1, init='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])  # accuracy mae fmeasure precision recall
    return model


def main():
    BATCH_SIZE = 100

    EPOCH = 3
    # load dataset
    trainX, trainY, testX, testY, validX, validY = getDataXY()
    model = build_model()
    trainX, trainY = preDataStep(trainX, trainY)
    validX, validY = preDataStep(validX, validY)
    testX, testY = preDataStep(testX, testY)

    # train model ...
    print "train..."
    earlyStopping = EarlyStopping(monitor='val_acc', patience=2)
    print numpy.shape(testX)

    model.fit(trainX, trainY, verbose=2, shuffle=True, callbacks=[earlyStopping], nb_epoch=EPOCH,
              validation_data=(validX, validY), batch_size=BATCH_SIZE)
    loss_and_metrics = model.evaluate(testX, testY, verbose=2, batch_size=BATCH_SIZE)
    print '==============='
    print 'loss_metrics: ', loss_and_metrics

    print numpy.shape(trainX)

    return 0


if __name__ == '__main__':
    main()
