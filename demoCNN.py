import os
import time

import h5py
import joblib
import numpy
import numpy as np
from baseZhang import save_model, save_model_weights, load_model, load_model_weights, if_no_create_it
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

SEED = 1007
np.random.seed(SEED)
LOAD_MODEL_FLAG = False
BATCH_SIZE = 32
EPOCH = 1000
# sub_combine_list=['13glfcc', '13mfcc_zc', '12lpcc_czg', '13rastaplp', '12chroma']
sub_combine_list = ['13glfcc', '13mfcc_zc', '12lpcc_czg', '13rastaplp', '12chroma']


def get_dataset_X_Y_encoder(dataset='dataset.h5', dataX='X', dataY='Y', encoder='None'):
    h5file = h5py.File(dataset, 'r')
    X = h5file[dataX][:]
    Y = h5file[dataY][:]
    h5file.close()
    X = numpy.reshape(X, (-1, 1, 2582, 63))
    Y_final = []
    for item_y in range(0, len(Y), 2582):
        Y_final.append(Y[item_y])
    Y = Y_final
    if encoder == 'None':
        encoder = LabelEncoder()
        encoder.fit(Y)
    encoder_Y = encoder.transform(Y)
    # convert integers to variables (i.e. one hot encoded)
    one_hot_Y = np_utils.to_categorical(encoder_Y)
    return X, one_hot_Y, encoder


# INPUT_DIM = 13  # 13 12 25 37
def load_train_test_data(onDataset, featureX='X', targetY='Y'):
    X, one_hot_Y, encoder = get_dataset_X_Y_encoder(onDataset, featureX, targetY)
    joblib.dump(encoder, 'encoder.joblib')
    X_train, Y_train = X, one_hot_Y
    X, one_hot_Y, encoder = get_dataset_X_Y_encoder(onDataset.replace('train', 'test'), featureX, targetY, encoder)
    X_test, Y_test, X_val, Y_val = X, one_hot_Y, X, one_hot_Y
    return X_train, X_test, Y_train, Y_test, X_val, Y_val


def build_cnn_model(onDataset='dataset.h5', featureX='X', targetY='Y', sub_combine_list=[]):
    onDataset_name = onDataset.split('/')[-1].split('.')[0]
    X_train, X_test, Y_train, Y_test, X_val, Y_val = load_train_test_data(onDataset, featureX, targetY)
    print 'len Xtrain=================>>>>>>>>>>>>>>>>>>', len(X_train)
    model_save_path = 'models/cnn_' + onDataset_name + '_model.json'
    model_weights_save_path = model_save_path.replace('.json', '.h5')
    print 'weight:', model_weights_save_path
    if LOAD_MODEL_FLAG and os.path.isfile(model_save_path) and os.path.isfile(model_weights_save_path):
        print model_save_path
        model = load_model(model_save_path)
        model = load_model_weights(model_weights_save_path, model)
    else:
        print 'create model'
        model = Sequential()
        #parm
        # nb_filter
        # nb_row
        # nb_col
        nb_filter_1, nb_row_1, nb_col_1, drop_rate = 63, 1, 10, 0.2
        model.add(Conv2D(nb_filter_1, nb_row_1, nb_col_1, input_shape=(1, 2582, 63), activation='relu'))
        # shapes: [?,nb_row_1,2582+1-nb_col_1,nb_filter_1]
        model.add(Dropout(drop_rate))
        model.add(MaxPooling2D(pool_size=(1, 20)))
        # shapes: [?,1,(2582+1-nb_col_1+1)/2,nb_filter_1]
        model.add(Dropout(drop_rate))
        model.add(Conv2D(63, 1, 8, activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(MaxPooling2D(pool_size=(1, 20)))
        model.add(Dropout(drop_rate))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(drop_rate))


        model.add(Dense(20, init='normal', activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])  # accuracy mae fmeasure precision recall
    # train model ...
    print "train..."
    earlyStopping = EarlyStopping(monitor='val_acc',patience=100)
    print numpy.shape(X_test)
    model.fit(X_train, Y_train, verbose=2, shuffle=True, callbacks=[earlyStopping], nb_epoch=EPOCH,
              validation_data=(X_val, Y_val), batch_size=BATCH_SIZE)
    if_no_create_it(model_save_path)
    save_model(model, model_save_path)
    if_no_create_it(model_weights_save_path)
    save_model_weights(model, model_weights_save_path)
    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2, batch_size=BATCH_SIZE)
    print '==============='
    print 'loss_metrics: ', loss_and_metrics

    return loss_and_metrics[1]


def main():
    useDataset = '../../data/SID/dnn_artist20_combine_feature_train.h5'
    useFeature = 'combine_featureX'
    useTarget = useFeature.replace('X', 'Y')
    build_cnn_model(useDataset, useFeature, useTarget,
                    sub_combine_list)  # step 1. test on dnn model

    return 0


if __name__ == '__main__':
    print "start..."
    before = time.time()
    main()
    after = time.time()
    print 'takes time : %.0f(s)' % (after - before)