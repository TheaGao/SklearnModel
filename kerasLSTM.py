from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearnModel import get_accuracy
from dataprocessing import get_data
import numpy
from keras.datasets import imdb
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os


numpy.random.seed(24)
trainX,trainY,testX,testY,validX,validY=get_data()

def process_data_LSTM(X,Y):
    finalX=[]
    finalY=[]
    for i in range(0,len(X),7):
        if list(Y[i:i+7]).count('sing')==7:
            finalX.append(X[i:i+7])
            finalY.append(1)
        elif list(Y[i:i+7]).count('nosing')==7:
            finalX.append(X[i:i+7])
            finalY.append(0)
            print i,'second'
        else:
            pass
    finalX = numpy.array(finalX)
    finalY = numpy.array(finalY)
    return finalX,finalY


trainX,trainY=process_data_LSTM(trainX[:100000],trainY[:100000])
validX,validY=process_data_LSTM(validX[:10000],validY[:10000])
testX,testY=process_data_LSTM(testX[:10000],testY[:10000])
print trainY


def create_model():
    model = Sequential()
    model.add(LSTM(100,input_shape=(7,13)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
if os.path.isfile('kerasLSTM.json'):
    json_file=open('kerasLSTM.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)
    model.load_weights('kerasLSTM.h5')
    print("Saved model to disk")
else:
    model=create_model()    
    model.fit(trainX,trainY,nb_epoch=10,batch_size=32)

if not os.path.isfile('kerasLSTM.json'):
    model_json = model.to_json()
    with open("kerasLSTM.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('kerasLSTM.h5')



predict_Y=model.predict(validX)
scores = model.evaluate(validX, validY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
