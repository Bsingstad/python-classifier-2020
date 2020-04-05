#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
<<<<<<< Updated upstream
=======
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from scipy.io import loadmat
#import numpy as np, os, sys
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Bidirectional, InputLayer, Flatten
>>>>>>> Stashed changes

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)

<<<<<<< Updated upstream
    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='finalized_model.sav'
    loaded_model = joblib.load(filename)
=======
def load_12ECG_model():
    # load the model from disk
   # model = Sequential()
    #model.add(LSTM(32, input_shape=(1, 12)))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(9, activation='softmax'))

    model = Sequential()
    model.add(InputLayer(input_shape=(1, 12)))
    model.add(Bidirectional(LSTM(100, activation='tanh', return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100,activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(500)))
    model.add(Dropout(0.2))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(9, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(9, activation='sigmoid'))

    #model.load_weights("LSTM_physionet_comp2020.h5")
    model.load_weights("bidir_LSTM_physionet_comp2020.h5")

    #model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['acc'])
    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['categorical_accuracy', 'categorical_crossentropy'])
>>>>>>> Stashed changes

    return loaded_model
