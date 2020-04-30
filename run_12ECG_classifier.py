#!/usr/bin/env python

import numpy as np
#import joblib
from get_12ECG_features import get_12ECG_features
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam, SGD
from scipy.io import loadmat
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Bidirectional, InputLayer, Flatten, Conv1D, MaxPooling1D, BatchNormalization



def run_12ECG_classifier(data,header_data,classes,model):
    # optimized thresholds:
    threshold = [0.48336382, 0.48216414, 0.20476912, 0.51797735, 0.35201463,0.35771265, 0.61279034, 0.42107908, 0.22069441]


    # Use your classifier here to obtain a label and score for each class. 
    data12lead = data
    testdata = pad_sequences(data12lead, maxlen=10000, truncating='post',padding="post")
    reshaped12lead = testdata.reshape(1,10000,12)
    score = model.predict_proba(reshaped12lead)
    score = score.ravel()
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1

    return binary_prediction, score


def load_12ECG_model():
    # load the model from disk


    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu',input_shape=(10000,12),use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride=3,data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride = 3, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride = 3, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride = 3, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride = 3, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=5, stride = 3, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True))
    model.add(MaxPooling1D(pool_size=3, stride = 1, data_format="channels_last" ))
    model.add(BatchNormalization())
    model.add(LSTM(512))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='sigmoid'))


    model.load_weights("weights_512_23apr_1725.hdf5")

    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['categorical_accuracy', 'categorical_crossentropy'])


    return model
