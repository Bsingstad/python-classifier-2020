#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing.sequence import pad_sequences
import numpy as np, os, sys, joblib
from scipy.io import loadmat
from get_12ECG_features import get_12ECG_features


def create_model(): 
    # define two sets of inputs
    inputA = keras.layers.Input(shape=(10000,12)) 
    inputB = keras.layers.Input(shape=(2,))
    # the first branch operates on the first input
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",input_shape=(10000,12),use_bias=True)(inputA)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=5, strides=3,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Conv1D(filters=512, kernel_size=5, activation="relu",use_bias=True)(mod1)
    mod1 = keras.layers.MaxPool1D(pool_size=3, strides=1,data_format="channels_last")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.LSTM(512, activation="hard_sigmoid")(mod1)
    mod1 = keras.layers.Dense(512, activation="softsign")(mod1)
    mod1 = keras.layers.Dropout(0)(mod1)
    mod1 = keras.layers.Dense(25, activation='sigmoid')(mod1)
    mod1 = keras.Model(inputs=inputA, outputs=mod1)

    # the second branch opreates on the second input
    mod2 = keras.layers.Dense(2, activation="relu")(inputB)
    mod2 = keras.Model(inputs=inputB, outputs=mod2)
    # combine the output of the two branches
    combined = keras.layers.concatenate([mod1.output, mod2.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = keras.layers.Dense(25, activation="sigmoid")(combined)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.Model(inputs=[mod1.input, mod2.input], outputs=z)
    #@title Plot model for better visualization
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy','categorical_accuracy',"categorical_crossentropy"])
    return model


def run_12ECG_classifier(data,header_data,loaded_model):

    threshold = [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004,
     0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]
    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=10000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,10000,12)

    gender = header_data[14][6:-1]
    age=header_data[13][6:-1]
    if gender == "Male":
        gender = 0
    elif gender == "male":
        gender = 0
    elif gender =="M":
        gender = 0
    elif gender == "Female":
        gender = 1
    elif gender == "female":
        gender = 1
    elif gender == "F":
        gender = 1
    elif gender =="NaN":
        gender = 2

    # Age processing - replace with nicer code later
    if age == "NaN":
        age = -1
    else:
        age = int(age)

    demo_data = np.asarray([age,gender])
    reshaped_demo_data = demo_data.reshape(1,2)

    combined_data = [reshaped_signal,reshaped_demo_data]


    score  = model.predict(combined_data)[0]
    
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1
    classes = ['10370003', '111975006', '164889003', '164890007', '164909002', '164917005',
 '164934002', '164947007', '17338001', '251146004', '270492004', '39732003',
 '426177001', '426627000', '426783006', '427084000', '427393009', '445118002',
 '47665007', '59118001', '59931005', '63593006', '698252002', '713426002',
 'undefined class']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    loaded_model=create_model()
    loaded_model.load_weights("model.h5")

    return loaded_model
