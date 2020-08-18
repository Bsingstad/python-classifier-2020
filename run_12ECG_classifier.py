#!/usr/bin/env python
import numpy as np, os, sys, joblib
import joblib
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat





def create_model(): 
    # define two sets of inputs
    inputA = keras.layers.Input(shape=(5000,12)) 
    inputB = keras.layers.Input(shape=(2,))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8,input_shape=(5000,12), padding='same')(inputA)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(24, activation='sigmoid')(gap_layer) #HUSK Å SETTE TIL 24
    
    mod1 = keras.Model(inputs=inputA, outputs=output_layer)
        #mod1 = keras.layers.add([mod1,mod1_shortcut])
        # the second branch opreates on the second input
    mod2 = keras.layers.Dense(100, activation="relu")(inputB) # 2 -> 100
    mod2 = keras.layers.Dense(50, activation="relu")(mod2) # Added this layer
    mod2 = keras.Model(inputs=inputB, outputs=mod2)
        # combine the output of the two branches
    combined = keras.layers.concatenate([mod1.output, mod2.output])
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        
    z = keras.layers.Dense(24, activation="sigmoid")(combined) #HUSK Å SETTE TIL 24

        # our model will accept the inputs of the two branches and
        # then output a single value
    model = keras.Model(inputs=[mod1.input, mod2.input], outputs=z)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5), tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation",
        name="AUC", multi_label=True, label_weights=None)])
    return model


def run_12ECG_classifier(data,header_data,loaded_model):
    
    #HUSK Å SETTE TIL UNCOMMENTE threshold

    threshold = np.array([0.21551216, 0.20299779, 0.0955278 , 0.17289791, 0.18090656, 0.2227711 , 0.16741777, 0.22866722, 
    0.27118915, 0.23771854, 0.0912293 , 0.09410764, 0.20950935, 0.34517996, 0.02659288, 0.23399662, 0.15980351, 0.16177394,
    0.20402484, 0.25333636, 0.25657814, 0.22106934, 0.45621441, 0.0743871])


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,5000,12)

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
 '426177001', '426627000', '426783006' ,'427084000' ,'427393009', '445118002',
 '47665007' ,'59118001', '59931005', '63593006', '698252002', '713426002']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    model = create_model()
    f_out='model.h5'
    filename = os.path.join(model_input,f_out)
    model.load_weights(filename)

    return model
