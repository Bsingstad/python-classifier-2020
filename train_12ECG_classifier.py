#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
#from keras.preprocessing.sequence import pad_sequences
import numpy as np, os, sys, joblib
from scipy.io import loadmat
from get_12ECG_features import get_12ECG_features


def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')

    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    for ecgfilename in sorted(os.listdir(input_directory)):
        if ecgfilename.endswith(".mat"):
            data, header_data = load_challenge_data(input_directory+ecgfilename)
            labels.append(header_data[15][5:-1])
            ecg_filenames.append(ecgfilename)
            gender.append(header_data[14][6:-1])
            age.append(header_data[13][6:-1])
    
    # Gender processing - replace with nicer code later
    gender = np.asarray(gender)
    gender[np.where(gender == "Male")] = 0
    gender[np.where(gender == "male")] = 0
    gender[np.where(gender == "M")] = 0
    gender[np.where(gender == "Female")] = 1
    gender[np.where(gender == "female")] = 1
    gender[np.where(gender == "F")] = 1
    gender[np.where(gender == "NaN")] = 2
    gender = gender.astype(np.int)

    # Age processing - replace with nicer code later
    age = np.asarray(age)
    age[np.where(age == "NaN")] = -1
    age = age.astype(np.int)

    # Load SNOMED codes
    SNOMED_scored=pd.read_csv("SNOMED_mappings_scored.csv", sep=";")
    SNOMED_unscored=pd.read_csv("SNOMED_mappings_unscored.csv", sep=";")

    # Load labels to dataframe
    df_labels = pd.DataFrame(labels)

    # Remove unscored labels
    for i in range(len(SNOMED_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(SNOMED_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    # Replace overlaping SNOMED codes

    codes_to_replace=['713427006','284470004','427172004']
    replace_with = ['59118001','63593006','17338001']

    for i in range(len(codes_to_replace)):
        df_labels.replace(to_replace=codes_to_replace[i], inplace=True ,value=replace_with[i], regex=True)
    
    # One-Hot encode classes
    one_hot = MultiLabelBinarizer()
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    print(one_hot.classes_)
    print("classes: {}".format(y.shape[1]))

    # Train model.
    print('Training model...')

    model=create_model(y)
    batchsize = 30
    history = model.fit_generator(generator=batch_generator(batch_size=batchsize, gen_x=generate_X(input_directory), gen_y=generate_y(y), gen_z=generate_z(age,gender), ohe_labels = one_hot.classes_),steps_per_epoch=(len(y)/batchsize), epochs=3)

    # Save model.
    print('Saving model...')

    model.save_weights("model_weights.h5")

    #final_model={'model':model, 'imputer':imputer,'classes':classes}

    #filename = os.path.join(output_directory, 'finalized_model.sav')
    #joblib.dump(final_model, filename, protocol=0)

# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)

def create_model(y): 
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
    mod1 = keras.layers.Dense(y.shape[1], activation='sigmoid')(mod1)
    mod1 = keras.Model(inputs=inputA, outputs=mod1)

    # the second branch opreates on the second input
    mod2 = keras.layers.Dense(2, activation="relu")(inputB)
    mod2 = keras.Model(inputs=inputB, outputs=mod2)
    # combine the output of the two branches
    combined = keras.layers.concatenate([mod1.output, mod2.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = keras.layers.Dense(y.shape[1], activation="sigmoid")(combined)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.Model(inputs=[mod1.input, mod2.input], outputs=z)
    #@title Plot model for better visualization
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy','categorical_accuracy',"categorical_crossentropy"])
    return model

def generate_y(y):
    while True:
        for i in range(len(y)):
            y_train = y[i]
            yield y_train

def generate_X(input_directory):
    while True:
        for filen in sorted(os.listdir(input_directory)):
            if filen.endswith(".mat"):
                data, header_data = load_challenge_data(input_directory+filen)
                X_train_new = keras.preprocessing.sequence.pad_sequences(data, maxlen=10000, truncating='post',padding="post")
                X_train_new = X_train_new.reshape(10000,12)
                yield X_train_new

def generate_z(age, gender):
    while True:
        for i in range(len(age)):
            gen_age = age[i]
            gen_gender = gender[i]
            z_train = [gen_age , gen_gender]
            yield z_train

def batch_generator(batch_size, gen_x,gen_y, gen_z, ohe_labels): 
    batch_features = np.zeros((batch_size,10000, 12))
    batch_labels = np.zeros((batch_size,len(ohe_labels)))
    batch_demo_data = np.zeros((batch_size,2))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            batch_demo_data[i] = next(gen_z)

        X_combined = [batch_features, batch_demo_data]
        yield X_combined, batch_labels