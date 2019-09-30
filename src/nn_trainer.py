import numpy as np
import tensorflow as tf
import random as rn

from keras import backend as K

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
import models

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store', default=None,
                    dest='train',
                    help='choose training sequence file',required=True)
parser.add_argument('-val', action='store', default=None,
                    dest='val',
                    help='choose validation sequence file',required=True)
parser.add_argument('-model_file', action='store', default="modelfile",
                    dest='model_file',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call',required=True)
parser.add_argument('-model_params', action='store', 
                    dest='model_params',nargs='+',required=True, 
                    help='model parameters (first parameter = past memory used for prediction)', type=int)
parser.add_argument('-log_file', action='store',
                    dest='log_file', default = "log_file",
                    help='Log file')
parser.add_argument('-lr', action='store', type=float,
                    dest='lr', default = 1e-3,
                    help='learning rate for Adam')
parser.add_argument('-noise', action='store', type=float,
                    dest='noise', default = 0.0,
                    help='amount of noise added to X (Unif[-noise,noise])')
parser.add_argument('-epochs', action='store',
                    dest='num_epochs', type = int, default = 20,
                    help='number of epochs to train (if 0, just store initial model))')

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_data(file_path,time_steps):
        series = np.load(file_path)
        series = series.reshape(-1, 1)

        series = series.reshape(-1)

        data = strided_app(series, time_steps+1, 1)

        X = data[:, :-1]
        Y = data[:, -1:]
        
        return X,Y

        
def fit_model(X_train, Y_train, X_val, Y_val, nb_epoch, model):
        optim = keras.optimizers.Adam(lr=arguments.lr)
        model.compile(loss='mean_squared_error', optimizer=optim)
        if nb_epoch == 0:
            model.save(arguments.model_file)
            return
        checkpoint = ModelCheckpoint(arguments.model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=3, verbose=1)

        callbacks_list = [checkpoint, csv_logger, early_stopping]
        model.fit(X_train, Y_train, epochs=nb_epoch, verbose=1, shuffle=True, callbacks=callbacks_list, validation_data = (X_val,Y_val))
 
 
arguments = parser.parse_args()
print(arguments)

num_epochs=arguments.num_epochs
sequence_length=arguments.model_params[0]
X_train,Y_train = generate_data(arguments.train, sequence_length)
X_val,Y_val = generate_data(arguments.val, sequence_length)
X_train = X_train + np.random.uniform(-arguments.noise,arguments.noise,np.shape(X_train))
X_val = X_val + np.random.uniform(-arguments.noise,arguments.noise,np.shape(X_val))

# predict the diff rather than absolute value
Y_train = Y_train-np.reshape(X_train[:,-1],np.shape(Y_train))
Y_val = Y_val-np.reshape(X_val[:,-1],np.shape(Y_val))

model = getattr(models, arguments.model_name)(*arguments.model_params)
fit_model(X_train, Y_train, X_val, Y_val, num_epochs, model)
