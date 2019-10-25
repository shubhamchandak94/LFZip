from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, GaussianNoise, GRU, Reshape
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, LSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
from keras import backend as K

# models for floating point data compression

# note that the first parameter to any function should be the input_dim

def FC(input_dim,num_hidden_layers,hidden_layer_size):
    if num_hidden_layers == 0:
        model = Sequential()
        model.add(Dense(1,input_dim=input_dim))
        return model
    assert num_hidden_layers > 0
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation='relu',input_dim=input_dim))
    model.add(BatchNormalization())
    for i in range(num_hidden_layers-1):
        model.add(Dense(hidden_layer_size, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(1))
    return model

def biGRU(input_dim, num_biGRU_layers, num_units_biGRU, num_units_dense):
        assert num_biGRU_layers > 0
        model = Sequential()
        model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
        for i in range(num_biGRU_layers-1):
            model.add(Bidirectional(GRU(num_units_biGRU, return_sequences=True)))
        model.add(Bidirectional(GRU(num_units_biGRU, return_sequences=False)))
        model.add(Dense(num_units_dense, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model
