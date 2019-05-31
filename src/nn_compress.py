#
# Compression application using adaptive arithmetic coding
#
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
#
# Copyright (c) Project Nayuki
#
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#

from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(42)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import json
import struct
import models
import tempfile
import shutil
import subprocess
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-mode', action='store', dest='mode',
        help='c or d (compress/decompress)', required=True)
parser.add_argument('-infile', action='store', dest='infile', help = 'infile .npy/.7z', type = str, required=True)
parser.add_argument('-outfile', action='store', dest='outfile', help = 'outfile .npy/.7z', type = str, required=True)
parser.add_argument('-maxerror', action='store', dest='maxerror', help = 'max allowed error for compression', type=float)
parser.add_argument('-model_file', action='store', dest='model_file',
                    help='model file', required=True)
parser.add_argument('-model_update_period', action='store', dest='model_update_period', help = 'train model (both during compression & decompression) after seeing these many symbols (default: never train)', type = int)
parser.add_argument('-lr', action='store', type=float,
	dest='lr', default = 1e-3,
	help='learning rate for Adam')
parser.add_argument('-epochs', action='store',
	dest='num_epochs', type = int, default = 1,
	help='number of epochs to train')

args = parser.parse_args()
from keras import backend as K

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate_data(series, time_steps):
    series = series.reshape(-1, 1)
    series = series.reshape(-1)
    data = strided_app(series, time_steps+1, 1)
    X = data[:, :-1]
    Y = data[:, -1:]
    return X,Y

model = load_model(args.model_file)
window_size = model.layers[0].input_shape[1]
print('window_size',window_size)
K.set_value(model.optimizer.lr, args.lr)

if args.mode == 'c':
    if args.maxerror == None:
        raise RuntimeError('maxerror not specified for mode c')
    tmpfile = args.outfile+'.tmp'
    reconfile = args.outfile+'.recon.npy'
    maxerror = np.float32(args.maxerror)
    # read file
    data = np.load(args.infile)
    data = np.array(data,dtype=np.float32)
    # initialize quantization (with roughly 65535 bins at the start)
    maxlevel = np.float32(65533*maxerror)
    minlevel = np.float32(-65533*maxerror)
    numbins = int((maxlevel-minlevel)/(2*maxerror))+2
    bins = np.linspace(minlevel,maxlevel,numbins,dtype=np.float32)
    fmtstring = 'H' # 16 bit unsigned
    bin_idx_len = 2 # in bytes

    reconstruction = np.zeros(np.shape(data),dtype=np.float32)
    f_out = open(tmpfile,'wb')
    # write max error to file (needed during decompression)
    f_out.write(struct.pack('f',maxerror))
    # write length of array to file
    f_out.write(struct.pack('I',len(data)))
    for i in tqdm(range(len(data))):
        if i <= window_size:
            predval = np.float32(0.0)
        else:
            if args.model_update_period != None:
                assert args.model_update_period > window_size+1
                if i%args.model_update_period == 0:
                    X_train, Y_train = generate_data(reconstruction[i-args.model_update_period:i-1], window_size)
                    # predict the diff rather than absolute value
                    Y_train = Y_train-np.reshape(X_train[:,-1],np.shape(Y_train))
                    model.fit(X_train, Y_train, epochs=args.num_epochs, verbose=0)
            predval = reconstruction[i-1] + np.float32(model.predict(np.reshape(reconstruction[i-window_size-1:i-1],(1,-1)))[0][0])
        diff = np.float32(data[i] - predval)
        if (diff > maxlevel + maxerror or diff < minlevel - maxerror):
            f_out.write(struct.pack(fmtstring,numbins))
            f_out.write(struct.pack('f',data[i]))
            reconstruction[i] = data[i]
        else:
            if diff > maxlevel:
                bin_idx = numbins - 1
            elif diff < minlevel:
                bin_idx = 0
            else:
                bin_idx = np.digitize(diff,bins)
            if(bin_idx != numbins and bin_idx != 0):
                if(np.abs(diff-bins[bin_idx])>np.abs(diff-bins[bin_idx-1])):
                    bin_idx -= 1
            f_out.write(struct.pack(fmtstring,bin_idx))
            reconstruction[i] = predval + bins[bin_idx]
    f_out.close()
    subprocess.run(['7z','a',args.outfile,tmpfile])
    os.remove(tmpfile)
    # save reconstruction to a file (for comparing later)
    np.save(reconfile,reconstruction)

    # compute the maximum error b/w reconstrution and data and check that it is within maxerror
    maxerror_observed = np.max(np.abs(data-reconstruction))
    print('maxerror_observed',maxerror_observed)
    assert maxerror_observed <= maxerror
    print('Length of time series: ', len(data))
    print('Size of compressed file: ',os.path.getsize(args.outfile), 'bytes')
    print('Reconstruction written to: ',reconfile)
elif args.mode == 'd':
    tmpfile = args.infile+'.tmp'
    # extract 7z archive
    subprocess.run(['7z','e',args.infile])
    f_in = open(tmpfile,'rb')
    # read max error from file
    maxerror = np.float32(struct.unpack('f',f_in.read(4))[0])
    # read length of data
    len_data = struct.unpack('I',f_in.read(4))[0]
    # initialize quantization (with roughly 65535 bins at the start)
    maxlevel = np.float32(65533*maxerror)
    minlevel = np.float32(-65533*maxerror)
    numbins = int((maxlevel-minlevel)/(2*maxerror))+2
    bins = np.linspace(minlevel,maxlevel,numbins,dtype=np.float32)
    fmtstring = 'H' # 16 bit unsigned
    bin_idx_len = 2 # in bytes
    reconstruction = np.zeros(len_data,dtype=np.float32)
    for i in tqdm(range(len_data)):
        if i <= window_size:
            predval = np.float32(0.0)
        else:
            if args.model_update_period != None:
                assert args.model_update_period > window_size+1
                if i%args.model_update_period == 0:
                    X_train, Y_train = generate_data(reconstruction[i-args.model_update_period:i-1], window_size)
                    # predict the diff rather than absolute value
                    Y_train = Y_train-np.reshape(X_train[:,-1],np.shape(Y_train))
                    model.fit(X_train, Y_train, epochs=args.num_epochs, verbose=0)
            predval = reconstruction[i-1] + np.float32(model.predict(np.reshape(reconstruction[i-window_size-1:i-1],(1,-1)))[0][0])
        bin_idx = struct.unpack(fmtstring,f_in.read(bin_idx_len))[0]
        if bin_idx == numbins:
            reconstruction[i] = np.float32(struct.unpack('f',f_in.read(4))[0])
        else:
            reconstruction[i] = predval + bins[bin_idx]
    os.remove(tmpfile)
    # save reconstruction to a file 
    np.save(args.outfile,reconstruction)
    print('Length of time series: ', len_data)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
