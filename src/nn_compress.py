# steps for reproducibility (also need to call with CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0)
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

import numpy as np
import argparse
import contextlib
import json
import struct
import tempfile
import shutil
import subprocess
import os
from tqdm import tqdm
import models
import tarfile
import shutil

BSC_PATH = os.path.dirname(os.path.realpath(__file__))+'/libbsc/bsc'

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--mode','-m', action='store', dest='mode',
        help='c or d (compress/decompress)', required=True)
parser.add_argument('--infile','-i', action='store', dest='infile', help = 'infile .npy/bsc', type = str, required=True)
parser.add_argument('--outfile','-o', action='store', dest='outfile', help = 'outfile .bsc/.npy', type = str, required=True)
parser.add_argument('--absolute_error','-a', action='store', dest='maxerror', help = 'max allowed error for compression', type=float)
parser.add_argument('--model_file', action='store', dest='model_file',
                    help='model file', required=True)
parser.add_argument('--quantization_bytes','-q', action='store', dest='quantization_bytes', help = 'number of bytes used to encode quantized error - decides number of quantization levels. Valid values are 1, 2 (deafult: 2)', type = int, default = 2)
parser.add_argument('--model_update_period', action='store', dest='model_update_period', help = 'train model (both during compression & decompression) after seeing these many symbols (default: never train)', type = int)
parser.add_argument('--lr', action='store', type=float,
	dest='lr', default = 1e-3,
	help='learning rate for Adam')
parser.add_argument('--epochs', action='store',
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


if args.mode == 'c':
    if args.maxerror == None:
        raise RuntimeError('maxerror not specified for mode c')
    
    model_update_flag = False
    if args.model_update_period is not None:
        assert args.model_update_period > window_size+1
        model_update_flag = True
        lr = np.float32(args.lr)
        model_update_period = args.model_update_period
        num_epochs = args.num_epochs
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')

    tmpfile = args.outfile+'.tmp'
    reconfile = args.outfile+'.recon.npy'
    maxerror = np.float32(args.maxerror)
    # read file
    data = np.load(args.infile)
    assert data.dtype == 'float32'
    if data.ndim != 1:
        if data.ndim == 2 and data.shape[0] == 1:
            data = np.reshpape(data, (-1))
    assert data.ndim == 1

    maxerror_original = np.float32(maxerror)
    assert maxerror_original > np.finfo(np.float32).resolution
    # reduce maxerror a little bit to make sure that we don't run into numeric precision issues while binning
    maxerror = maxerror_original - np.finfo(np.float32).resolution
    if args.quantization_bytes not in [1,2]:
        raise RuntimeError('Invalid quantization_bytes - valid values are 1,2')
    if args.quantization_bytes == 1:
        fmtstring = 'b' # 8 bit signed
        bin_idx_len = 1 # in bytes
        max_bin_idx = 127
        min_bin_idx = -127
    else:
        fmtstring = 'h' # 16 bit signed
        bin_idx_len = 2 # in bytes
        max_bin_idx = 32767
        min_bin_idx = -32767

    reconstruction = np.zeros(np.shape(data),dtype=np.float32)

    tmpdir = args.outfile+'.tmp.dir/'
    os.makedirs(tmpdir, exist_ok = True)
    tmpfile_bin_idx = tmpdir+'bin_idx'
    tmpfile_float = tmpdir+'float'
    tmpfile_params = tmpdir+'params'

    f_out_bin_idx = open(tmpfile_bin_idx, 'wb')
    f_out_params = open(tmpfile_params, 'wb')
    f_out_float = open(tmpfile_float, 'wb')

    # write max error to file (needed during decompression)
    f_out_params.write(struct.pack('f',maxerror))
    # write length of array to file
    f_out_params.write(struct.pack('I',len(data)))
    # write flag indicating if model update is used
    f_out_params.write(struct.pack('?',model_update_flag))
    # write num quantization bytes to file
    f_out_params.write(struct.pack('B',args.quantization_bytes))

    # if model update is used, write the associated parameters
    if model_update_flag:
        f_out_params.write(struct.pack('I',model_update_period))
        f_out_params.write(struct.pack('f',lr))
        f_out_params.write(struct.pack('I',num_epochs))
    for i in tqdm(range(len(data))):
        if i > window_size:
            if model_update_flag:
                if i%model_update_period == 0:
                    X_train, Y_train = generate_data(reconstruction[i-model_update_period:i-1], window_size)
                    # predict the diff rather than absolute value
                    Y_train = Y_train-np.reshape(X_train[:,-1],np.shape(Y_train))
                    model.fit(X_train, Y_train, epochs=num_epochs, verbose=0)
            predval = reconstruction[i-1] + np.float32(model.predict(np.reshape(reconstruction[i-window_size-1:i-1],(1,-1)))[0][0])
        else:
            predval = np.float32(0.0)

        diff = np.float32(data[i] - predval)
        bin_idx = int(round(diff/(2*maxerror)))
        if min_bin_idx <= bin_idx <= max_bin_idx:
            reconstruction[i] = predval + np.float32(bin_idx*2*maxerror)
            # check if numeric precision issues present, if yes, just store original data as it is
            if np.abs(reconstruction[i]-data[i]) <= maxerror_original:
                f_out_bin_idx.write(struct.pack(fmtstring,bin_idx))
                continue
        f_out_bin_idx.write(struct.pack(fmtstring,min_bin_idx-1))
        f_out_float.write(struct.pack('f',data[i]))
        reconstruction[i] = data[i]
    f_out_params.close()
    f_out_bin_idx.close()
    f_out_float.close()

    # create tar archive
    tar_archive_name = args.outfile+'.tar'
    with tarfile.open(tar_archive_name, "w:") as tar_handle:
        tar_handle.add(tmpfile_params,arcname=os.path.basename(tmpfile_params))
        tar_handle.add(tmpfile_bin_idx,arcname=os.path.basename(tmpfile_bin_idx))
        tar_handle.add(tmpfile_float,arcname=os.path.basename(tmpfile_float))

    # apply BSC compreession
    subprocess.run([BSC_PATH,'e',tar_archive_name,args.outfile,'-b64p','-e2'])
    # save reconstruction to a file (for comparing later)
    np.save(reconfile,reconstruction)

    # compute the maximum error b/w reconstrution and data and check that it is within maxerror
    maxerror_observed = np.max(np.abs(data-reconstruction))
    RMSE = np.sqrt(np.mean((data-reconstruction)**2))
    MAE = np.mean(np.abs(data-reconstruction))
    print('maxerror_observed',maxerror_observed)
    assert maxerror_observed <= maxerror_original
    print('RMSE:',RMSE)
    print('MAE:',MAE)
    print('Length of time series: ', len(data))
    print('Size of compressed file: ',os.path.getsize(args.outfile), 'bytes')
    print('Reconstruction written to: ',reconfile)
    shutil.rmtree(tmpdir)
    os.remove(tar_archive_name)
elif args.mode == 'd':
    tar_archive_name = args.outfile+'tmp.tar'
    tmpdir = args.outfile+'.tmp.dir/'
    os.makedirs(tmpdir, exist_ok = True)
    # perform BSC decompression
    subprocess.run([BSC_PATH,'d',args.infile,tar_archive_name])
    # untar
    with tarfile.open(tar_archive_name, "r:") as tar_handle:
        tar_handle.extractall(tmpdir)
    tmpfile_params = tmpdir+'params'
    tmpfile_bin_idx = tmpdir+'bin_idx'
    tmpfile_float = tmpdir+'float'
    f_in_params = open(tmpfile_params,'rb')
    f_in_bin_idx = open(tmpfile_bin_idx,'rb')
    f_in_float = open(tmpfile_float,'rb')
    # read max error from file
    maxerror = np.float32(struct.unpack('f',f_in_params.read(4))[0])
    # read length of data
    len_data = struct.unpack('I',f_in_params.read(4))[0]
    # read flag indicating if model update is used
    model_update_flag = bool(struct.unpack('?',f_in_params.read(1))[0])
    # read quantization_bytes from file
    quantization_bytes = struct.unpack('B',f_in_params.read(1))[0]
    if quantization_bytes == 1:
        fmtstring = 'b' # 8 bit signed
        bin_idx_len = 1 # in bytes
        max_bin_idx = 127
        min_bin_idx = -127
    elif quantization_bytes == 2:
        fmtstring = 'h' # 16 bit signed
        bin_idx_len = 2 # in bytes
        max_bin_idx = 32767
        min_bin_idx = -32767
    else:
        raise RuntimeError("Invalid value of quantization_bytes encountered")

    # if model update is used, read the associated parameters
    if model_update_flag:
        model_update_period = struct.unpack('I',f_in.read(4))[0]
        assert model_update_period > window_size+1
        lr = np.float32(struct.unpack('f',f_in.read(4))[0])
        num_epochs = struct.unpack('I',f_in.read(4))[0]
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
    
    reconstruction = np.zeros(len_data,dtype=np.float32)
    for i in tqdm(range(len_data)):
        if i > window_size:
            if model_update_flag:
                if i%model_update_period == 0:
                    X_train, Y_train = generate_data(reconstruction[i-model_update_period:i-1], window_size)
                    # predict the diff rather than absolute value
                    Y_train = Y_train-np.reshape(X_train[:,-1],np.shape(Y_train))
                    model.fit(X_train, Y_train, epochs=num_epochs, verbose=0)
            predval = reconstruction[i-1] + np.float32(model.predict(np.reshape(reconstruction[i-window_size-1:i-1],(1,-1)))[0][0])
        else:
            predval = np.float32(0.0)
        bin_idx = struct.unpack(fmtstring,f_in_bin_idx.read(bin_idx_len))[0]
        if bin_idx == min_bin_idx-1:
            reconstruction[i] = np.float32(struct.unpack('f',f_in_float.read(4))[0])
        else:
            reconstruction[i] = predval + np.float32(2*maxerror*bin_idx)
    # save reconstruction to a file 
    np.save(args.outfile,reconstruction)
    shutil.rmtree(tmpdir)
    os.remove(tar_archive_name)
    print('Length of time series: ', len_data)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
