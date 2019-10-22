# call with PYTHONHASHSEED=0 to ensure determinism
import numpy as np
import struct
import padasip as pa
import os
import subprocess
import argparse
import numpy as np
from tqdm import tqdm
import tarfile
import shutil

BSC_PATH = os.path.dirname(os.path.realpath(__file__))+'/libbsc/bsc'

def interpolate(diff_X, Y):
    recon = np.zeros(int(np.sum(diff_X)+1), dtype=np.float32)
    recon[0] = Y[0]
    curr_X = 0
    for i in range(1,len(Y)):
        prev_X = curr_X
        curr_X += diff_X[i-1]
        recon[curr_X] = Y[i]
        for j in range(prev_X+1,curr_X):
            recon[j] = recon[prev_X] + (recon[curr_X]-recon[prev_X])*np.float32(j-prev_X)/np.float32(curr_X-prev_X)
    return recon

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--mode','-m', action='store', dest='mode',
                    help='c or d (compress/decompress)', required=True)
parser.add_argument('--infile','-i', action='store', dest='infile', help = 'infile .npy/.bsc', type = str, required=True)
parser.add_argument('--outfile','-o', action='store', dest='outfile', help = 'outfile .bsc/.npy', type = str, required=True)
parser.add_argument('--absolute_error','-a', action='store', dest='maxerror', help = 'max allowed error for compression', type=float)

args = parser.parse_args()

if args.mode == 'c':
    if args.maxerror == None:
        raise RuntimeError('maxerror not specified for mode c')

    # read file
    data = np.load(args.infile)
    assert data.dtype == 'float32'
    if data.ndim != 1:
        if data.ndim == 2 and data.shape[0] == 1:
            data = np.reshpape(data, (-1))
    assert data.ndim == 1

    maxerror_original = np.float32(args.maxerror)
    assert maxerror_original > np.finfo(np.float32).resolution
    # reduce maxerror a little bit to make sure that we don't run into numeric precision issues
    maxerror = maxerror_original - np.finfo(np.float32).resolution

    tmpdir = args.outfile+'.tmp.dir/'
    os.makedirs(tmpdir, exist_ok = True)
    tmpfile_X = tmpdir + 'X'
    tmpfile_Y = tmpdir + 'Y'
    reconfile = args.outfile+'.recon.npy'

    X = [0]
    Y = [data[0]]
    archived_idx = 0
    held_idx = 0
    upper_slope = np.float32(np.inf)
    lower_slope = np.float32(-np.inf)
    while True:
        if held_idx == len(data)-1:
            X.append(held_idx)
            Y.append(data[held_idx])
            break
        new_idx = held_idx + 1
        slope_new_point = np.float32(data[new_idx]-data[archived_idx])/np.float32(new_idx-archived_idx)
        if lower_slope < slope_new_point < upper_slope:
            upper_slope = np.minimum(np.float32(data[new_idx]+maxerror-data[archived_idx])/np.float32(new_idx-archived_idx),upper_slope)
            lower_slope = np.maximum(np.float32(data[new_idx]-maxerror-data[archived_idx])/np.float32(new_idx-archived_idx),lower_slope)
            held_idx += 1
        else:
            X.append(held_idx)
            Y.append(data[held_idx])
            archived_idx = held_idx
            upper_slope = np.float32(np.inf)
            lower_slope = np.float32(-np.inf)
    
    # write to file
    X = np.array(X, dtype=np.uint32)
    diff_X = np.diff(X)
    Y = np.array(Y, dtype=np.float32)
    reconstruction = interpolate(diff_X,Y)
    f_X = open(tmpfile_X,'wb')
    f_Y = open(tmpfile_Y,'wb')
    for val in diff_X:
        f_X.write(struct.pack('I',val))
    f_X.close()
    for val in Y:
        f_Y.write(struct.pack('f',val))
    f_Y.close()
    # create tar archive
    tar_archive_name = args.outfile+'.tar' 
    with tarfile.open(tar_archive_name, "w:") as tar_handle:
        tar_handle.add(tmpfile_X,arcname=os.path.basename(tmpfile_X))
        tar_handle.add(tmpfile_Y,arcname=os.path.basename(tmpfile_Y))
    # apply BSC compression
    subprocess.run([BSC_PATH,'e',tar_archive_name,args.outfile,'-b64p','-e2'])
    # save reconstruction to a file (for comparing later)
    np.save(reconfile,reconstruction)
    # compute the maximum error b/w reconstrution and data and check that it is within maxerror
    maxerror_observed = np.max(np.abs(data-reconstruction))
    RMSE = np.sqrt(np.mean((data-reconstruction)**2))
    MAE = np.mean(np.abs(data-reconstruction))
    print('maxerror_observed:',maxerror_observed)
    print('RMSE:',RMSE)
    print('MAE:',MAE)
    assert maxerror_observed <= maxerror_original
    print('Length of time series:', len(data))
    print('Number of points retained:', len(Y))
    print('Size of compressed file:',os.path.getsize(args.outfile), 'bytes')
    print('Reconstruction written to:',reconfile)
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
    tmpfile_X = tmpdir+'X'
    tmpfile_Y = tmpdir+'Y'
    with open(tmpfile_X,'rb') as f_X:
        diff_X = np.array([val[0] for val in struct.iter_unpack('I',f_X.read())],dtype=np.uint32)
    with open(tmpfile_Y,'rb') as f_Y:
        Y = np.array(list(struct.iter_unpack('f',f_Y.read())),dtype=np.float32)
    reconstruction = interpolate(diff_X,Y)
    # save reconstruction to a file 
    np.save(args.outfile,reconstruction)
    print('Shape of time series:', np.shape(reconstruction))
    shutil.rmtree(tmpdir)
    os.remove(tar_archive_name)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
