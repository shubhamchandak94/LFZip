#!/usr/bin/env python3
# call with PYTHONHASHSEED=0 to ensure determinism
import struct
import os
import subprocess
import argparse
import numpy as np
import tarfile
import shutil

BSC_PATH = os.path.dirname(os.path.realpath(__file__))+'/libbsc/bsc'
if not os.path.exists(BSC_PATH):
    BSC_PATH = 'bsc'

NLMS_PATH = os.path.dirname(os.path.realpath(__file__))+'/nlms_helper.out'

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--mode','-m', action='store', dest='mode',
                    help='c or d (compress/decompress)', required=True)
parser.add_argument('--infile','-i', action='store', dest='infile', help = 'infile .npy/.bsc', type = str, required=True)
parser.add_argument('--outfile','-o', action='store', dest='outfile', help = 'outfile .bsc/.npy', type = str, required=True)
parser.add_argument('--NLMS_order','-n', action='store', nargs = '+', dest='n', help = 'order of NLMS filter for compression (default 32) - single value or one per variable', type = int, default = [32])
parser.add_argument('--mu', action='store', nargs = '+', dest='mu', help = 'learning rate of NLMS for compression (default 0.5) - single value or one per variable', type = float, default = [0.5])
parser.add_argument('--absolute_error','-a', action='store', nargs='+', dest='maxerror', help = 'max allowed error for compression - single value or one per variable', type=float)
parser.add_argument('--quantization_bytes','-q', action='store', nargs='+', dest='quantization_bytes', help = 'number of bytes used to encode quantized error - decides number of quantization levels. Valid values are 1, 2 (default: 2) - single value or one per variable', type=int, default = [2])

args = parser.parse_args()

if args.mode == 'c':
    # read file
    data = np.load(args.infile)
    assert data.dtype == 'float32'
    if data.ndim == 1:
        nseries = 1
        data = np.reshape(data,(1,-1))
    else:
        nseries = data.shape[0]
    assert 1 <= nseries < 256

    maxerror = args.maxerror
    if maxerror == None:
        raise RuntimeError('maxerror not specified for mode c')
    elif len(maxerror) == 1:
        maxerror = maxerror*nseries
    elif len(maxerror) != nseries:
        raise RuntimeError('Invalid length of maxerror argument (expected 1 or nseries)')

    quantization_bytes = args.quantization_bytes
    if len(quantization_bytes) != 1 and len(quantization_bytes) != nseries:
        raise RuntimeError('Invalid length of quantization_bytes argument (expected 1 or nseries)')
    elif len(quantization_bytes) == 1:
        quantization_bytes = quantization_bytes*nseries

    n_NLMS = args.n
    if len(n_NLMS) != 1 and len(n_NLMS) != nseries:
        raise RuntimeError('Invalid length of n argument (expected 1 or nseries)')
    elif len(n_NLMS) == 1:
        n_NLMS = n_NLMS*nseries

    mu = args.mu
    if len(mu) != 1 and len(mu) != nseries:
        raise RuntimeError('Invalid length of mu argument (expected 1 or nseries)')
    elif len(mu) == 1:
        mu = mu*nseries

    tmpdir = args.outfile+'.tmp.dir/'
    os.makedirs(tmpdir, exist_ok = True)
    tmpfile_bin_idx = [tmpdir+'bin_idx'+'.'+str(j) for j in range(nseries)]
    tmpfile_float = [tmpdir+'float'+'.'+str(j) for j in range(nseries)]
    tmpfile_params = tmpdir+'params'
    tmpfile_params_maxerror_original = tmpdir+'params.maxerror_original'
    # to write maxerror_original needed by C++ nlms helper but not by decompressor
    reconfile = args.outfile+'.recon.npy'

    f_out_params = open(tmpfile_params,'wb')
    f_out_params_maxerror_original = open(tmpfile_params_maxerror_original,'wb')
    # write shape of array to file
    f_out_params.write(struct.pack('B',data.shape[0])) # nseries
    f_out_params.write(struct.pack('I',data.shape[1])) # length
    maxerror_original = []
    for j in range(nseries):
        maxerror_original.append(np.float32(maxerror[j]))
        assert maxerror_original[j] > np.finfo(np.float32).resolution
        # reduce maxerror a little bit to make sure that we don't run into numeric precision issues while binning
        maxerror[j] = maxerror_original[j]-np.finfo(np.float32).resolution
        if quantization_bytes[j] not in [1,2]:
            raise RuntimeError('Invalid quantization_bytes - valid values are 1,2')
        mu[j] = np.float32(mu[j])
        # write max error to file (needed during decompression)
        f_out_params.write(struct.pack('f',maxerror[j]))
        f_out_params_maxerror_original.write(struct.pack('f',maxerror_original[j]))
        # write n to file
        f_out_params.write(struct.pack('I',n_NLMS[j]))
        # write mu to file
        f_out_params.write(struct.pack('f',mu[j]))
        # write num quantization bytes to file
        f_out_params.write(struct.pack('B',quantization_bytes[j]))

    f_out_params.close()
    f_out_params_maxerror_original.close()
    # call c++ executable to perform the compression
    # and write reconstruction to temporary file.
    tmpfile_data = tmpdir+'data.bin'
    tmpfile_recon = tmpdir+'recon.bin'
    with open(tmpfile_data, 'wb') as f:
        f.write(data.tobytes('F'))
    subprocess.run(NLMS_PATH+ ' c ' + tmpdir, shell=True)
    # read reconstruction array from temporary file
    with open(tmpfile_recon, 'rb') as f:
        reconstruction = np.fromfile(f, dtype=np.float32)
        assert reconstruction.size == data.size
    # remove temporary reconstruction file generated by c++ executable
    os.remove(tmpfile_data)
    os.remove(tmpfile_recon)
    os.remove(tmpfile_params_maxerror_original)
    # create tar archive
    tar_archive_name = args.outfile+'.tar'
    with tarfile.open(tar_archive_name, "w:") as tar_handle:
        tar_handle.add(tmpfile_params,arcname=os.path.basename(tmpfile_params))
        for j in range(nseries):
            tar_handle.add(tmpfile_bin_idx[j],arcname=os.path.basename(tmpfile_bin_idx[j]))
            tar_handle.add(tmpfile_float[j],arcname=os.path.basename(tmpfile_float[j]))
    # apply BSC compression
    subprocess.run([BSC_PATH,'e',tar_archive_name,args.outfile,'-b64p','-e2'])
    reconstruction = np.reshape(reconstruction, (nseries,-1), order='F')
    # save reconstruction to a file (for comparing later)
    np.save(reconfile,reconstruction)
    # compute the maximum error b/w reconstrution and data and check that it is within maxerror
    for j in range(nseries):
        print('j:',j)
        maxerror_observed = np.max(np.abs(data[j,:]-reconstruction[j,:]))
        RMSE = np.sqrt(np.mean((data[j,:]-reconstruction[j,:])**2))
        MAE = np.mean(np.abs(data[j,:]-reconstruction[j,:]))
        print('maxerror_observed:',maxerror_observed)
        print('RMSE:',RMSE)
        print('MAE:',MAE)
        assert maxerror_observed <= maxerror_original[j]
    print('Shape of time series:', np.shape(data))
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
    # call c++ executable to perform the decompression
    # and write reconstruction to temporary file.
    subprocess.run(NLMS_PATH + ' d ' + tmpdir, shell=True)
    tmpfile_params = tmpdir+'params'
    f_in_params = open(tmpfile_params,'rb')
    # read shape of data
    nseries = struct.unpack('B',f_in_params.read(1))[0]
    len_data = struct.unpack('I',f_in_params.read(4))[0]

    f_in_params.close()

    # read reconstrution from temporary file into numpy array
    tmpfile_recon = tmpdir+'recon.bin'
    with open(tmpfile_recon, 'rb') as f:
        reconstruction = np.fromfile(f, dtype=np.float32)
        assert reconstruction.size == nseries*len_data
    reconstruction = np.reshape(reconstruction,(nseries,-1),order='F')
    # save reconstruction to a file
    np.save(args.outfile,reconstruction)
    print('Shape of time series:', np.shape(reconstruction))
    shutil.rmtree(tmpdir)
    os.remove(tar_archive_name)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
