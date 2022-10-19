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

class NLMS_predictor:
    def __init__(self, n, mu, nseries, j):
        self.filt = pa.filters.FilterNLMS(n*nseries+j, mu=mu,w="zeros")
        self.n = n
        self.nseries = nseries
        self.j = j
        return
    def predict(self, past, idx):
        if idx > self.n:
            self.filt.adapt(past[self.j + self.nseries*(idx-1)], past[self.nseries*(idx-self.n-1):self.j + self.nseries*(idx-1)])
            return self.filt.predict(past[self.nseries*(idx-self.n):self.j+self.nseries*idx])
        elif 0 < idx <= self.n:
            return past[self.j + self.nseries*(idx-1)]
        else:
            return 0

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
    reconfile = args.outfile+'.recon.npy'
    f_out_bin_idx = []
    f_out_float = []

    f_out_params = open(tmpfile_params,'wb')
    # write shape of array to file
    f_out_params.write(struct.pack('B',data.shape[0])) # nseries
    f_out_params.write(struct.pack('I',data.shape[1])) # length
    maxerror_original = []
    fmtstring = []
    bin_idx_len = []
    max_bin_idx = []
    min_bin_idx = []
    predictor = []
    for j in range(nseries):
        maxerror_original.append(np.float32(maxerror[j]))
        assert maxerror_original[j] > np.finfo(np.float32).resolution
        # reduce maxerror a little bit to make sure that we don't run into numeric precision issues while binning
        maxerror[j] = maxerror_original[j]-np.finfo(np.float32).resolution
        if quantization_bytes[j] not in [1,2]:
            raise RuntimeError('Invalid quantization_bytes - valid values are 1,2')
        if quantization_bytes[j] == 1:
            fmtstring.append('b') # 8 bit signed
            bin_idx_len.append(1) # in bytes
            max_bin_idx.append(127)
            min_bin_idx.append(-127)
        else:
            fmtstring.append('h') # 16 bit signed
            bin_idx_len.append(2) # in bytes
            max_bin_idx.append(32767)
            min_bin_idx.append(-32767)
        # initialize predictors
        mu[j] = np.float32(mu[j])
        predictor.append(NLMS_predictor(n_NLMS[j], mu[j], nseries, j))
        # write max error to file (needed during decompression)
        f_out_params.write(struct.pack('f',maxerror[j]))
        # write n to file
        f_out_params.write(struct.pack('I',n_NLMS[j]))
        # write mu to file
        f_out_params.write(struct.pack('f',mu[j]))
        # write num quantization bytes to file
        f_out_params.write(struct.pack('B',quantization_bytes[j]))
        f_out_bin_idx.append(open(tmpfile_bin_idx[j],'wb'))
        f_out_float.append(open(tmpfile_float[j],'wb'))

    # flattened reconstruction array to speed up processing
    reconstruction = np.zeros(np.size(data),dtype=np.float32)
    for i in tqdm(range(data.shape[1])):
        for j in range(nseries):
            predval = np.float32(predictor[j].predict(reconstruction,i))
            diff = np.float32(data[j][i] - predval)
            bin_idx = int(round(diff/(2*maxerror[j])))
            if min_bin_idx[j] <= bin_idx <= max_bin_idx[j]:
                reconstruction[j+nseries*i] = predval + np.float32(bin_idx*2*maxerror[j])
                # check if numeric precision issues present, if yes, just store original data as it is
                if np.abs(reconstruction[j+nseries*i]-data[j][i]) <= maxerror_original[j]:
                    f_out_bin_idx[j].write(struct.pack(fmtstring[j],bin_idx))
                    continue
            f_out_bin_idx[j].write(struct.pack(fmtstring[j],min_bin_idx[j]-1))
            f_out_float[j].write(struct.pack('f',data[j][i]))
            reconstruction[j+nseries*i] = data[j][i]
    f_out_params.close()
    for j in range(nseries):
        f_out_bin_idx[j].close()
        f_out_float[j].close()
    # create tar archive
    tar_archive_name = args.outfile+'.tar'
    with tarfile.open(tar_archive_name, "w:") as tar_handle:
        tar_handle.add(tmpfile_params,arcname=os.path.basename(tmpfile_params))
        for j in range(nseries):
            tar_handle.add(tmpfile_bin_idx[j],arcname=os.path.basename(tmpfile_bin_idx[j]))
            tar_handle.add(tmpfile_float[j],arcname=os.path.basename(tmpfile_float[j]))
    # apply BSC compression
    subprocess.run([BSC_PATH,'e',tar_archive_name,args.outfile,'-b64p','-e2'])
    reconstruction = np.reshape(reconstruction, (nseries,-1),order='F')
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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_handle, tmpdir)
    tmpfile_params = tmpdir+'params'
    f_in_params = open(tmpfile_params,'rb')
    # read shape of data
    nseries = struct.unpack('B',f_in_params.read(1))[0]
    len_data = struct.unpack('I',f_in_params.read(4))[0]

    tmpfile_bin_idx = [tmpdir+'bin_idx'+'.'+str(j) for j in range(nseries)]
    tmpfile_float = [tmpdir+'float'+'.'+str(j) for j in range(nseries)]

    maxerror = []
    n_nlms = []
    mu_nlms = []
    quantization_bytes = []
    fmtstring = []
    bin_idx_len = []
    max_bin_idx = []
    min_bin_idx = []
    predictor = []
    f_in_bin_idx = []
    f_in_float = []
    for j in range(nseries):
        f_in_bin_idx.append(open(tmpfile_bin_idx[j],'rb'))
        f_in_float.append(open(tmpfile_float[j],'rb'))
        # read max error from file
        maxerror.append(np.float32(struct.unpack('f',f_in_params.read(4))[0]))
        # read n from file
        n_nlms.append(struct.unpack('I',f_in_params.read(4))[0])
        # read mu from file
        mu_nlms.append(np.float32(struct.unpack('f',f_in_params.read(4))[0]))
        # read quantization_bytes from file
        quantization_bytes.append(struct.unpack('B',f_in_params.read(1))[0])

        if quantization_bytes[j] == 1:
            fmtstring.append('b') # 8 bit signed
            bin_idx_len.append(1) # in bytes
            max_bin_idx.append(127)
            min_bin_idx.append(-127)
        elif quantization_bytes[j] == 2:
            fmtstring.append('h') # 16 bit signed
            bin_idx_len.append(2) # in bytes
            max_bin_idx.append(32767)
            min_bin_idx.append(-32767)
        else:
            raise RuntimeError("Invalid value of quantization_bytes encountered")
        # initialize predictor
        predictor.append(NLMS_predictor(n_nlms[j], mu_nlms[j], nseries,j))

    # flattened reconstruction array to speed up processing
    reconstruction = np.zeros(nseries*len_data,dtype=np.float32)
    for i in tqdm(range(len_data)):
        for j in range(nseries):
            predval = np.float32(predictor[j].predict(reconstruction,i))
            bin_idx = struct.unpack(fmtstring[j],f_in_bin_idx[j].read(bin_idx_len[j]))[0]
            if bin_idx == min_bin_idx[j]-1:
                reconstruction[j+nseries*i] = np.float32(struct.unpack('f',f_in_float[j].read(4))[0])
            else:
                reconstruction[j+nseries*i] = predval + np.float32(2*maxerror[j]*bin_idx)
    reconstruction = np.reshape(reconstruction,(nseries,-1),order='F')
    # save reconstruction to a file
    np.save(args.outfile,reconstruction)
    print('Shape of time series:', np.shape(reconstruction))
    shutil.rmtree(tmpdir)
    os.remove(tar_archive_name)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
