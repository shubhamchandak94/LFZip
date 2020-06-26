# LFZip

## Multivariate floating-point time series lossy compression under maximum error distortion

[![Build Status](https://travis-ci.org/shubhamchandak94/LFZip.svg?branch=master)](https://travis-ci.org/shubhamchandak94/LFZip)

#### Arxiv: https://arxiv.org/abs/1911.00208

##### See update [below](#update-on-nlms-order) on selecting value of NLMS order for LFZip 

### Download and install dependencies

#### Using Conda (Linux/MacOSX): 
LFZip (NLMS prediction mode) is now available on conda through the conda-forge channel. For the neural network prediction mode or to run from source, see the next section.
```
conda create --name lfzip_env
conda activate lfzip_env
conda config --add channels conda-forge
conda install lfzip
```
After the installation, LFZip (NLMS) can be run using the command `lfzip-nlms`. To install LFZip in a conda virtual environment, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

#### From source (Linux/MacOSX):
Download:
```
git clone https://github.com/shubhamchandak94/LFZip.git
```
To set up virtual environment and dependencies (on Linux):
```
cd LFZip/src/
python3 -m venv env
source env/bin/activate
./install.sh
```

On macOS, you need gcc compiler for running BSC which is the entropy coder used in LFZip. For this, install gcc@9 using brew as follows:
```
brew update
brew install gcc@9
```
and then replace the last statement of the Linux instructions with
```
./install_macos.sh
```

For processors without AVX instructions (e.g., Intel Pentium/Celeron) used in the latest Tensorflow package, do the following instead (requires a working conda installation):
```
cd LFZip/src/
conda create --name no_avx_env python=3.6
conda activate no_avx_env
./install_without_avx.sh
```

### General comments
- Note that LFZip (NLMS), LFZip (NN) and CA (critical aperture) expect the input to be in numpy array (.npy) format and support only float32 arrays.
- LFZip (NLMS) additionally supports multivariate time series with at most 256 variables, where the input is a numpy array of shape `(k,T)` where `k` is the number of variables and `T` is the length of the time series.
- During compression, the reconstructed time series is also generated as a byproduct and stored as `compressed_file.bsc.recon.npy`. This can be used to verify the correctness of the compression-decompression pipeline.
- **Examples** are shown after the usages below [[link](#examples)].

### LFZip (NLMS)
#### Compression/Decompression:
If installed using conda, replace `python3 nlms_compress.py` by `lfzip-nlms`.
```
python3 nlms_compress.py [-h] --mode MODE --infile INFILE --outfile OUTFILE
                        [--NLMS_order N [N ...]] [--mu MU [MU ...]]
                        [--absolute_error MAXERROR [MAXERROR ...]]
                        [--quantization_bytes QUANTIZATION_BYTES [QUANTIZATION_BYTES ...]]
```
with the parameters:
```
  -h, --help            show this help message and exit
  --mode MODE, -m MODE  c or d (compress/decompress)
  --infile INFILE, -i INFILE
                        infile .npy/.bsc
  --outfile OUTFILE, -o OUTFILE
                        outfile .bsc/.npy
  --NLMS_order N [N ...], -n N [N ...]
                        order of NLMS filter for compression (default 32) -
                        single value or one per variable
  --mu MU [MU ...]      learning rate of NLMS for compression (default 0.5) -
                        single value or one per variable
  --absolute_error MAXERROR [MAXERROR ...], -a MAXERROR [MAXERROR ...]
                        max allowed error for compression - single value or
                        one per variable
  --quantization_bytes QUANTIZATION_BYTES [QUANTIZATION_BYTES ...], -q QUANTIZATION_BYTES [QUANTIZATION_BYTES ...]
                        number of bytes used to encode quantized error -
                        decides number of quantization levels. Valid values
                        are 1, 2 (default: 2) - single value or one per variable
```
Note that `nlms_compress_python.py` is an older and slower version with a similar interface
but with the core NLMS compression code written in Python instead of C++.

##### Update on NLMS order
While the default order for NLMS is 32, we have found that for certain dataset, the optimal order is 0 (i.e., the prediction step is skipped). We recommend that the user try out both values using the `-n` flag for a given data source before selecting the order. We are currently working on making this process automatic.

### LFZip (NN)
#### Training a model
First select the appropriate function from `models.py`, e.g., `FC` or `biGRU`. Then call
```
python3 nn_trainer.py -train training_data.npy -val validation_data.npy -model_file saved_model.h5 \
-model_name model_name -model_params model_params [-lr lr -noise noise -epochs epochs]
```
with the parameters:
```
model_name:   (str) name of model (function name from models.py)
model_params: space separated list of parameters to the function model_name
lr:           (float) learning rate (default 1e-3 for Adam)
noise:        (float) noise added to input during training (uniform[-noise,noise]), default 0
epochs:       (int) number of epochs to train (0 means store random model)
```
#### Compression/Decompression:
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 nn_compress.py [-h] --mode MODE --infile INFILE --outfile OUTFILE
                      [--absolute_error MAXERROR] --model_file MODEL_FILE
                      [--quantization_bytes QUANTIZATION_BYTES]
                      [--model_update_period MODEL_UPDATE_PERIOD] [--lr LR]
                      [--epochs NUM_EPOCHS]
```
with the parameters:
```
  -h, --help            show this help message and exit
  --mode MODE, -m MODE  c or d (compress/decompress)
  --infile INFILE, -i INFILE
                        infile .npy/bsc
  --outfile OUTFILE, -o OUTFILE
                        outfile .bsc/.npy
  --absolute_error MAXERROR, -a MAXERROR
                        max allowed error for compression
  --model_file MODEL_FILE
                        model file
  --quantization_bytes QUANTIZATION_BYTES, -q QUANTIZATION_BYTES
                        number of bytes used to encode quantized error -
                        decides number of quantization levels. Valid values
                        are 1, 2 (deafult: 2)
  --model_update_period MODEL_UPDATE_PERIOD
                        train model (both during compression & decompression)
                        after seeing these many symbols (default: never train)
  --lr LR               learning rate for Adam when model update used
  --epochs NUM_EPOCHS   number of epochs to train when model update used
```
The `CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0` environment variables are set to ensure that the decompression works precisely the same as the compression and generates the correct reconstruction.

### Critical aperture (CA)
WARNING: in some cases, maxerror constraint can be slightly violated (~1e-5) due to numerical precision issues (only for the CA implementation).
#### Compression/Decompression:
```
python3 ca_compress.py [-h] --mode MODE --infile INFILE --outfile OUTFILE
                      [--absolute_error MAXERROR]

Input

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE, -m MODE  c or d (compress/decompress)
  --infile INFILE, -i INFILE
                        infile .npy/.bsc
  --outfile OUTFILE, -o OUTFILE
                        outfile .bsc/.npy
  --absolute_error MAXERROR, -a MAXERROR
                        max allowed error for compression
```


### Other helpful scripts
- `data/dat_to_np.py`: convert a .dat file (with 1 time series value in plaintext per line) to .npy file
- `data/npy_to_bin.py`: convert a .npy file to binary file used as input to SZ
- `data/bin_to_npy.py`: convert a .bin file to .npy file

### Examples

#### LFZip (NLMS)
If installed using conda, replace `python nlms_compress.py` by `lfzip-nlms`. See also [update](#update-on-nlms-order) above on selecting the NLMS order.

Compression:
```
python nlms_compress.py -m c -i ../data/evaluation_datasets/dna/nanopore_test.npy -o nanopore_test_compressed.bsc -a 0.01
```
Decompression:
```
python nlms_compress.py -m d -i nanopore_test_compressed.bsc -o nanopore_test.decompressed.npy
```
Verification:
```
cmp nanopore_test.decompressed.npy nanopore_test_compressed.bsc.recon.npy
```

#### LFZip (NN)
Training a fully connected model (`FC` in `models.py`) with `input_dim = 32`, `num_hidden_layers = 4`, `hidden_layer_size = 128` for 5 epochs with uniform noise in \[-0.05,0.05\] added to the input.
```
python nn_trainer.py -train ../data/evaluation_datasets/dna/nanopore_train.npy -val ../data/evaluation_datasets/dna/nanopore_val.npy -model_name FC -model_params 32 4 128 -model_file nanopore_trained.h5 -noise 0.05 -epochs 5
```
Compression:
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -m c -i ../data/evaluation_datasets/dna/nanopore_test.npy -o nanopore_test_compressed.bsc -a 0.01 --model_file nanopore_trained.h5
```
Decompression:
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -m d -i nanopore_test_compressed.bsc -o nanopore_test.decompressed.npy --model_file nanopore_trained.h5
```
Verification:
```
cmp nanopore_test.decompressed.npy nanopore_test_compressed.bsc.recon.npy
```


#### CA
Compression:
```
python ca_compress.py -m c -i ../data/evaluation_datasets/dna/nanopore_test.npy -o nanopore_test_compressed.bsc -a 0.01
```
Decompression:
```
python ca_compress.py -m d -i nanopore_test_compressed.bsc -o nanopore_test.decompressed.npy
```
Verification:
```
cmp nanopore_test.decompressed.npy nanopore_test_compressed.bsc.recon.npy
```
