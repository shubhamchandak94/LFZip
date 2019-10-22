# DeepZip

## Floating point time series compression
### Download and install dependencies
Download:
```
git clone -b floating_lossy https://github.com/shubhamchandak94/DeepZip.git
```
To set up virtual environment and dependencies:
```
cd DeepZip/src/
python3 -m venv env
source env/bin/activate
./install.sh
```

For processors without AVX instructions (e.g., Intel Pentium/Celeron) used in the latest Tensorflow package, do the following instead:
```
cd DeepZip/src/
conda create --name no_avx_env python=3.6
conda activate no_avx_env
./install_without_avx.sh
```
### NLMS model
#### Compression/Decompression:
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
                        order of NLMS filter for compression (default 4) -
                        single value or one per time series
  --mu MU [MU ...]      learning rate of NLMS for compression (default 0.5) -
                        single value or one per time series
  --absolute_error MAXERROR [MAXERROR ...], -a MAXERROR [MAXERROR ...]
                        max allowed error for compression - single value or
                        one per time series
  --quantization_bytes QUANTIZATION_BYTES [QUANTIZATION_BYTES ...], -q QUANTIZATION_BYTES [QUANTIZATION_BYTES ...]
                        number of bytes used to encode quantized error -
                        decides number of quantization levels. Valid values
                        are 1, 2 (default: 2) - single value or one per time
                        series
```
The reconstructed time series is also generated as a byproduct and stored as `compressed_file.bsc.recon.npy`.

### Neural network model
#### Training a model
First select the appropriate function from `models.py`, e.g., `FC_siemens` or `biGRU_siemens`. Then call
```
python3 nn_trainer.py -train training_data.npy -val validation_data.npy -model_file saved_model.h5 \
-model_name model_name -model_params model_params [-lr lr -noise noise -epochs epochs]
```
with the parameters:
```
model_name:   (str) name of model (function name from models.py)
model_params: space separated list of parameters to model_name
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
  --lr LR               learning rate for Adam
  --epochs NUM_EPOCHS   number of epochs to train
```
The `CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0` environment variables are set to ensure that the decompression works precisely the same as the compression and generates the correct reconstruction.

### Critical aperture compression

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
