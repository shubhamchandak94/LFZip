# LFZip

## Multivariate floating-point time series lossy compression under maximum error distortion
### Download and install dependencies
Download:
```
git clone https://github.com/shubhamchandak94/LFZip.git
```
To set up virtual environment and dependencies:
```
cd LFZip/src/
python3 -m venv env
source env/bin/activate
./install.sh
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

### LFZip (NLMS)
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
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -m c -i ../data/evaluation_datasets/dna/nanopore_test.npy -o nanopore_test.bsc -a 0.01 --model_file nanopore_trained.h5 
```
Decompression:
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -m d -i nanopore_test.bsc -o nanopore_test.decompressed.npy --model_file nanopore_trained.h5
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
