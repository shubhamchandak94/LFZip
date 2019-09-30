# DeepZip

## Floating point time series compression
### Download and install dependencies
Download:
```
git clone -b floating_lossy https://github.com/shubhamchandak94/DeepZip.git
```
To set up virtual environment and dependencies:
```
sudo apt-get install p7zip-full
cd DeepZip/src/
python3 -m venv env
source env/bin/activate
./install.sh
```

For processors without AVX instructions (e.g., Intel Pentium/Celeron) used in the latest Tensorflow package, do the following instead:
```
sudo apt-get install p7zip-full
cd DeepZip/src/
conda create --name no_avx_env python=3.6
conda activate no_avx_env
./install_without_avx.sh
```
### NLMS model
#### Compression:
```
python3 nlms_compress.py -mode c -infile file_to_be_compressed.npy -outfile compressed_file.7z \
                         [-n n] [-mu mu] [-maxerror maxerror]
```
with the parameters: 
```
n:        (int) order of NLMS filter, default 4
mu:       (float) learning rate of NLMS filter, default 0.5
maxerror: (float) maximum allowed error in reconstruction
```
The reconstructed time series is also generated as a byproduct and stored as `compressed_file.7z.recon.npy`.
#### Decompression
```
python3 nlms_compress.py -mode d -infile compressed_file.7z -outfile decompressed_file.npy
```

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
#### Compression 
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 nn_compress.py -mode c -infile file_to_be_compressed.npy \
-outfile compressed_file.7z -model_file saved_model.h5 -maxerror maxerror\
[-model_update_period model_update_period -lr lr -epochs epochs]
```
with the parameters:
```
maxerror:             (float) maximum allowed error in reconstruction
model_update_period:  (int) frequency of updating model, default: never
lr:                   (float) learning rate of Adam when updating model, default 1e-3
epochs:               (int) number of training epochs in each model update, deafult 1
```
The `CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0` environment variables are set to ensure that the decompression works precisely the same as the compression and generates the correct reconstruction.
#### Decompression
```
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 nn_compress.py -mode d -infile compressed_file.7z \
-outfile decompressed_file.npy -model_file saved_model.h5
```

### Other helpful scripts
`data/dat_to_np.py`: convert a .dat file (with 1 time series value in plaintext per line) to .npy file
