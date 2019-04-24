# DeepZip

## Floating point time series compression
To set up virtual environment and dependencies:
```
sudo apt-get install p7zip-full
cd src/
python3 -m venv env
source env/bin/activate
./install.sh
```

### Relevant scripts
The instructions for each script are available in the help for the corresponding file.
1. data/dat_to_np.py: convert a .dat file (with 1 time series value in plaintext per line) to .npy file
2. src/models.py: neural network models in the form of functions
3. src/nlms_compress.py: compression and decompression using the NLMS prediction setup
4. src/nn_trainer.py: train neural networks
5. src/nn_compress.py: compress/decompress using trained NN models


## ----- OLD ------
## Description
DNA_compression using neural networks

This branch is the non-GPU implementation of deepzip.
*WARNING* The training is significantly slower on a CPU and the code will take significant longer than reporterd in the manuscript. 
The code has been tested on macOS Mojave. Please file an issue if there are problems.

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8

A simple way to install and run is to use the docker files provided:

```bash
cd docker
make bash BACKEND=tensorflow DATA=/path/to/data/
```

## Code
To run a compression experiment: 

### Data Preparation
1. Place all the data to be compressed in data/files_to_be_compressed
2. Run the parser 

```bash
cd data
./run_parser.sh
```

### Running models
1. All the models are listed in models.py
2. Pick a model, to run compression experiment on all the data files in the data/files_to_be_compressed directory

```
cd src
./run_experiments.sh biLSTM
```
