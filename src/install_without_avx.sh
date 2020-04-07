# use with python 3.6 (conda create --name snakes python=3.6 \ conda activate snakes)
pip install --upgrade pip
pip install \
    tensorflow==1.5\
    tqdm
conda install \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      scikit-learn \
      six \
      theano
pip install padasip sklearn keras==2.1.5
#libbsc for entropy coding
(cd libbsc && make CC=g++-9)
# compile nlms_helper for actual compression with nlms
g++ nlms_helper.cpp -std=c++11 -o nlms_helper.out -Wall -O3 -march=native
